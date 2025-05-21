import sys
import os
import numpy as np
import os
import logging
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm  # Import tqdm for progress bars
from torchmetrics.classification import JaccardIndex
from torch.utils.data import DataLoader
from project_datasets.LoveDA import LoveDADataset
from torchvision import transforms
import models.deeplabv2 as dlb 
import time
from fvcore.nn import FlopCountAnalysis
# Configurazione del logger
log_dir = "training/logs"
os.makedirs(log_dir, exist_ok=True)  # Crea la cartella dei log se non esiste
log_file = os.path.join(log_dir, "training_DeepLab.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting training...")

#------------------------------------#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency_on_validation(model, val_loader, device, num_runs=100):
    """
    Misura la latenza su un batch reale dal validation set.
    """
    model.eval()
    # Ottieni un batch dal validation set
    inputs, _ = next(iter(val_loader))
    inputs = inputs.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # Misura del tempo
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(inputs)
    end_time = time.time()

    # Calcolo latenza media
    total_time = end_time - start_time
    latency = (total_time / num_runs) * 1000  # Converti in millisecondi
    return latency


""" def calculate_flops_on_validation(model, val_loader):
    # Calcola i FLOPs su un batch reale dal validation set.
    
    # Ottieni un batch dal validation set
    inputs, _ = next(iter(val_loader))

    print(f"Input shape for profiling: {inputs.shape}")

    with torch.no_grad():
        flops = torchprofile.profile_macs(model, inputs)
    return flops """


#------------------------------------#
def squeeze_channel(tensor):
    return tensor.squeeze(0).long()

def calculate_iou(predictions, targets, num_classes):
    """
    Calculate IoU for each class.
    """
    iou = []
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)

        intersection = torch.logical_and(pred_cls, target_cls).sum().item()
        union = torch.logical_or(pred_cls, target_cls).sum().item()

        if union == 0:
            iou.append(float('nan'))  # Ignore classes with no presence
        else:
            iou.append(intersection / union)

    return iou

def mean_iou(predictions, targets, num_classes):
    """
    Compute the mean IoU across all classes.
    """
    ious = calculate_iou(predictions, targets, num_classes)
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    return sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
#------------------------------------#





def rescale_labels(tensor):
    """
    Rescales labels:
      - Class 0 -> -1 (ignore)
      - Classes 1-7 -> 0-6
    """
    tensor = tensor.squeeze(0).long()
    tensor = tensor - 1  # Shift labels down by 1
    tensor[tensor == -1] = -1  # Ensure 0 becomes -1
    return tensor

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


if __name__ == '__main__':
    # Define transformations for images and masks
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 512x512
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize mask
        transforms.PILToTensor(), # Convert mask to tensor
        transforms.Lambda(rescale_labels)  # Remove channel dimension
    ])


    # Dataset paths
    train_dataset = LoveDADataset(
        image_dir="../project_datasets/LoveDA/Train/Urban/images_png",
        mask_dir="../project_datasets/LoveDA/Train/Urban/masks_png",
        transform=img_transforms,
        target_transform=mask_transforms
    )

    val_dataset = LoveDADataset(
        image_dir="../project_datasets/LoveDA/Val/Urban/images_png",
        mask_dir="../project_datasets/LoveDA/Val/Urban/masks_png",
        transform=img_transforms,
        target_transform=mask_transforms
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    num_classes = 7  # Update based on LoveDA

    # Define model
    model = dlb.get_deeplab_v2(num_classes=num_classes)
    
    # Define optimizer and loss function
    # optimizer = optim.SGD(model.optim_parameters(0.0001), momentum=0.9, weight_decay=0.0001) Migliori parametri trovati
    optimizer = optim.SGD(model.optim_parameters(0.0001), momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396]).cuda())

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #iou = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
    num_epochs = 20  # Number of epochs
    init_lr = 0.0001
    max_iter = num_epochs * len(train_loader)  # Total iterations, serve per fast_hist
    best_mIoU = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        #total_miou = 0.0
        total_correct_predictions = 0
        total_elements = 0
        total_confusion_matrix = np.zeros((num_classes, num_classes))

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

        for iter, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)

            #Facciamo update di lr
            """ current_iter = epoch * len(train_loader) + iter  # Current global iteration
            current_lr = poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter=max_iter, power=0.9) """

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            mask = masks != -1
            # Apply the mask to predictions and targets
            masked_preds = preds[mask]
            masked_targets = masks[mask]

            # Compute parameters per accuracy
            correct_predictions = torch.sum(masked_preds == masked_targets).item()
            elements = masked_preds.numel()
            #accuracy = (correct_predictions / total_elements) * 100
            #total_accuracy += accuracy
            total_correct_predictions += correct_predictions
            total_elements += elements

            # Update confusion matrix per andare a calcolare MioU
            total_confusion_matrix += fast_hist(masked_targets.cpu().numpy(), masked_preds.cpu().numpy(), num_classes)
            # Compute Mean IoU
            #batch_miou = iou(masked_preds, masked_targets)
            # Compute IoU
            #batch_miou = mean_iou(preds, masks, num_classes)
            #total_miou += batch_miou.item()
            
            # Loss computation
            loss = criterion(outputs, masks.long())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=running_loss / (train_pbar.n + 1))

        # Calculate per-class IoU and mean IoU    
        per_class_iou_values = per_class_iou(total_confusion_matrix)
        mIoU_training = np.mean(per_class_iou_values) #non più -> avg_miou = total_miou / len(train_loader)
        
        #avg_accuracy = total_accuracy / len(train_loader)
        avg_accuracy = (total_correct_predictions / total_elements) * 100
         
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {running_loss/len(train_loader):.4f}, mIoU: {mIoU_training:.4f}, Average accuracy: {avg_accuracy:.4f}")

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] Training - "
            f"Loss: {running_loss/len(train_loader):.4f}, "
            f"mIoU: {mIoU_training:.4f}, "
            f"Accuracy: {avg_accuracy:.2f}%"
        )



        # Validation
        model.eval()
        val_loss = 0.0
        #total_miou = 0.0
        total_correct_predictions = 0
        total_elements = 0
        total_confusion_matrix = np.zeros((num_classes, num_classes))

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        # Instantiate Jaccard Index for semantic segmentation
        #iou = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

        with torch.no_grad():
            num_runs = 0
            total_flops = 0
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                
                
                total_flops += FlopCountAnalysis(model, images).total()
                num_runs+=1
                preds = torch.argmax(outputs, dim=1)         
                mask = masks != -1
                # Apply the mask to predictions and targets
                masked_preds = preds[mask]
                masked_targets = masks[mask]

                correct_predictions = torch.sum(masked_preds == masked_targets).item()
                elements = masked_preds.numel()
                #accuracy = (correct_predictions / total_elements) * 100
                total_correct_predictions += correct_predictions
                total_elements += elements

                # Update confusion matrix
                total_confusion_matrix += fast_hist(masked_targets.cpu().numpy(), masked_preds.cpu().numpy(), num_classes)

                # Loss computation
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()

                val_pbar.set_postfix(loss=val_loss / (val_pbar.n + 1))
                
                # Compute Mean IoU
                #batch_miou = iou(masked_preds, masked_targets)
                # Compute IoU
                #batch_miou = mean_iou(preds, masks, num_classes)
                #total_miou += batch_miou.item()
                #total_accuracy += accuracy
                # Calcolo latenza media
            total_time = end_time - start_time
            avg_latency = (total_time / num_runs) * 1000  # Converti in millisecondi
        #avg_miou = total_miou / len(val_loader)
        # Calculate per-class IoU and mean IoU
        per_class_iou_values = per_class_iou(total_confusion_matrix)
        mIoU = np.mean(per_class_iou_values)
        
        #avg_accuracy = total_accuracy / len(val_loader)
        avg_accuracy = (total_correct_predictions / total_elements) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss/len(val_loader):.4f}, mIoU: {mIoU:.4f}, Average accuracy: {avg_accuracy:.4f} %, Total FLOPs: {total_flops / 1e9:.2f} GFLOPs, Average Latency: {avg_latency:.2f} ms")
        print()

        if best_mIoU < mIoU:
            best_miou = mIoU
            torch.save(model.state_dict(), "training/deeplabv2_loveda_best.pth")
            logging.info(f"Best model saved at epoch {epoch+1} with mIoU: {mIoU:.4f}")

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] Validation - "
            f"Loss: {val_loss/len(val_loader):.4f}, "
            f"mIoU: {mIoU:.4f}, "
            f"Accuracy: {avg_accuracy:.2f} %, "
            f"{total_flops / 1e9:.2f} GFLOPs, "
            f"Latency: {avg_latency:.2f} ms"
        )
        logging.info("-" * 50)  # Linea divisoria nel file di log


    logging.info("Training completed.")

    

"""     #------------------------------------#

    # Calcolo dei parametri
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params}")

    # Calcolo della latenza su dati reali
    latency = measure_latency_on_validation(model, val_loader, device)
    print(f"Latency (real input): {latency:.2f} ms")

    # Calcolo dei FLOPs su dati reali
    #flops = calculate_flops_on_validation(model, val_loader)
    print(f"FLOPs (real input): {flops / 1e9:.2f} GFLOPs")


    logging.info(
            f"mIoU: {mean_iou:.2f}\n"
            f"Number of Parameters: {num_params}\n"
            f"Latency: {latency:.2f} ms\n"
            f"FLOPs: {flops / 1e9:.2f} GFLOPs\n"
        ) """
