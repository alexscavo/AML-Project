import sys
import os
import numpy as np
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


if __name__ == '__main__':
    # Define transformations for images and masks
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 512x512
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize mask
        transforms.PILToTensor(),  # Convert mask to tensor
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
    model = dlb.get_deeplab_v2(num_classes=num_classes)

    # Define optimizer and loss function
    optimizer = optim.SGD(model.optim_parameters(0.0001), momentum=0.9, weight_decay=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    iou = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
    num_epochs = 20  # Number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Training Progress Bar
        total_miou = 0.0
        total_accuracy = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            mask = masks != -1

            # Apply the mask to predictions and targets
            masked_preds = preds[mask]
            masked_targets = masks[mask]

            correct_predictions = torch.sum(masked_preds == masked_targets).item()
            total_elements = masked_preds.numel()

            accuracy = (correct_predictions / total_elements) * 100
            # Compute Mean IoU
            batch_miou = iou(masked_preds, masked_targets)
            # Compute IoU
            #batch_miou = mean_iou(preds, masks, num_classes)
            total_miou += batch_miou.item()
            total_accuracy += accuracy


            loss = criterion(outputs, masks.long())
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=running_loss / (train_pbar.n + 1))
        avg_miou = total_miou / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
         
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {running_loss/len(train_loader):.4f}, mIoU: {avg_miou:.4f}, Average accuracy: {avg_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        total_miou = 0.0
        total_accuracy = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        # Instantiate Jaccard Index for semantic segmentation
        iou = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks.long())
            
                val_loss += loss.item()
               
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                                
                mask = masks != -1

                # Apply the mask to predictions and targets
                masked_preds = preds[mask]
                masked_targets = masks[mask]

                correct_predictions = torch.sum(masked_preds == masked_targets).item()
                total_elements = masked_preds.numel()

                accuracy = (correct_predictions / total_elements) * 100
                
                # Compute Mean IoU
                batch_miou = iou(masked_preds, masked_targets)
                # Compute IoU
                #batch_miou = mean_iou(preds, masks, num_classes)
                total_miou += batch_miou.item()
                total_accuracy += accuracy
                val_pbar.set_postfix(loss=val_loss / (val_pbar.n + 1))
        avg_miou = total_miou / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss/len(val_loader):.4f}, mIoU: {avg_miou:.4f}, Average accuracy: {avg_accuracy:.4f}")