import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision import transforms
from project_datasets.LoveDA import LoveDADataset
import torch
import models.deeplabv2 as dlb 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
# Configurazione del logger
log_dir = "results/logs"
os.makedirs(log_dir, exist_ok=True)  # Crea la cartella dei log se non esiste
log_file = os.path.join(log_dir, "eval_DeepLab.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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


    val_dataset = LoveDADataset(
        image_dir="project_datasets/LoveDA/Val/Urban/images_png",
        mask_dir="project_datasets/LoveDA/Val/Urban/masks_png",
        transform=img_transforms,
        target_transform=mask_transforms
    )

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    
    
    num_classes = 7

    model = dlb.get_deeplab_v2(num_classes=num_classes)
    device ='cuda'
    parameters_path = 'training/results/deeplabv2_loveda_best.pth'
    state_dict = torch.load(parameters_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)


    model.eval()
    val_loss = 0.0
    #total_miou = 0.0
    total_correct_predictions = 0
    total_elements = 0
    total_confusion_matrix = np.zeros((num_classes, num_classes))
    profiler_logged = False  # only once per epoch

    val_pbar = tqdm(val_loader, desc=f"[Validation]")
    with torch.no_grad():
        num_runs = 0
        total_flops = 0
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            num_runs += 1
            preds = torch.argmax(outputs, dim=1)         
            mask = masks != -1
            # Apply the mask to predictions and targets
            masked_preds = preds[mask]
            masked_targets = masks[mask]

            correct_predictions = torch.sum(masked_preds == masked_targets).item()
            elements = masked_preds.numel()
            total_correct_predictions += correct_predictions
            total_elements += elements

            # Update confusion matrix
            total_confusion_matrix += fast_hist(masked_targets.cpu().numpy(), masked_preds.cpu().numpy(), num_classes)

            
    per_class_iou_values = per_class_iou(total_confusion_matrix)
    mIoU = np.mean(per_class_iou_values)
    
    avg_accuracy = (total_correct_predictions / total_elements) * 100


    logging.info(
        f"mIoU: {mIoU:.4f}, "
        f"Accuracy: {avg_accuracy:.2f} %, "
    )
    logging.info("-" * 50) 
    class_names = [
    'background', 'building', 'road',
    'water', 'barren', 'forest', 'agriculture'
    ]

    logging.info("Per-class IoU:")
    for cls_name, iou in zip(class_names, per_class_iou_values):
        logging.info(f"  {cls_name:12s}: {iou:.4f}")
