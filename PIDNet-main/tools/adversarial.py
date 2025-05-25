# cd PIDNet-main
# $env:KMP_DUPLICATE_LIB_OK="TRUE"
# python -m tools.adversarial

from pathlib import Path
import sys
from configs import config
import datasets
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
import albumentations as A
project_root = Path(__file__).resolve().parents[1]  # Goes to project_root/
pidnet_path = project_root / "PIDNet-main" / "models"
sys.path.append(str(pidnet_path))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from types import SimpleNamespace
from configs import update_config
import argparse
import models.pidnet
import matplotlib.pyplot as plt
import numpy as np
import random
from utils.utils import FullModel
import os 
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
device = torch.device("cuda")

# Define class labels
CLASS_LABELS = {
    0: 'Ignored',
    1: 'Background',
    2: 'Building',
    3: 'Road',
    4: 'Water',
    5: 'Barren',
    6: 'Forest',
    7: 'Agriculture'
}
def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def validate(testloader, model):
    model.eval()
    nums = 2
    confusion_matrix = np.zeros((8, 8, 2))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)


    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # Discard the first class (index 0)
        IoU_array = IoU_array[1:]

        mean_IoU = IoU_array.mean()
        
        # logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    return mean_IoU, IoU_array

def decode_segmap(segmentation, num_classes=8):
    # Define a simple color map for 8 classes
    label_colors = np.array([
        [0, 0, 0],         # 0 - Ignored
        [128, 64, 128],    # 1 - Background
        [70, 70, 70],      # 2 - Building
        [128, 0, 0],       # 3 - Road
        [0, 0, 255],       # 4 - Water
        [153, 153, 153],   # 5 - Barren
        [0, 128, 0],       # 6 - Forest
        [255, 255, 0],     # 7 - Agriculture
    ])

    rgb = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label in range(num_classes):
        rgb[segmentation == label] = label_colors[label]
    return rgb

def visualize_predictions(model, dataloader, adv_images, adv_labels, num_samples=3):
    model.eval()
    indices = random.sample(range(len(adv_images)), num_samples)

    for i, idx in enumerate(indices):
        # Get original input + GT from dataset
        image, label, _, _, _= dataloader.dataset[idx]
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)
        gt = label

        with torch.no_grad():
            logits = model.model(image)[-2]
            logits = F.interpolate(logits, size=gt.shape, mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            pred_clean = logits.argmax(1).squeeze(0).cpu().numpy()
            logits_adv = model.model(adv_images[idx:idx+1].to(device))[-2]
            logits_adv = F.interpolate(logits_adv, size=gt.shape, mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            pred_adv = logits_adv.argmax(1).squeeze(0).cpu().numpy()
        # Prepare original RGB image for display
        

        img_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()  # [H, W, C], range [0, 1]


        # Unnormalize
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * imagenet_std + imagenet_mean
        img_np = np.clip(img_np, 0, 1)

        # Decode segmentation masks to RGB
        rgb_clean = decode_segmap(pred_clean)
        rgb_adv = decode_segmap(pred_adv)
        rgb_gt = decode_segmap(gt)

        fig, axs = plt.subplots(1, 4, figsize=(20, 8))

        # Plot the 4 images
        titles = ["Original RGB Image", "Original Prediction", "FGSM Prediction", "Ground Truth"]
        for ax, img, title in zip(axs, [img_np, rgb_clean, rgb_adv, rgb_gt], titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        # Legend on top
        handles = [
            plt.Line2D([0], [0], marker='s', color='w', label=name,
                    markerfacecolor=np.array(decode_segmap(np.array([[i]]))[0, 0]) / 255.0,
                    markersize=10)
            for i, name in CLASS_LABELS.items()
        ]
        fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.90))

        # Give space at the top for the legend
        plt.subplots_adjust(top=0.88)
        plt.show()




def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/loveda/pidnet_small_loveda.yaml", #file di configurazione da usare
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args) #aggiorna config con tutti i parametri trovati nel file di configurazione

    return args

def fgsm_attack(model, dataloader, epsilon=0.03, device='cuda', ignore_index=0, max_samples=5):
    model.eval()
    adv_images_all = []
    labels_all = []

    class_weights = torch.tensor(
        [0, 0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396],
        device=device
    )

    collected = 0

    for images, labels, _, _, _ in tqdm(dataloader, desc="Generating FGSM adversarial examples"):
        images = images.to(device).float()
        labels = labels.to(device).long()
        images.requires_grad = True

        logits = model.model(images)[-2]
        logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
        model.zero_grad()
        loss.backward()
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406],
                             device=device).view(1, 3, 1, 1)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225],
                             device=device).view(1, 3, 1, 1)

        img_unnorm = images * imagenet_std + imagenet_mean
        adv_unnorm = img_unnorm + epsilon * images.grad.sign()
        adv_unnorm = adv_unnorm.clamp(0.0, 1.0)
        # perturbed_images = images + epsilon * images.grad.sign()
        # perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = (adv_unnorm - imagenet_mean) / imagenet_std


        for img, lbl in zip(perturbed_images.detach().cpu(), labels.detach().cpu()):
            adv_images_all.append(img)
            labels_all.append(lbl)
            collected += 1
            if collected >= max_samples:
                return torch.stack(adv_images_all), torch.stack(labels_all)

    return torch.stack(adv_images_all), torch.stack(labels_all)


def evaluate_on_adv(model, adv_images, labels, num_classes):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(adv_images), 4):  # batch size 4
            x = adv_images[i:i+4].to(device)
            logits = model.model(x)[-2]
            logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

            pred = logits.argmax(1).cpu()
            preds.append(pred)

    preds = torch.cat(preds)
    labels = labels[:len(preds)]

    ious = []
    for cls in range(num_classes):
        if cls != 0:
            pred_inds = (preds == cls)
            label_inds = (labels == cls)
            intersection = (pred_inds & label_inds).sum().item()
            union = (pred_inds | label_inds).sum().item()
            if union > 0:
                ious.append(intersection / union)
    mean_iou = sum(ious) / len(ious)
    print(f"Mean IoU on adversarial samples: {mean_iou:.4f}")



if __name__ == '__main__':
    args = parse_args()

    gpus = list(config.GPUS)
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                thres=config.LOSS.OHEMTHRES,
                                min_kept=config.LOSS.OHEMKEEP,
                                weight=test_dataset.class_weights)
    bd_criterion = BondaryLoss()

    base_model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)
    model = FullModel(base_model.to(device), sem_criterion, bd_criterion).to(device)

    model_path = "output/loveda/pidnet_small_loveda/dacs_cj/best.pt"
    checkpoint = torch.load(str(model_path), map_location="cpu")
    
    state_dict = checkpoint
    if list(state_dict.keys())[0].startswith("module."):
        from collections import OrderedDict
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    
    #state_dict = torch.load(str(model_path), map_location="cpu")
    
    
    model.load_state_dict(state_dict, strict=True)

    miou, iou_array = validate(testloader, model)
    print(miou)
    print(iou_array)

    model.eval()

    # Run FGSM attack
    adv_images, adv_labels = fgsm_attack(
        model,
        testloader,
        epsilon=0.03,
        device=device,
        ignore_index=config.TRAIN.IGNORE_LABEL,
        max_samples=200  # only generate 5 samples
    )
    # Evaluate on adversarial images (optional)
    evaluate_on_adv(model, adv_images, adv_labels, num_classes=config.DATASET.NUM_CLASSES)

    # Visualize predictions for a few random samples
    visualize_predictions(model, testloader, adv_images, adv_labels, num_samples=10)