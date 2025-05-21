import argparse
import _init_paths

from configs import config
from configs import update_config
import torch
import datasets
import random
import models
import torch.nn as nn
from utils.utils import FullModel
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
import numpy as np
import cv2
import albumentations as A


def show_mixed_visualization(x_s, y_s, x_t, y_t, x_mixed, y_mixed, bd_mixed):
    # Move tensors to CPU and convert to NumPy
    x_s = x_s[0].cpu().clone()
    x_t = x_t[0].cpu().clone()
    x_mixed = x_mixed[0].cpu().clone()

    y_s = y_s[0].cpu().numpy()
    y_t = y_t[0].cpu().numpy()
    y_mixed = y_mixed[0].cpu().numpy()
    edge_map = bd_mixed[0].cpu().numpy()
    edge_map_vis = (edge_map != config.TRAIN.IGNORE_LABEL).astype(np.uint8) * edge_map
    edge_map_vis = (edge_map_vis > 0).astype(np.uint8) * 255  # Make edges white on black
    # If normalized, unnormalize (adjust if you used custom values)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    for t, m, s in zip(x_s, imagenet_mean, imagenet_std):
        t.mul_(s).add_(m)
    for t, m, s in zip(x_t, imagenet_mean, imagenet_std):
        t.mul_(s).add_(m)
    for t, m, s in zip(x_mixed, imagenet_mean, imagenet_std):
        t.mul_(s).add_(m)

    x_s = np.transpose(x_s.numpy(), (1, 2, 0))
    x_t = np.transpose(x_t.numpy(), (1, 2, 0))
    x_mixed = np.transpose(x_mixed.numpy(), (1, 2, 0))

    fig, axs = plt.subplots(2, 4, figsize=(16, 6))

    axs[0, 0].imshow(x_s)
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(x_t)
    axs[0, 1].set_title("Target Image")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(x_mixed)
    axs[0, 2].set_title("Mixed Image")
    axs[0, 2].axis("off")


    axs[0, 3].imshow(edge_map_vis, cmap='gray', vmin=0, vmax=255)
    axs[0, 3].set_title("Edge Map")
    axs[0, 3].axis("off")

    axs[1, 0].imshow(y_s, cmap='tab20')
    axs[1, 0].set_title("Source Label")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(y_t, cmap='tab20')
    axs[1, 1].set_title("Pseudo Label")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(y_mixed, cmap='tab20')
    axs[1, 2].set_title("Mixed Label")
    axs[1, 2].axis("off")

    axs[1, 3].axis("off")  # Can be used for something else if needed

    plt.tight_layout()
    plt.show() 

def classmix(x1, y1, x2, y2, verbose=False, deterministic=False):
    classes = y1.unique().tolist()
    selected = random.sample(classes, len(classes) // 2)
    if deterministic:
        selected = classes[:len(classes) // 2]
    mask = torch.zeros_like(y1, dtype=torch.bool)
    if verbose:
        print(f"Classes from the source domain: {selected}") 
    for c in selected:
        mask |= (y1 == c) # c
    mask = mask.unsqueeze(1)
    x_mix = torch.where(mask, x1, x2)
    y_mix = torch.where(mask.squeeze(1), y1, y2)
    return x_mix, y_mix, mask.squeeze(1)

def generate_safe_edge_map(y_mixed, source_mask, edge_size=4, edge_pad=False, ignore_label=255):
    B, H, W = y_mixed.shape
    edge_maps = []
    kernel = np.ones((edge_size, edge_size), np.uint8)
    y_k_size = 6
    x_k_size = 6

    for i in range(B):
        label_np = y_mixed[i].cpu().numpy().astype(np.uint8)
        mask_np = source_mask[i].cpu().numpy().astype(np.uint8)
        label_masked = label_np * mask_np
        edge = cv2.Canny(label_masked, 0.1, 0.2)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.uint8)
        edge[mask_np == 0] = ignore_label
        edge_maps.append(torch.tensor(edge, dtype=torch.long).unsqueeze(0))

    return torch.cat(edge_maps, dim=0).to(y_mixed.device)

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg', default="configs/loveda/pidnet_small_loveda.yaml", type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    args = parse_args()
    gpus = list(config.GPUS)
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    imgnet = 'imagenet' in config.MODEL.PRETRAINED


    train_trasform = None

    if config.TRAIN.AUGMENTATION.ENABLE:
        list_augmentations = []
        if config.TRAIN.AUGMENTATION.TECHNIQUES.RANDOM_CROP:
            list_augmentations.append(A.RandomResizedCrop(1024, 1024, p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP:
            list_augmentations.append(A.HorizontalFlip(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.COLOR_JITTER:
            list_augmentations.append(A.ColorJitter(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR:
            list_augmentations.append(A.GaussianBlur(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_NOISE:
            list_augmentations.append(A.GaussNoise(std_range=(0.2, 0.3), p=0.5))
        if len(list_augmentations) != 0:
            train_trasform = A.Compose(list_augmentations)


    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT, 
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES, 
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP, 
        enable_augmentation=True, 
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE, 
        crop_size=crop_size, 
        scale_factor=config.TRAIN.SCALE_FACTOR,
        horizontal_flip=config.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP,
        gaussian_blur=config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR, 
        transform=train_trasform)

    target_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT, list_path=config.DATASET.TARGET_SET,
        num_classes=config.DATASET.NUM_CLASSES, multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP, enable_augmentation=True, ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE, crop_size=crop_size, scale_factor=config.TRAIN.SCALE_FACTOR,
        horizontal_flip=config.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP,
        gaussian_blur=config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR, 
        transform=train_trasform)

    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS, 
        pin_memory=False, 
        drop_last=True)

    targetloader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS, pin_memory=False, drop_last=True)

    source_iter = iter(trainloader)
    target_iter = iter(targetloader)
    n_to_show = 10
    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_iter, target_iter)):
        if batch_idx > n_to_show-1:
           break

        x_s, y_s, *extra = source_batch
        x_t, real_gt, *extra = target_batch
        x_s, y_s = x_s.cuda(), y_s.cuda()
        x_t = x_t.cuda()
        print("Label map:\nbackground – 1\nbuilding – 2\nroad – 3\nwater – 4\nbarren – 5\nforest – 6\nagriculture – 7\nno-data regions-0 (should be ignored)")

        print(f"Unique labels found in urban image: {y_s.unique().tolist()}")
        print(f"Unique labels found in rural image: {real_gt.unique().tolist()}")
    

        with torch.no_grad():
            logits_t = model.module.model(x_t)[-2]
            logits_t = torch.nn.functional.interpolate(
                logits_t, size=x_t.shape[2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            pseudo_t = torch.argmax(logits_t, dim=1)
            conf_t = torch.softmax(logits_t, dim=1).max(dim=1)[0] # It is a [B, H, W] tensor representing the maximum class probability per pixel
            
            
            use_confidence = False
            if use_confidence:
                confidence_mask = conf_t > config.TRAIN.DACS.THRESHOLD # It tells the position of the pixel whose pseudo-labels have a confidence higher than  the threshold
                pseudo_t = torch.where(
                    confidence_mask,
                    pseudo_t,  # keep if confident
                    torch.full_like(pseudo_t, config.TRAIN.IGNORE_LABEL) # ignore if not confident
                )

        x_mixed, y_mixed, source_mask = classmix(x_s, y_s, x_t, pseudo_t, verbose=True)  
        bd_mixed = generate_safe_edge_map(y_mixed, source_mask, edge_size=3,
                                          edge_pad=True, ignore_label=config.TRAIN.IGNORE_LABEL)        # secondo me threshold tra 0.5 e 0.7 massimo

        show_mixed_visualization(x_s, y_s, x_t, pseudo_t, x_mixed, y_mixed, bd_mixed)
