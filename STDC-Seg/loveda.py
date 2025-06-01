import glob
import os
import os.path as osp
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from transform import Compose, ColorJitter, HorizontalFlip, RandomScale, RandomCrop
import albumentations as A
class LoveDA(Dataset):
    """
    PyTorch Dataset for LoveDA semantic segmentation.

    Arguments:
        rootpth (str): path to folder containing Train/, Val/, Test/
        cropsize (tuple): size to crop (width, height) during training
        mode (str): one of 'train', 'val', 'test'
        randomscale (tuple): scales for RandomScale
    """
    scenes = ('urban', 'rural')

    def __init__(self,
                rootpth,
                cropsize=(640, 640),
                mode='train',
                randomscale=(0.125, 0.25, 0.375, 0.5, 0.675,
                            0.75, 0.875, 1.0, 1.25, 1.5), apply_augmentation=False, resolution=(224, 224), *args, **kwargs):
        super(LoveDA, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'target'), "mode must be 'train', 'val', or 'target'"
        self.mode = mode
        self.ignore_lb = 255  # no-data regions in LoveDA
        self.rootpth = rootpth
        self.apply_augmentation = apply_augmentation
        # Determine scenes depending on mode
        if mode == 'train':
            self.scenes = ['urban']
        elif mode == 'val':
            self.scenes = ['rural']
        else:
            self.scenes = ['rural']
        # Base path: nested folder Train/Train or Val/Val
        if mode == 'train' or mode == 'target':
            split_dir = 'Train'
        else:
            split_dir = 'Val'
        base = osp.join(self.rootpth, split_dir, split_dir)

        # Collect image and mask paths
        self.images = {}
        self.labels = {}
        for scene in self.scenes:
            scene_dir = osp.join(base, scene.capitalize()) 
            img_dir = osp.join(scene_dir, 'images_png')
            for ip in glob.glob(osp.join(img_dir, '*.png')):
                name = osp.splitext(osp.basename(ip))[0]
                self.images[name] = ip

            if mode != 'test':
                lb_dir = osp.join(scene_dir, 'masks_png')
                for lp in glob.glob(osp.join(lb_dir, '*.png')):
                    name = osp.splitext(osp.basename(lp))[0]
                    self.labels[name] = lp

        # Ensure consistency
        self.imnames = sorted(self.images.keys())
        assert mode == 'test' or set(self.imnames) == set(self.labels.keys())

        # Preprocessing transforms
        self.to_tensor = T.Compose([
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.mask_transforms = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST),  # Resize mask
            T.PILToTensor(),  # Remove channel dimension
        ])
        self.trans_train = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize),
        ])
        self.trans_target =  A.Compose([
            A.ColorJitter(p=0.5) 
        ])
         


    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, idx):
        key = self.imnames[idx]
        img = Image.open(self.images[key]).convert('RGB')

        if self.mode != 'test':
            lb = Image.open(self.labels[key])
            data = {'im': img, 'lb': lb}
            if self.mode == 'train' and self.apply_augmentation:
                # convert to numpy arrays
                img_np = np.array(img)
                lb_np  = np.array(lb)
                # call with named args
                aug = self.trans_target(image=img_np, mask=lb_np)
                # pull out augmented arrays
                img = Image.fromarray(aug['image'])
                lb  = Image.fromarray(aug['mask'])
                data = {'im': img, 'lb': lb}
                # self.trans_train(data)
            elif self.mode == 'target' and self.apply_augmentation:
                # convert to numpy arrays
                img_np = np.array(img)
                lb_np  = np.array(lb)
                # call with named args
                aug = self.trans_target(image=img_np, mask=lb_np)
                # pull out augmented arrays
                img = Image.fromarray(aug['image'])
                lb  = Image.fromarray(aug['mask'])
                data = {'im': img, 'lb': lb}
                # self.trans_target(data)
            img, lb = data['im'], data['lb']
        else:
            lb = None
            img = img  # no mask for test

        img = self.to_tensor(img)
        lb = self.mask_transforms(lb)

        if lb is not None:
            label = np.array(lb, dtype=np.int16)  # allow negatives

            label[label == 0] = -1     # no-data → temporary -1
            label = label - 1          # now: [1–7] → [0–6], and -1 becomes -2

            label[label < 0] = 255     # set ignore index (no-data or underflow)
            label[label > 6] = 255     # safety check

            return img, torch.from_numpy(label).long()
        else:
            return img
