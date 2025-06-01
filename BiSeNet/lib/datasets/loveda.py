import numpy as np
import cv2
import lib.data.transform_cv2 as T
import lib.data.base_dataset as base_dataset

class LoveDA(base_dataset.BaseDataset):
    """
    LoveDA semantic‐segmentation dataset in CityScapes‐style API.

    Arguments:
        dataroot (str): root folder for images and masks
        annpath  (str): (unused) kept for API compatibility
        trans_func (callable): augmentation pipeline, e.g. albumentations
        mode     (str): 'train' / 'val' / 'test'
    """
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super().__init__(dataroot, annpath, trans_func, mode)

        # 7 valid classes
        self.n_cats    = 7
        # ignore index for no-data
        self.lb_ignore = 255

        # build mapping [0–255] → trainId (0–6 valid, 255 ignore)
        self.lb_map = np.arange(256, dtype=np.uint8)
        mapping = {
            0: 255,  # no-data → ignore
            1:   0,  # background
            2:   1,  # building
            3:   2,  # road
            4:   3,  # water
            5:   4,  # barren
            6:   5,  # forest
            7:   6,  # agriculture
        }
        for raw_id, train_id in mapping.items():
            self.lb_map[raw_id] = train_id

        # normalize with ImageNet stats (same as your other pipelines)
        self.to_tensor = T.ToTensor(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        )

    """def __getitem__(self, idx):
        # BaseDataset.__getitem__ should load (img, label) as numpy arrays
        img, label = super().__getitem__(idx)

        # remap labels
        label = self.lb_map[label]
        # convert to torch Tensor and normalize image
        img   = self.to_tensor(img)

        return img, label"""
