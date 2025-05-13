from utils import fda
from configs import config
from torch.utils import data
import cv2
import torch
import os
from .base_dataset import BaseDataset
import numpy as np



class FDADataset(BaseDataset):
    def __init__(self,
                root,
                source_list_path,
                target_list_path,
                num_classes=8, 
                multi_scale=False, 
                flip=False,
                ignore_label=0,
                base_size=1024,
                crop_size=(1024, 1024),
                scale_factor=16, #multi scale usato come data augmentation alredy provided
                enable_augmentation=False,
                augmentation_probability=0.5, 
                horizontal_flip=False,
                gaussian_blur=False,
                multiply=False,
                random_brightness=False,
                random_crop=False,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                bd_dilate_size=4,
                pseudo_label=False,
                transform=None):


        # estende il base_dataset
        super(FDADataset, self).__init__(ignore_label, base_size, crop_size, scale_factor, mean, std)

        self.root = root
        self.source_list_path = source_list_path
        self.target_list_path = target_list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.enable_augmentation = enable_augmentation
        self.augmentation_probability = augmentation_probability
        self.horizontal_flip = horizontal_flip
        self.gaussian_blur = gaussian_blur
        self.multiply = multiply
        self.random_brightness = random_brightness
        self.random_crop = random_crop
        self.bd_dilate_size = bd_dilate_size

        self.source_files = [line.strip().split() for line in open(root + source_list_path)]
        self.target_files = [line.strip().split() for line in open(root + target_list_path)]
        self.source_files = self.read_files_source()
        self.target_files = self.read_files_target()

        print("Source file list length:", len(self.source_files))
        print("Target file list length:", len(self.target_files))
        #anche se lunghezze diverse io faccio mapping 1 a 1 e tengo conto della lunghezza del source (amche la minore)
        self.files= self.source_files #serve a BaseDataset per il metodo __len__


        self.color_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2],
                            [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]
        self.class_weights = torch.tensor([0.000000, 0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396])
        self.pseudo_label = pseudo_label
        self.transform=transform

    def read_files_source(self):
        files = []

        for item in self.source_files:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

        return files
    
    def read_files_target(self):
        files = []

        for item in self.target_files:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

        return files


    def __getitem__(self, index):
        source_item = self.source_files[index]
        target_item = self.target_files[index]

        source_image = cv2.imread(source_item["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(source_item["label"], cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(target_item["img"], cv2.IMREAD_COLOR)
        # label di target inutile

        # Converti in float32 e RGB
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if config.TRAIN.FDA.ENABLE:
             # Prepara immagini per FDA: CHW
            src_img = source_image.transpose(2, 0, 1)
            trg_img = target_image.transpose(2, 0, 1)

            source_image = fda.FDA_source_to_target_np(src_img, trg_img)
            source_image = source_image.transpose(1, 2, 0) # Torna a HWC (come si aspetta gen_sample)

        source_image, label, edge = self.gen_sample(
            source_image, label,
            edge_pad=False,
            edge_size=self.bd_dilate_size,
            city=False,
            transform=self.transform,
            show=False
        )

        return source_image.copy(), label.copy(), edge.copy()