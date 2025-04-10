# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import cv2
import os
import numpy as np
import torch
import random
import logging
from PIL import Image
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from base_dataset import BaseDataset

def compare_images(image, blurred_image):
    # Compute the absolute difference between the images
    # Ensure both images are in the same format (H, W, C)
    blurred_image = blurred_image.transpose(1, 2, 0)  # Change (C, H, W) to (H, W, C)

    # Compute the absolute difference between the images
    diff = np.abs(image - blurred_image)
    
    # Plot the images and their difference side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)  # Original image
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(blurred_image)  # Blurred image
    axs[1].set_title("Blurred Image")
    axs[1].axis("off")

    axs[2].imshow(diff)  # Difference image
    axs[2].set_title("Difference Image")
    axs[2].axis("off")

    plt.show()

    return diff


def show_images(x_original, x_augmented, unnormalize = False):
    
    if unnormalize:
        # ImageNet mean and std
        imagenet_mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        imagenet_std = np.array([0.229, 0.224, 0.225])[:, None, None]

        # Denormalize using NumPy broadcasting
        x_original = x_original * imagenet_std + imagenet_mean
        x_augmented = x_augmented * imagenet_std + imagenet_mean

        # Clip to [0, 1] in case of overflows
        x_original = np.clip(x_original, 0, 1)
        x_augmented = np.clip(x_augmented, 0, 1)

        # Transpose to HWC for matplotlib
        x_original = np.transpose(x_original, (1, 2, 0))
        x_augmented = np.transpose(x_augmented, (1, 2, 0))

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(x_original)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(x_augmented)
    axs[1].set_title("Augmented Image")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


#classe per fare la data augmentation
class DataAugmentation:
    def __init__(self, config, dataset_instance):
        self.enable = config["ENABLE"]
        self.probability = config["PROBABILITY"]
        self.techniques = config["TECHNIQUES"]
        self.dataset = dataset_instance  # Riferimento all'istanza del dataset

    def apply(self, image, label, edge):
        
        if not self.enable or random.random() > self.probability: #50% di probabilità di applicare la data augmentation
            return image,label,edge #non faccio augmentation

        if self.techniques.get("HORIZONTAL_FLIP", False):
            image,label,edge = self.horizontal_flip(image, label, edge)

        if self.techniques.get("GAUSSIAN_BLUR", False):
            image, label, edge = self.gaussian_blur(image, label, edge)

        if self.techniques.get("MULTIPLY", False):
            image, label, edge = self.multiply(image, label, edge)

        if self.techniques.get("RANDOM_BRIGHTNESS", False):
            image, label, edge = self.random_brightness(image, label, edge)

        if self.techniques.get("RANDOM_CROP", False):
            image, label, edge = self.random_crop(image, label, edge) 


        return image, label, edge



    def random_crop(self, image, label, edge):
        return self.dataset.rand_crop(image, label, edge)  # Usa l'istanza del dataset



    def horizontal_flip(self, image, label, edge):
        # Inverti orizzontalmente immagine, label ed edge
        flipped_image = image[:, :, ::-1]
        flipped_label = label[:, ::-1]
        flipped_edge = edge[:, ::-1]
        return flipped_image, flipped_label, flipped_edge
    
    

    def gaussian_blur(self, image, label, edge, kernel_size=5, show = False):
        # Applica il Gaussian Blur solo all'immagine
        transposed_image = image.transpose(1, 2, 0)  # From (C, H, W) to (H, W, C)
    
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(transposed_image, (kernel_size, kernel_size), 0)
        
        # If you want to return it to the PyTorch format (C, H, W)
        blurred_image = blurred_image.transpose(2, 0, 1)  # From (H, W, C) to (C, H, W)

        if show:
            show_images(image, blurred_image)

        return blurred_image, label, edge
    


    def multiply(self, image, label, edge, factor_range=(0.8, 1.2), show = False):
        # Convert image to float32 to avoid overflow issues
        factor = random.uniform(*factor_range)
        image = image.astype(np.float32)  # Ensure safe multiplication

        # Check if image is normalized (0-1), rescale before multiplication
        if image.max() <= 1.0:
            image *= 255.0  # Scale to 0-255 range before multiplication

        multiplied_image = image * factor

        if show:
            show_images(image, multiplied_image)
        
        return multiplied_image, label, edge

    def random_brightness(self, image, label, edge, brightness_range=(-0.5, 0.5), show = False):
        # Modify image brightness
        brightness = np.float32(np.random.uniform(*brightness_range)) 
        brightened_image = image + brightness  # Keep within [0,1]
        
        if show:
            show_images(image, brightened_image)
        return brightened_image, label, edge
    


class LoveDA(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=7, 
                 multi_scale=False, 
                 flip=False,
                 ignore_label=0,
                 base_size=1024,
                 crop_size=(512, 512),
                 scale_factor=16, #multi scale usato come data augmentation alredy provided
                 enable_augmentation=False,
                 augmentation_probability=0.5, 
                 horizontal_flip=False,
                 gaussian_blur=False,
                 multiply=False,
                 random_brightness=False,
                 random_crop=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 pseudo_label=False,
                 transform=None):

        # estende il base_dataset
        super(LoveDA, self).__init__(ignore_label, base_size,
                                     crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
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

        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.files = self.read_files()
        self.color_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2],
                            [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]
        self.class_weights = torch.tensor([0.000000, 0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396])
        self.pseudo_label = pseudo_label
        self.transform=transform

    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

        return files

    # da immagine a label
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    # da label a immagine
    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        # image = Image.open(item["img"]).convert('RGB')
        # image = np.array(image) #H,W,3
        size = image.shape

        """ if 'val' in self.list_path:
            label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            #generazione edge
            edge_size=4
            edge = cv2.Canny(label, 0.1, 0.2)
            kernel = np.ones((edge_size, edge_size), np.uint8)
            edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0

            return image.copy(), label.copy(), edge.copy(), np.array(size), name
        """
        # color_map = Image.open(item["label"]).convert('RGB')
        # color_map = np.array(color_map)
        # label = self.color2label(color_map) #label diventa (H,W)
         
        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)



        #edge (H,W)
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_pad=False,
                                             edge_size=self.bd_dilate_size, city=False) #image diventa (C,H,W)
        
        # Augmentation
        config_dict = {
            "ENABLE": self.enable_augmentation, 
            "PROBABILITY": self.augmentation_probability, 
            "TECHNIQUES":{
                "HORIZONTAL_FLIP": self.horizontal_flip, 
                "GAUSSIAN_BLUR": self.gaussian_blur, 
                "MULTIPLY": self.multiply, 
                "RANDOM_BRIGHTNESS": self.random_brightness,
                "RANDOM_CROP": self.random_crop
            }
            
        }
        # augmentation = DataAugmentation(config_dict,self)
        # image, label, edge = augmentation.apply(image, label, edge)
        

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))



