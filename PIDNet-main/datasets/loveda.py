# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import cv2
import os
import numpy as np
import random
from PIL import Image

from .base_dataset import BaseDataset

#classe per fare la data augmentation
class DataAugmentation:
    def __init__(self, config):
        self.enable = config["ENABLE"]
        self.probability = config["PROBABILITY"]
        self.techniques = config["TECHNIQUES"]

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

        return image, label, edge



    def horizontal_flip(self, image, label, edge):
        # Inverti orizzontalmente immagine, label ed edge
        flipped_image = image[:, ::-1, :]
        flipped_label = label[:, ::-1]
        flipped_edge = edge[:, ::-1]
        return flipped_image, flipped_label, flipped_edge

    def gaussian_blur(self, image, label, edge, kernel_size=5):
        # Applica il Gaussian Blur solo all'immagine
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image, label, edge

    def multiply(self, image, label, edge, factor_range=(0.8, 1.2)):
        # Modifica il contrasto dell'immagine
        factor = random.uniform(*factor_range)
        multiplied_image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return multiplied_image, label, edge

    def random_brightness(self, image, label, edge, brightness_range=(-30, 30)):
        # Modifica la luminosità dell'immagine
        brightness = random.randint(*brightness_range)
        brightened_image = np.clip(image + brightness, 0, 255).astype(np.uint8)
        return brightened_image, label, edge
    


class LoveDA(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=6,
                 multi_scale=False, 
                 flip=False,
                 ignore_label=0,
                 base_size=1024,
                 crop_size=(1024, 1024),
                 scale_factor=16, #multi scale usato come data augmentation alredy provided
                 horizontal_flip=False,
                 gaussian_blur=False,
                 multiply=False,
                 random_brightness=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

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
        self.horizontal_flip = horizontal_flip
        self.gaussian_blur = gaussian_blur
        self.multiply = multiply
        self.random_brightness = random_brightness
        self.bd_dilate_size = bd_dilate_size

        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.files = self.read_files()
        self.color_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2],
                            [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
        self.class_weights = None

        

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

    # da label a immagine
    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(item["img"]).convert('RGB')
        image = np.array(image)
        size = image.shape

        color_map = Image.open(item["label"]).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_pad=False,
                                             edge_size=self.bd_dilate_size, city=False)
        
        # Augmentation
        augmentation = DataAugmentation(self.config.TRAIN.AUGMENTATION)
        image, label, edge = augmentation.apply(image, label, edge)
        

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



