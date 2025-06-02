# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import random
import albumentations as A
from torch.nn import functional as F
from torch.utils import data
import matplotlib.pyplot as plt
y_k_size = 6
x_k_size = 6

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
    axs[0].axis("off")

    axs[1].imshow(x_augmented)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()





class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=1024,
                 crop_size=(1024, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image, city=False):
        if city:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)

        # if no padding is needed, return the original image
        if pad_h == 0 and pad_w == 0:
            return image

        # should be (H, W, C)
        if len(image.shape) == 3 and image.shape[0] <= 3:  
            image = np.transpose(image, (1, 2, 0))  

        # add padding
        pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue)

        
        if len(image.shape) == 3 and image.shape[2] <= 3:  
            pad_image = np.transpose(pad_image, (2, 0, 1))  

        return pad_image

    def rand_crop(self, image, label, edge):
       
        if len(image.shape) == 3 and image.shape[0] <= 3: 
            image = np.transpose(image, (1, 2, 0)) 

        h, w = image.shape[:2]

      
        if h < self.crop_size[0] or w < self.crop_size[1]:
            image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
            label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
            edge = self.pad_image(edge, h, w, self.crop_size, (0.0,))

       
        new_h, new_w = label.shape
        if new_h < self.crop_size[0] or new_w < self.crop_size[1]:
            raise ValueError(f"Dimensioni insufficienti per il ritaglio: label={label.shape}, crop_size={self.crop_size}")

  
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])

      
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]

      
        '''
        # Estrai la regione da sfocare
        cropped_region = image[y:y+crop_size[0], x:x+crop_size[1]]

        # Applica il Gaussian Blur alla regione
        blurred_region = cv2.GaussianBlur(cropped_region, (15, 15), 0)

        # Sostituisci la regione originale con quella sfocata
        augmented_image = image.copy()
        augmented_image[y:y+crop_size[0], x:x+crop_size[1]] = blurred_region
        '''

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge


    def gen_sample(self, image, label, edge_pad=True, edge_size=4, city=False, transform=None, show=False):
        

        if transform:
            # Pass both image and mask
            augmented = transform(image=image, mask=label)
            
            if show:
                show_images(image, augmented["image"])
            
            # Extract results
            image = augmented['image']
            label = augmented['mask']



        #It's important keeping the edge generation after the data augmentation
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0    


        # input trasformation
        image = self.input_transform(image, city=city) #Se city=True, converte l'immagine da RGB in BGR per opencv
        label = self.label_transform(label) #converte la label in un array di interi
        image = image.transpose((2, 0, 1)) #H,W,C -> C,H,W

        return image, label, edge


    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        
        
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        
        
        return pred.exp()