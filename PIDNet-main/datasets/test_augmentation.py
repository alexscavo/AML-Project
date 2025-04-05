import sys
import os

#TO MAKE IT WORKING JUST REMOVE THE . IN BASE DATASET

# Aggiungi la directory principale del progetto al PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

import argparse
import matplotlib.pyplot as plt
import numpy as np
from loveda import LoveDA, DataAugmentation
from configs import config
from configs import update_config
import datasets

crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

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

args = parse_args()

print(config.DATASET.DATASET)

train_dataset = eval('datasets.'+'loveda')(
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
                    multiply=config.TRAIN.AUGMENTATION.TECHNIQUES.MULTIPLY,
                    random_brightness=config.TRAIN.AUGMENTATION.TECHNIQUES.RANDOM_BRIGHTNESS)


# Carica un'immagine dal dataset
index = 0  # Cambia l'indice per testare altre immagini
image, label, edge, size, name = train_dataset[index] #output della getItem

# Funzione per visualizzare immagine, label ed edge
def visualize_original(image, label, edge, title_prefix=""):
    if image.shape[0] == 3:  # Se l'immagine è in formato (C, H, W)
        image = image.transpose(1, 2, 0)  # Da (C, H, W) a (H, W, C)

    # Ripristina la normalizzazione
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image * std) + mean  # Ripristina la deviazione standard e la media
    image = image * 255.0  # Ripristina i valori a 0-255

    # Converte in uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Visualizza immagine, label ed edge
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"{title_prefix}Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap="tab20")
    plt.title(f"{title_prefix}Label")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge, cmap="gray")
    plt.title(f"{title_prefix}Edge")
    plt.axis("off")

    plt.show()

# Funzione per testare una singola tecnica di augmentation
def test_single_augmentation(augmentation_instance, augmentation_method, image, label, edge, technique_name):
    # Chiama il metodo di augmentation sull'istanza
    augmented_image, augmented_label, augmented_edge = getattr(augmentation_instance, augmentation_method)(image, label, edge)

    # Visualizza immagine originale
    visualize_original(image, label, edge, title_prefix="Original ")

    # Visualizza immagine con augmentation
    visualize_original(augmented_image, augmented_label, augmented_edge, title_prefix=f"{technique_name} ")


# Configura le augmentation
config_dict = {
    "ENABLE": True,
    "PROBABILITY": 1.0,  # Applica sempre le augmentation
    "TECHNIQUES": {}
}
augmentation = DataAugmentation(config_dict)

# Testa il Gaussian Blur
augmented_image, augmented_label, augmented_edge = augmentation.gaussian_blur(image, label, edge)
visualize_original(augmented_image, augmented_label, augmented_edge, title_prefix="Gaussian Blur ")

# Testa il Flip Orizzontale
augmented_image, augmented_label, augmented_edge = augmentation.horizontal_flip(image, label, edge)
visualize_original(augmented_image, augmented_label, augmented_edge, title_prefix="Horizontal Flip ")

# Testa il Random Brightness
augmented_image, augmented_label, augmented_edge = augmentation.random_brightness(image, label, edge)
visualize_original(augmented_image, augmented_label, augmented_edge, title_prefix="Random Brightness ")

# Testa il Multiply
augmented_image, augmented_label, augmented_edge = augmentation.multiply(image, label, edge)
visualize_original(augmented_image, augmented_label, augmented_edge, title_prefix="Multiply ")