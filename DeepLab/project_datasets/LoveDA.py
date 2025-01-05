import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class LoveDADataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        """
        Args:
            image_dir (str): Path to the directory with input images.
            mask_dir (str): Path to the directory with masks.
            transform (callable, optional): Transformations for the input images.
            target_transform (callable, optional): Transformations for the masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Use PIL to load images
        image = Image.open(image_path).convert("RGB")  # Convert to 3-channel RGB
        mask = Image.open(mask_path)  # Grayscale mask (single channel)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
