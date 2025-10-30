import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ISICDataset(Dataset):
    """
    Custom dataset for loading ISIC skin lesion images and their labels.
    """
    def __init__(self, image_file_paths, class_labels, transform=None):
        self.image_file_paths = image_file_paths
        self.class_labels = class_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, index):
        image_path = self.image_file_paths[index]
        image = Image.open(image_path)
        label = self.class_labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label