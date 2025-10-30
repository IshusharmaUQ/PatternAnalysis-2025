# ===============================
# Data Loader and Preprocessing
# ===============================

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -------------------------------
# Configurations
# -------------------------------
IMAGE_DIR = "/kaggle/input/isic-2020-jpg-224x224-resized/train"
TRAIN_CSV = "/kaggle/input/isic-2020/labels.csv"   
BATCH_SIZE = 32
TRAIN_SPLIT = 584  


# -------------------------------
# Data Transformations
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------------
# Basic ISIC Dataset
# -------------------------------
class ISICDataset(Dataset):
    """
    Dataset for individual image-label pairs.
    Expects DataFrame with columns ['isic_id' or 'image_name', 'target'].
    """
    def __init__(self, df, image_dir=IMAGE_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row.get('isic_id') or row.get('image_name') or row.get('image')
        label = int(row.get('target', 0))
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------------
# Triplet Dataset
# -------------------------------
class TripletMelanomaDataset(Dataset):
    """
    Creates triplets (anchor, positive, negative) consistent with the notebook's triplet sampling idea.
    """
    def __init__(self, df, image_dir=IMAGE_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.pos_idx = self.df[self.df['target'] == 1].index.tolist()
        self.neg_idx = self.df[self.df['target'] == 0].index.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]
        anchor_label = int(anchor_row['target'])

        if anchor_label == 1:
            pos_idx = np.random.choice(self.pos_idx)
            neg_idx = np.random.choice(self.neg_idx)
        else:
            pos_idx = np.random.choice(self.neg_idx)
            neg_idx = np.random.choice(self.pos_idx)

        pos_row = self.df.iloc[pos_idx]
        neg_row = self.df.iloc[neg_idx]

        def load_row(r):
            img_id = r.get('isic_id') or r.get('image_name') or r.get('image')
            path = os.path.join(self.image_dir, f"{img_id}.jpg")
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img

        anchor_img = load_row(anchor_row)
        pos_img = load_row(pos_row)
        neg_img = load_row(neg_row)

        return anchor_img, pos_img, neg_img, anchor_label


# -------------------------------
# DataLoader Builder
# -------------------------------
def build_loaders(df, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Build train and validation DataLoaders trying to mirror the notebook splitting logic.
    Returns train_loader, val_loader (triplet loaders).
    """
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:split].reset_index(drop=True)
    val_df = df_shuffled.iloc[split:].reset_index(drop=True)

    train_ds = TripletMelanomaDataset(train_df, image_dir=image_dir, transform=train_transform)
    val_ds = TripletMelanomaDataset(val_df, image_dir=image_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
