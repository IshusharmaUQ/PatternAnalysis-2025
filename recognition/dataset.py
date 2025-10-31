import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class TripletMelanomaDataset(Dataset):
    """Creates triplet samples: anchor, positive (same class), negative (different class)"""
    
    def __init__(self, benign_paths, malignant_paths, transform=None, num_triplets=10000, seed=None):
        """
        Initialize the triplet dataset
        
        Args:
            benign_paths: List of file paths to benign images
            malignant_paths: List of file paths to malignant images
            transform: Optional transforms to apply to images
            num_triplets: Number of triplets to generate
            seed: Random seed for reproducibility
        """
        self.transform = transform
        self.benign_paths = benign_paths
        self.malignant_paths = malignant_paths
        self.num_triplets = num_triplets
        # Initialize random number generator with seed
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate triplets for consistent dataset
        self.triplets = []
        self._generate_triplets()
    
    def _generate_triplets(self):
        """Generate triplets: anchor, positive, negative"""
        for _ in range(self.num_triplets):
            # 50% chance anchor is benign, 50% malignant
            if self.rng.random() < 0.5:
                # Anchor and positive are benign, negative is malignant
                anchor, positive = self.rng.choice(self.benign_paths, size=2, replace=False)
                negative = self.rng.choice(self.malignant_paths)
                label = 0  # benign
            else:
                # Anchor and positive are malignant, negative is benign
                anchor, positive = self.rng.choice(self.malignant_paths, size=2, replace=True)
                negative = self.rng.choice(self.benign_paths)
                label = 1  # malignant
            
            # Store the triplet with its label
            self.triplets.append((anchor, positive, negative, label))
    
    def __len__(self):
        """Return the total number of triplets"""
        return self.num_triplets
    
    def __getitem__(self, idx):
        """
        Get a single triplet sample
        
        Args:
            idx: Index of the triplet to retrieve
            
        Returns:
            Tuple of (anchor, positive, negative, label)
        """
        # Unpack the triplet paths and label
        anchor_path, positive_path, negative_path, label = self.triplets[idx]
        
        # Load images and convert to RGB
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")
        
        # Apply transforms if provided
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        # Return triplet with label as tensor
        return anchor, positive, negative, torch.tensor(label, dtype=torch.long)