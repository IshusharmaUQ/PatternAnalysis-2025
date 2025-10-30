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
        """
        Args:
            image_file_paths (list): List of full paths to image files.
            class_labels (list): Corresponding list of integer labels (0 or 1).
            transform (callable, optional): Optional image transformations.
        """
        self.image_file_paths = image_file_paths
        self.class_labels = class_labels
        self.transform = transform

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_file_paths)

    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding label at the given index.
        Applies transformation if specified.
        """
        image_path = self.image_file_paths[index]
        image = Image.open(image_path)
        label = self.class_labels[index]

        # Apply transformations (resize, normalize, etc.) if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def load_isic_data(image_directory, csv_train_path, batch_size=32, image_size=224):
    """
    Loads and balances the ISIC dataset, returning a PyTorch DataLoader.

    Args:
        image_directory (str): Path to directory containing all ISIC images.
        csv_train_path (str): Path to the CSV file containing image IDs and labels.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        image_size (int, optional): Resize dimension for input images. Default is 224.

    Returns:
        DataLoader: A PyTorch DataLoader containing balanced image-label pairs.
    """
    # Load the CSV file containing image IDs and their corresponding target labels
    dataframe = pd.read_csv(csv_train_path)

    # Split IDs into benign and malignant groups
    benign_image_ids = dataframe[dataframe['target'] == 0]['isic_id'].values
    malignant_image_ids = dataframe[dataframe['target'] == 1]['isic_id'].values

    # Ensure dataset is balanced by taking equal samples from each class
    balanced_sample_size = min(len(benign_image_ids), len(malignant_image_ids))

    # Construct full image paths for both benign and malignant samples
    benign_image_paths = [os.path.join(image_directory, f"{img_id}.jpg") 
                          for img_id in benign_image_ids[:balanced_sample_size]]
    malignant_image_paths = [os.path.join(image_directory, f"{img_id}.jpg") 
                             for img_id in malignant_image_ids[:balanced_sample_size]]

    # Define image preprocessing transformations
    image_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # Combine benign and malignant samples into one dataset
    train_dataset = ISICDataset(
        image_file_paths=benign_image_paths + malignant_image_paths,
        class_labels=[0] * len(benign_image_paths) + [1] * len(malignant_image_paths),
        transform=image_transforms
    )

    # Create a DataLoader to efficiently load images in batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
