import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Convolutional feature extractor
        self.conv_block1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=10)
        self.conv_block2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7)
        self.conv_block3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4)
        self.conv_block4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        
        # Fully connected layers for embedding generation
        self.fc_embedding = nn.Linear(256 * 6 * 6, 4096)
        self.fc_output = nn.Linear(4096, 1)

    def forward_once(self, image):
        """
        Forward pass for a single image through the Siamese branch.
        Produces an embedding vector representing the image features.
        """
        x = F.relu(F.max_pool2d(self.conv_block1(image), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv_block2(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv_block3(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv_block4(x), kernel_size=2))
        
        # Flatten the convolutional features into a vector
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc_embedding(x))
        x = self.fc_output(x)
        
        return x

    def forward(self, img_left, img_right):
        """
        Forward pass for a pair of images.
        Each image is processed independently through the same network.
        Returns embeddings for both.
        """
        embedding_left = self.forward_once(img_left)
        embedding_right = self.forward_once(img_right)
        return embedding_left, embedding_right


def contrastive_loss(embedding_left, embedding_right, label, margin=1.0):
    """
    Computes the Contrastive Loss between two embeddings.

    Args:
        embedding_left, embedding_right: Output vectors from the Siamese branches.
        label: 0 if images are from the same class, 1 if they are from different classes.
        margin: Minimum distance enforced between embeddings of dissimilar pairs.
    """
    # Compute Euclidean distance between the embeddings
    distance = F.pairwise_distance(embedding_left, embedding_right)
    
    # Compute contrastive loss as per Hadsell et al. (2006)
    loss = torch.mean(
        (1 - label) * torch.pow(distance, 2) +
        (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    )
    return loss
