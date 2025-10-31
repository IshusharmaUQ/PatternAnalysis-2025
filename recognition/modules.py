import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedSiameseNetwork(nn.Module):
    """
    Siamese Network for melanoma classification using triplet learning
    Combines metric learning (triplet loss) with classification
    """
    def __init__(self, embedding_dim=256):
        """
        Initialize the Siamese Network
        
        Args:
            embedding_dim: Dimension of the embedding space (default: 256)
        """
        super(ImprovedSiameseNetwork, self).__init__()
        
        # Load pretrained EfficientNet-B0 as feature extraction backbone
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = base_model.features  # Extract convolutional layers only
        
        # Global average pooling to reduce spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding network: maps features to embedding space
        # EfficientNet-B0 outputs 1280 features, reduced to embedding_dim
        self.embedding = nn.Sequential(
            nn.Flatten(),  # Flatten pooled features
            nn.Linear(1280, 512),  # First compression layer
            nn.BatchNorm1d(512),  # Batch normalization for stable training
            nn.ReLU(inplace=True),  # Non-linear activation
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, embedding_dim),  # Second compression to embedding dimension
            nn.BatchNorm1d(embedding_dim),  # Final batch normalization
        )
        
        # Classification head for binary classification (benign vs malignant)
        # Takes embeddings as input and outputs 2 class logits
        self.classifier = nn.Sequential(
            nn.ReLU(),  # Activation after embedding
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(embedding_dim, 2)  # Binary classification output
        )
    
    def forward_once(self, x):
        """
        Extract and normalize embedding for a single image
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
        
        Returns:
            Normalized embedding vector [batch_size, embedding_dim]
        """
        # Pass through feature extractor (EfficientNet backbone)
        x = self.feature_extractor(x)
        
        # Global average pooling: [batch, 1280, H, W] -> [batch, 1280, 1, 1]
        x = self.global_pool(x)
        
        # Pass through embedding network: [batch, 1280, 1, 1] -> [batch, embedding_dim]
        x = self.embedding(x)
        
        # L2 normalization: ensures embeddings lie on unit hypersphere
        # This is crucial for metric learning with cosine/euclidean distance
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet learning
        
        Args:
            anchor: Anchor image tensor [batch_size, 3, H, W]
            positive: Positive image (same class as anchor) [batch_size, 3, H, W]
            negative: Negative image (different class) [batch_size, 3, H, W]
        
        Returns:
            Tuple of (anchor_embed, positive_embed, negative_embed)
            Each embedding is [batch_size, embedding_dim]
        """
        # Extract embeddings for all three images in the triplet
        anchor_embed = self.forward_once(anchor)
        positive_embed = self.forward_once(positive)
        negative_embed = self.forward_once(negative)
        
        return anchor_embed, positive_embed, negative_embed
    
    def classify(self, x):
        """
        Classify an image using the embedding and classification head
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
        
        Returns:
            Classification logits [batch_size, 2]
        """
        # Extract embedding from input image
        embed = self.forward_once(x)
        
        # Pass through classification head to get class logits
        return self.classifier(embed)


class CombinedLoss(nn.Module):
    """
    Combined loss function using both triplet loss and classification loss
    Enables the model to learn both discriminative embeddings and direct classification
    """
    def __init__(self, margin=1.0, alpha=0.5):
        """
        Initialize the combined loss function
        
        Args:
            margin: Margin for triplet loss (default: 1.0)
                   Enforces minimum distance between anchor-negative and anchor-positive
            alpha: Weight for triplet loss (default: 0.5)
                   Final loss = alpha * triplet + (1-alpha) * classification
        """
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        
        # Triplet loss: encourages anchor to be closer to positive than to negative
        # Uses L2 (Euclidean) distance with specified margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
        # Cross-entropy loss for binary classification
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, anchor, positive, negative, labels, class_logits):
        """
        Calculate combined loss from triplet embeddings and classification
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            labels: Ground truth class labels [batch_size]
            class_logits: Classification logits [batch_size, 2]
        
        Returns:
            Tuple of (total_loss, triplet_loss_value, classification_loss_value)
        """
        # Calculate triplet loss: minimizes distance to positive, maximizes to negative
        triplet = self.triplet_loss(anchor, positive, negative)
        
        # Calculate classification loss: standard cross-entropy
        classification = self.ce_loss(class_logits, labels)
        
        # Combine losses with weighted average
        # alpha controls the balance between metric learning and classification
        total_loss = self.alpha * triplet + (1 - self.alpha) * classification
        
        # Return total loss and individual components for logging
        return total_loss, triplet, classification