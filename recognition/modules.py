import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(ImprovedSiameseNetwork, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = base_model.features
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding network with attention
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Classification head (for auxiliary task)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 2)  # Binary classification
        )
    
    def forward_once(self, x):
        """Extract features for one image"""
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
    
    def forward(self, anchor, positive, negative):
        """Forward pass for triplet"""
        anchor_embed = self.forward_once(anchor)
        positive_embed = self.forward_once(positive)
        negative_embed = self.forward_once(negative)
        return anchor_embed, positive_embed, negative_embed
    
    def classify(self, x):
        """Classification from embeddings"""
        embed = self.forward_once(x)
        return self.classifier(embed)


class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, anchor, positive, negative, labels, class_logits):
        # Triplet loss
        triplet = self.triplet_loss(anchor, positive, negative)
        
        # Classification loss
        classification = self.ce_loss(class_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * triplet + (1 - self.alpha) * classification
        return total_loss, triplet, classification