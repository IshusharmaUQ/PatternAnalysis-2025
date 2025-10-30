# Model components and loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Small convolutional branch (used as fallback)
class SimpleConvBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Siamese network using ResNet18 backbone 
class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            b = models.resnet18(pretrained=pretrained)
            modules = list(b.children())[:-1]  # drop final fc
            self.backbone = nn.Sequential(*modules)
            feat_dim = 512
        else:
            # fallback small conv
            self.backbone = SimpleConvBranch(out_dim=embedding_dim)
            feat_dim = embedding_dim

        if backbone == 'resnet18':
            self.fc = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, embedding_dim))
        else:
            self.fc = nn.Identity()

    def forward_once(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return e1, e2

# Triplet variant that returns anchor, positive and negative embeddings
class TripletSiamese(nn.Module):
    def __init__(self, embedding_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        self.siamese = ImprovedSiameseNetwork(embedding_dim, backbone, pretrained)

    def forward(self, anchor, positive, negative):
        a = self.siamese.forward_once(anchor)
        p = self.siamese.forward_once(positive)
        n = self.siamese.forward_once(negative)
        return a, p, n

# Contrastive loss 
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        # label: 1 -> similar, 0 -> dissimilar 
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

# Triplet loss wrapper
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, a, p, n):
        return self.loss(a, p, n)

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, margin=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.trip = TripletLoss(margin=margin)
        self.contrast = ContrastiveLoss(margin=1.0)

    def forward(self, a=None, p=None, n=None, out1=None, out2=None, labels=None):
        loss = 0.0
        if (a is not None) and (p is not None) and (n is not None):
            loss = loss + self.alpha * self.trip(a, p, n)
        if (out1 is not None) and (out2 is not None) and (labels is not None):
            loss = loss + self.beta * self.contrast(out1, out2, labels)
        return loss
