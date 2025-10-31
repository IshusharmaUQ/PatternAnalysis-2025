import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from modules import ImprovedSiameseNetwork, CombinedLoss
from dataset import TripletMelanomaDataset


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= CONFIGURATION =============
IMAGE_DIR = "train-image\image"  # Directory containing training images
TRAIN_CSV_PATH = "train-metadata.csv"  # Path to metadata CSV file
SAMPLE_SIZE = 584  # Number of samples to use from each class
TRAIN_SPLIT = 584  # Will be recalculated as 80% of SAMPLE_SIZE
BATCH_SIZE = 32  # Number of samples per batch
IMG_SIZE = 224  # Image dimension (224x224)

# ============= DATA LOADING AND SPLITTING =============
# Load metadata CSV containing image IDs and labels
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"Total images in metadata: {len(train_df)}")

# Separate benign (target=0) and malignant (target=1) image IDs
benign_ids = train_df[train_df['target'] == 0]['isic_id'].values
malignant_ids = train_df[train_df['target'] == 1]['isic_id'].values

print(f"No of Benign images: {len(benign_ids)}, No of Malignant images: {len(malignant_ids)}")

# Initialize random number generator with fixed seed for reproducibility
rng = np.random.default_rng(42)

# Randomly sample equal number of benign and malignant images
sampled_benign = rng.choice(benign_ids, SAMPLE_SIZE, replace=False)
sampled_malignant = rng.choice(malignant_ids, SAMPLE_SIZE, replace=False)

# Create full file paths for sampled images
benign_paths = [os.path.join(IMAGE_DIR, f"{img_id}.jpg") for img_id in sampled_benign]
malignant_paths = [os.path.join(IMAGE_DIR, f"{img_id}.jpg") for img_id in sampled_malignant]

# Calculate train/test split (80% train, 20% test)
TRAIN_SPLIT = int(0.8 * SAMPLE_SIZE)

# Split benign images into train and test sets
train_benign = benign_paths[:TRAIN_SPLIT]
test_benign  = benign_paths[TRAIN_SPLIT:]

# Split malignant images into train and test sets
train_malignant = malignant_paths[:TRAIN_SPLIT]
test_malignant  = malignant_paths[TRAIN_SPLIT:]

print(f"Train benign: {len(train_benign)}, Test benign: {len(test_benign)}")
print(f"Train malignant: {len(train_malignant)}, Test malignant: {len(test_malignant)}")


# ============= ENHANCED TRANSFORMS WITH MORE AUGMENTATION =============
# Training transforms with aggressive data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),  # Vertical flip with 50% probability
    transforms.RandomRotation(30),  # Random rotation up to ±30 degrees
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Color variation
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # Translation and scaling
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective distortion
    transforms.ToTensor(),  # Convert to tensor [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Random erasing for robustness
])

# Test transforms without augmentation (only resize and normalize)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


# ============= CREATE DATASETS =============
# Create training dataset with 20,000 triplets
triplet_train_dataset = TripletMelanomaDataset(
    train_benign, train_malignant,
    transform=train_transform,
    num_triplets=20000,  # Number of triplets to generate
    seed=42  # Random seed for reproducibility
)

# Create test dataset with 4,000 triplets
triplet_test_dataset = TripletMelanomaDataset(
    test_benign, test_malignant,
    transform=test_transform,
    num_triplets=4000,  # Number of triplets to generate
    seed=123  # Different seed for test set
)

# Create data loaders for batching
train_loader = DataLoader(triplet_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(triplet_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============= MODEL SETUP =============
# Initialize Siamese network with 256-dimensional embeddings
model = ImprovedSiameseNetwork(embedding_dim=256).to(device)

# Weight initialization function for newly added layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        # Initialize batch normalization layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Apply weight initialization to embedding and classifier layers
model.embedding.apply(init_weights)
model.classifier.apply(init_weights)

# Print model parameter counts
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============= TRAINING SETUP =============
# Combined loss function with triplet loss and classification loss
criterion = CombinedLoss(margin=1.0, alpha=0.6)  # margin for triplet loss, alpha for loss weighting

# AdamW optimizer with layerwise learning rates
optimizer = torch.optim.AdamW([
    {'params': model.feature_extractor.parameters(), 'lr': 5e-6},  # Lower LR for pretrained backbone
    {'params': model.embedding.parameters(), 'lr': 1e-4},  # Higher LR for embedding layer
    {'params': model.classifier.parameters(), 'lr': 1e-4}  # Higher LR for classifier
], weight_decay=1e-4)  # L2 regularization

# Cosine annealing scheduler with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1, eta_min=1e-7  # Restart every 5 epochs, minimum LR of 1e-7
)

num_epochs = 25  # Total number of training epochs


# ============= EVALUATION FUNCTION =============
def evaluate_model(model, data_loader, device):
    """
    Evaluate model on given data loader
    Returns classification accuracy
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for anchor, positive, negative, labels in data_loader:
            anchor = anchor.to(device)
            labels = labels.to(device)
            
            # Get classification predictions from anchor images
            logits = model.classify(anchor)
            preds = torch.argmax(logits, dim=1)
            
            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store embeddings for potential analysis
            embeds = model.forward_once(anchor)
            all_embeddings.append(embeds.cpu())
            all_labels.append(labels.cpu())
    
    accuracy = correct / total
    model.train()  # Set model back to training mode
    return accuracy


# ============= TRAINING LOOP =============
best_acc = 0.0  # Track best test accuracy
patience_counter = 0  # Counter for early stopping
patience_limit = 7  # Stop if no improvement for 7 epochs

print("\nTraining Started for 25 epochs...\n")

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0  # Accumulator for total loss
    epoch_triplet = 0.0  # Accumulator for triplet loss
    epoch_class = 0.0  # Accumulator for classification loss
    correct_train = 0  # Count correct predictions
    total_train = 0  # Total training samples
    
    # Iterate through training batches
    for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
        # Move data to device
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels = labels.to(device)
        
        # Forward pass through Siamese network
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
        # Get classification logits
        class_logits = model.classifier(anchor_embed)
        
        # Calculate combined loss (triplet + classification)
        loss, triplet_loss, class_loss = criterion(
            anchor_embed, positive_embed, negative_embed, labels, class_logits
        )
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
        optimizer.step()  # Update weights
        
        # Accumulate losses for logging
        epoch_loss += loss.item()
        epoch_triplet += triplet_loss.item()
        epoch_class += class_loss.item()
        
        # Calculate training accuracy
        with torch.no_grad():
            preds = torch.argmax(class_logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
    
    # Step the learning rate scheduler
    scheduler.step()
    
    # Calculate accuracies
    train_acc = correct_train / total_train
    test_acc = evaluate_model(model, test_loader, device)
    
    # Calculate average losses
    avg_loss = epoch_loss / len(train_loader)
    avg_triplet = epoch_triplet / len(train_loader)
    avg_class = epoch_class / len(train_loader)
    
    # Print epoch statistics
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Loss: {avg_loss:.4f} (Triplet: {avg_triplet:.4f}, Class: {avg_class:.4f})")
    print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model based on test accuracy
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0  # Reset patience counter
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, 'best_triplet_model.pth')
        print(f"  Best model Updated (Acc: {best_acc:.4f})")
    else:
        patience_counter += 1  # Increment patience counter
    
    # Early stopping check
    if patience_counter >= patience_limit:
        print(f"\n  Early stopping after {patience_limit} epochs without improvement")
        break
    print()

# Print final training summary
print(f"\n{'='*50}")
print(f"Training Completed!")
print(f"Best Test Accuracy: {best_acc:.4f}")
print(f"{'='*50}")