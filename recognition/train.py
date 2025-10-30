import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from modules import ImprovedSiameseNetwork, CombinedLoss
from dataset import TripletMelanomaDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "train-image\image"
TRAIN_CSV_PATH = "train-metadata.csv"
SAMPLE_SIZE = 584  
TRAIN_SPLIT = 584
BATCH_SIZE = 32
IMG_SIZE = 224

# --- Split sampled images into train/test ---
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"Total images in metadata: {len(train_df)}")

# Separate benign and malignant IDs
benign_ids = train_df[train_df['target'] == 0]['isic_id'].values
malignant_ids = train_df[train_df['target'] == 1]['isic_id'].values

print(f"No of Benign images: {len(benign_ids)}, No of Malignant images: {len(malignant_ids)}")

rng = np.random.default_rng(42)

sampled_benign = rng.choice(benign_ids, SAMPLE_SIZE, replace=False)
sampled_malignant = rng.choice(malignant_ids, SAMPLE_SIZE, replace=False)
benign_paths = [os.path.join(IMAGE_DIR, f"{img_id}.jpg") for img_id in sampled_benign]
malignant_paths = [os.path.join(IMAGE_DIR, f"{img_id}.jpg") for img_id in sampled_malignant]

TRAIN_SPLIT = int(0.8 * SAMPLE_SIZE)  # 80% train, 20% test

train_benign = benign_paths[:TRAIN_SPLIT]
test_benign  = benign_paths[TRAIN_SPLIT:]

train_malignant = malignant_paths[:TRAIN_SPLIT]
test_malignant  = malignant_paths[TRAIN_SPLIT:]

print(f"Train benign: {len(train_benign)}, Test benign: {len(test_benign)}")
print(f"Train malignant: {len(train_malignant)}, Test malignant: {len(test_malignant)}")


# ============= ENHANCED TRANSFORMS WITH MORE AUGMENTATION =============
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============= CREATE DATASETS =============
triplet_train_dataset = TripletMelanomaDataset(
    train_benign, train_malignant,
    transform=train_transform,
    num_triplets=20000,
    seed=42
)

triplet_test_dataset = TripletMelanomaDataset(
    test_benign, test_malignant,
    transform=test_transform,
    num_triplets=4000,
    seed=123
)

train_loader = DataLoader(triplet_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(triplet_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============= MODEL SETUP =============
model = ImprovedSiameseNetwork(embedding_dim=256).to(device)

# Initialize new layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.embedding.apply(init_weights)
model.classifier.apply(init_weights)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============= TRAINING SETUP =============
criterion = CombinedLoss(margin=1.0, alpha=0.6)

# Layerwise learning rates
optimizer = torch.optim.AdamW([
    {'params': model.feature_extractor.parameters(), 'lr': 5e-6},
    {'params': model.embedding.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1, eta_min=1e-7
)

num_epochs = 25


# ============= EVALUATION FUNCTION =============
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, positive, negative, labels in data_loader:
            anchor = anchor.to(device)
            labels = labels.to(device)
            
            # Get classification predictions
            logits = model.classify(anchor)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store embeddings
            embeds = model.forward_once(anchor)
            all_embeddings.append(embeds.cpu())
            all_labels.append(labels.cpu())
    
    accuracy = correct / total
    model.train()
    return accuracy


# ============= TRAINING LOOP =============
best_acc = 0.0
patience_counter = 0
patience_limit = 7

print("\nTraining Started for 25 epochs...\n")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_triplet = 0.0
    epoch_class = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels = labels.to(device)
        
        # Forward pass
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
        class_logits = model.classifier(anchor_embed)
        
        # Calculate loss
        loss, triplet_loss, class_loss = criterion(
            anchor_embed, positive_embed, negative_embed, labels, class_logits
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += loss.item()
        epoch_triplet += triplet_loss.item()
        epoch_class += class_loss.item()
        
        # Calculate training accuracy
        with torch.no_grad():
            preds = torch.argmax(class_logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
    
    # Step scheduler
    scheduler.step()
    
    # Evaluation
    train_acc = correct_train / total_train
    test_acc = evaluate_model(model, test_loader, device)
    
    avg_loss = epoch_loss / len(train_loader)
    avg_triplet = epoch_triplet / len(train_loader)
    avg_class = epoch_class / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Loss: {avg_loss:.4f} (Triplet: {avg_triplet:.4f}, Class: {avg_class:.4f})")
    print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, 'best_triplet_model.pth')
        print(f"  ✓ New best model saved! (Acc: {best_acc:.4f})")
    else:
        patience_counter += 1
    
    if patience_counter >= patience_limit:
        print(f"\n  Early stopping after {patience_limit} epochs without improvement")
        break
    print()

print(f"\n{'='*50}")
print(f"Training Completed!")
print(f"Best Test Accuracy: {best_acc:.4f}")
print(f"{'='*50}")