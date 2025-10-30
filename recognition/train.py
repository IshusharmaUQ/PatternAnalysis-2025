# ===============================================
# Training, Validation, Testing, and Model Saving
# ===============================================
# Imports model from modules.py and data loader from dataset.py

import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modules import TripletSiamese, CombinedLoss
from dataset import build_loaders, TRAIN_CSV, IMAGE_DIR


# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Hyperparameters
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "siamese_model.pth"


# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv(TRAIN_CSV)
train_loader, val_loader = build_loaders(
    df, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
)


# -------------------------------
# Initialize Model, Loss, Optimizer, and Scheduler
# -------------------------------
model = TripletSiamese(
    embedding_dim=256, backbone='resnet18', pretrained=True
).to(device)

criterion = CombinedLoss(alpha=1.0, beta=1.0, margin=0.3)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# -------------------------------
# Training and Validation
# -------------------------------
train_losses = []
val_losses = []
best_val_loss = np.inf

for epoch in range(NUM_EPOCHS):
    # ---- Training ----
    model.train()
    running_loss = 0.0

    for i, (anchor, pos, neg, label) in enumerate(train_loader):
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()
        a, p, n = model(anchor, pos, neg)
        loss = criterion(a=a, p=p, n=n)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- Validation ----
    model.eval()
    running_val = 0.0

    with torch.no_grad():
        for anchor, pos, neg, label in val_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            a, p, n = model(anchor, pos, neg)
            loss = criterion(a=a, p=p, n=n)
            running_val += loss.item()

    avg_val_loss = running_val / len(val_loader)
    val_losses.append(avg_val_loss)

    # ---- Logging ----
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- Scheduler Step and Model Saving ----
    scheduler.step()
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved best model to {MODEL_SAVE_PATH} (val loss {best_val_loss:.4f})")


# -------------------------------
# Plot and Save Loss Curve
# -------------------------------
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')
print('Saved loss curve to loss_curve.png')
