# Training, validation, testing and saving.
# Imports model from modules.py and data loader from dataset.py

import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modules import TripletSiamese, CombinedLoss
from dataset import build_loaders, TRAIN_CSV, IMAGE_DIR

# device setting 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters 
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "siamese_model.pth"

# load dataframe 
df = pd.read_csv(TRAIN_CSV)

# build loaders 
train_loader, val_loader = build_loaders(df, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# instantiate model from modules.py 
model = TripletSiamese(embedding_dim=256, backbone='resnet18', pretrained=True).to(device)

# loss and optimizer 
criterion = CombinedLoss(alpha=1.0, beta=1.0, margin=0.3)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_losses = []
val_losses = []
best_val_loss = np.inf

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (anchor, pos, neg, label) in enumerate(train_loader):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        a, p, n = model(anchor, pos, neg)
        loss = criterion(a=a, p=p, n=n)  
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for anchor, pos, neg, label in val_loader:
            anchor = anchor.to(device); pos = pos.to(device); neg = neg.to(device)
            a, p, n = model(anchor, pos, neg)
            loss = criterion(a=a, p=p, n=n)
            running_val += loss.item()
    avg_val_loss = running_val / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

    # scheduler step and save best model
    scheduler.step()
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved best model to {MODEL_SAVE_PATH} (val loss {best_val_loss:.4f})")

# Plot losses 
plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train','val'])
plt.title('Training and Validation Loss')
plt.savefig('loss_curve.png')
print('Saved loss curve to loss_curve.png')
