import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import SiameseNetwork, contrastive_loss
from dataset import load_data  # Make sure function name matches your dataset file

# ------------------------------------------------------------
# Device setup (GPU if available, else CPU)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Initialize model, optimizer, and training parameters
# ------------------------------------------------------------
siamese_model = SiameseNetwork().to(device)
train_data_loader = load_data(image_dir="/path/to/images", train_csv="/path/to/train.csv")

optimizer = optim.Adam(siamese_model.parameters(), lr=0.0005)
num_training_epochs = 10
epoch_loss_history = []

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
for epoch_index in range(num_training_epochs):
    siamese_model.train()
    cumulative_epoch_loss = 0.0

    # Iterate over each batch of (image1, image2, label)
    for batch_data in train_data_loader:
        image_left, image_right, pair_label = batch_data
        image_left, image_right, pair_label = (
            image_left.to(device),
            image_right.to(device),
            pair_label.to(device),
        )

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: compute embeddings for both images
        embedding_left, embedding_right = siamese_model(image_left, image_right)

        # Compute contrastive loss based on similarity/dissimilarity
        batch_loss = contrastive_loss(embedding_left, embedding_right, pair_label)

        # Backpropagation and parameter update
        batch_loss.backward()
        optimizer.step()

        # Track total loss for this epoch
        cumulative_epoch_loss += batch_loss.item()

    # Compute average loss for the epoch
    average_epoch_loss = cumulative_epoch_loss / len(train_data_loader)
    epoch_loss_history.append(average_epoch_loss)

    # Log epoch progress
    print(f"Epoch [{epoch_index + 1}/{num_training_epochs}] - Loss: {average_epoch_loss:.4f}")

    # Save model checkpoint after each epoch
    torch.save(siamese_model.state_dict(), "siamese_model.pth")

# ------------------------------------------------------------
# Plot training loss curve
# ------------------------------------------------------------
plt.plot(epoch_loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Siamese Network Training Loss")
plt.show()
