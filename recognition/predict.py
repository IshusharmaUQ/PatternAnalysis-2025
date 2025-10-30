import torch
from modules import SiameseNetwork
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------
# Device setup (use GPU if available)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Load the trained Siamese model
# ------------------------------------------------------------
siamese_model = SiameseNetwork().to(device)
siamese_model.load_state_dict(torch.load("siamese_model.pth"))
siamese_model.eval()  # Set to evaluation mode (no dropout, no gradient updates)


def predict_similarity(image_path_left, image_path_right):
    """
    Given two image file paths, compute their similarity using the trained Siamese network.

    Args:
        image_path_left (str): Path to the first image.
        image_path_right (str): Path to the second image.
    """
    # Define preprocessing transformations
    image_preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess both input images
    image_left = Image.open(image_path_left)
    image_right = Image.open(image_path_right)

    image_left = image_preprocessing(image_left).unsqueeze(0).to(device)
    image_right = image_preprocessing(image_right).unsqueeze(0).to(device)

    # Forward pass through the Siamese model
    with torch.no_grad():  # Disable gradient computation for inference
        embedding_left, embedding_right = siamese_model(image_left, image_right)

        # Compute Euclidean distance between the embeddings
        similarity_distance = torch.nn.functional.pairwise_distance(embedding_left, embedding_right)

    # Print similarity score (lower = more similar)
    print(f"Euclidean Distance between images: {similarity_distance.item():.4f}")


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
predict_similarity("/path/to/image1.jpg", "/path/to/image2.jpg")
