import torch
from torchvision import transforms
from PIL import Image

from modules import ImprovedSiameseNetwork


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224  # Image dimension for prediction (must match training size)

# Transform pipeline for prediction (no augmentation, only normalization)
predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert PIL image to tensor [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


def load_model(checkpoint_path, embedding_dim=256):
    """
    Load the trained Siamese model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the saved model checkpoint (.pth file)
        embedding_dim: Dimension of the embedding layer (default: 256)
    
    Returns:
        model: Loaded model in evaluation mode
    """
    # Initialize model architecture
    model = ImprovedSiameseNetwork(embedding_dim=embedding_dim).to(device)
    
    # Load checkpoint containing model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the saved state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode (disables dropout, batch norm in eval mode)
    model.eval()
    
    return model


def predict_image(model, image_path, transform=predict_transform):
    """
    Predict if a skin lesion image is benign (0) or malignant (1)
    
    Args:
        model: Trained Siamese network model
        image_path: Path to the image file to classify
        transform: Transform pipeline to apply to the image
    
    Returns:
        prediction: Integer label (0 for benign, 1 for malignant)
        probabilities: Numpy array of class probabilities [benign_prob, malignant_prob]
    """
    # Load and convert image to RGB format
    image = Image.open(image_path).convert("RGB")
    
    # Apply transforms and add batch dimension [1, C, H, W]
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference without computing gradients
    with torch.no_grad():
        # Get raw classification logits
        logits = model.classify(image_tensor)
        
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted class (0 or 1)
        prediction = torch.argmax(logits, dim=1).item()
    
    # Return prediction and probabilities as numpy array
    return prediction, probabilities[0].cpu().numpy()


if __name__ == "__main__":
    # Example usage demonstrating model loading and prediction
    
    # Load the trained model from checkpoint
    model = load_model('best_triplet_model.pth')
    
    # Path to the image to classify
    image_path = "train-image\image\ISIC_0015719.jpg"
    
    # Perform prediction
    prediction, probabilities = predict_image(model, image_path)
    
    # Display results
    print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
    print(f"Probabilities - Benign: {probabilities[0]:.4f}, Malignant: {probabilities[1]:.4f}")