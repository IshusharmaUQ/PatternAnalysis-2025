import torch
from torchvision import transforms
from PIL import Image

from modules import ImprovedSiameseNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# Transform for prediction
predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(checkpoint_path, embedding_dim=256):
    """Load the trained model from checkpoint"""
    model = ImprovedSiameseNetwork(embedding_dim=embedding_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_image(model, image_path, transform=predict_transform):
    """Predict if an image is benign (0) or malignant (1)"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model.classify(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction, probabilities[0].cpu().numpy()


if __name__ == "__main__":
    # Example usage
    model = load_model('best_triplet_model.pth')
    
    # Example prediction
    image_path = "train-image\image\ISIC_0015719.jpg"
    prediction, probabilities = predict_image(model, image_path)
    
    print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
    print(f"Probabilities - Benign: {probabilities[0]:.4f}, Malignant: {probabilities[1]:.4f}")