# Example usage of trained model (loads siamese_model.pth saved by train.py)

import torch
from modules import ImprovedSiameseNetwork
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = "siamese_model.pth"
IMAGE_DIR = "/kaggle/input/isic-2020-jpg-224x224-resized/train"  
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# load model architecture matching train.py
model = ImprovedSiameseNetwork(embedding_dim=256, backbone='resnet18', pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_pair(path1, path2):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        e1 = model.forward_once(t1)
        e2 = model.forward_once(t2)
        dist = torch.nn.functional.pairwise_distance(e1, e2)
    print(f"Distance between {os.path.basename(path1)} and {os.path.basename(path2)}: {dist.item():.4f}")
    return dist.item()

if __name__ == '__main__':
    p1 = os.path.join(IMAGE_DIR, 'ISIC_0000000.jpg')  
    p2 = os.path.join(IMAGE_DIR, 'ISIC_0000001.jpg')
    if os.path.exists(p1) and os.path.exists(p2):
        predict_pair(p1, p2)
    else:
        print("Example files not found in IMAGE_DIR. Replace p1/p2 with actual filenames from your dataset.")
