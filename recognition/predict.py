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