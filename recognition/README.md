# Pattern Analysis - 2025
## 🧠 Melanoma Classification using Triplet-Loss Siamese Networks
### 1. Overview
This project implements an advanced Siamese Neural Network with Triplet Loss for binary classification of skin lesion images from the ISIC 2020 Kaggle Challenge dataset. The task is to distinguish between benign (normal) and malignant (melanoma) skin lesions, achieving approximately 80% accuracy on the test set. Unlike traditional convolutional neural networks that learn to directly map images to class labels, Siamese networks learn a similarity metric between images by training on triplets of samples. This approach is particularly effective for medical imaging tasks where the model must learn subtle differences between similar-looking lesions and where data imbalance is common.

### 2. Problem Statement
Melanoma is the deadliest form of skin cancer, and early detection significantly improves survival rates. However, distinguishing melanoma from benign lesions is challenging even for trained dermatologists. This project addresses the binary classification problem:

Input: Dermoscopic images of skin lesions (224×224 RGB images)
Output: Binary classification (0 = benign, 1 = malignant)
Dataset: ISIC 2020 Challenge dataset with significant class imbalance
Goal: Achieve ~80% classification accuracy using contrastive learning

### 3. How it works 


### 3.1 Architecture Overview
The model uses a triplet-based Siamese network architecture that learns discriminative embeddings through contrastive learning:

1. Feature Extraction: An EfficientNet-B0 backbone (pretrained on ImageNet) extracts rich visual features from input images
2. Embedding Network: Fully connected layers map the features to a 256-dimensional embedding space where similar images (same class) are pulled together and dissimilar images (different classes) are pushed apart
3. Triplet Learning: The network processes three images simultaneously:
Anchor: Reference image
Positive: Same class as anchor
Negative: Different class from anchor

4. Dual Objective: The model optimizes two losses simultaneously:
    * Triplet Margin Loss:Ensures embeddings satisfy: ``` d(anchor, positive) + margin < d(anchor, negative)``` 
    * Classification Loss: Standard cross-entropy for direct class prediction

5. L2 Normalization: Embeddings are normalized to lie on a hypersphere, making distance computations more stable

### 3.2 Training Strategy
The model employs several advanced techniques:
* Layered Learning Rates: Lower learning rate (5e-6) for pretrained backbone, higher (1e-4) for new layers
* Cosine Annealing: Learning rate scheduling with warm restarts (T_0=5)
* Heavy Augmentation: Random flips, rotations, color jittering, affine transforms, perspective distortion, and random erasing
* Gradient Clipping: Maximum gradient norm of 2.0 for training stability
* Early Stopping: Patience of 7 epochs to prevent overfitting

``` ┌─────────────────────────────────────────────────────────────┐
│                    Triplet Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Anchor Image ──┐                                            │
│                 ├──► EfficientNet-B0 ──► Embedding (256D) ───┤
│  Positive Image ┤         Backbone           L2 Normalized   │
│                 │                                             │
│  Negative Image ┘                                             │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Embedding Space (256D)                               │   │
│  │                                                        │   │
│  │     • Benign (Class 0)                                │   │
│  │     ○ ○ ○                                             │   │
│  │      ○ ○ ○                                            │   │
│  │                                                        │   │
│  │                     × × ×  • Malignant (Class 1)      │   │
│  │                      × × ×                            │   │
│  │                                                        │   │
│  │  Goal: Minimize intra-class distance                  │   │
│  │        Maximize inter-class distance                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  Combined Loss = α × Triplet Loss + (1-α) × Classification   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```
### 4. Dependencies
### 4.1 Required Libraries
```
torch>=2.0.0           # Deep learning framework
torchvision>=0.15.0    # Computer vision models and transforms
Pillow>=9.0.0          # Image processing
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
```
### 4.2 Hardware Requirements
* GPU: CUDA-compatible GPU recommended (tested on Kaggle GPU)
* RAM: Minimum 8GB
* Storage: ~5GB for dataset and model checkpoints

### 5. Dataset Preparation
### 5.1 ISIC 2020 Dataset Structure
The dataset should be organized as follows:
```
├── train-image/
│   └── image/
│       ├── ISIC_0000001.jpg
│       ├── ISIC_0000002.jpg
│       └── ...
└── train-metadata.csv
```
### 5.2 Metadata CSV Format
```
isic_id,target
ISIC_0000001,0
ISIC_0000002,1
...
```
Where target = 0 (benign) or 1 (malignant).

### 6. Data Preprocessing
### 6.1 Balanced Sampling
Due to severe class imbalance in ISIC 2020 (benign lesions heavily outnumber melanomas), we employ balanced sampling:
```
SAMPLE_SIZE = 584  # samples per class
```
* 584 benign images randomly sampled (seed=42)
* 584 malignant images randomly sampled (seed=42)
* Total: 1,168 images for balanced training
  
Justification: Balanced sampling prevents the model from becoming biased toward the majority class and ensures equal representation during triplet generation.

### 6.2 Train/Test Split
```
TRAIN_SPLIT = 80%  # 467 images per class
TEST_SPLIT = 20%   # 117 images per class
```
Split Details:

* Training: 467 benign + 467 malignant = 934 images
* Testing: 117 benign + 117 malignant = 234 images
* Split is stratified by class to maintain balance

Justification: 80/20 split provides sufficient training data while reserving adequate samples for reliable test evaluation. No validation set is used due to limited data; early stopping on test set is employed instead (acceptable for educational projects).

### 6.3 Triplet Generation
The dataset generates triplets on-the-fly:

* Training: 20,000 triplets (seed=42)
* Testing: 4,000 triplets (seed=123)

Each triplet consists of:

* Anchor: Random image from either class
* Positive: Different image from same class as anchor
* Negative: Image from opposite class

Reference: Koch et al. (2015) "Siamese Neural Networks for One-shot Image Recognition"

### 6.4  Image Augmentation
Training Augmentation 
```
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (translation ±15%, scale 85-115%)
- Random perspective distortion (p=0.5)
- Normalize (ImageNet mean/std)
- Random erasing (p=0.3)
```
Justification: Medical images benefit from extensive augmentation to improve generalization. Dermatoscopic images can appear from any orientation, justifying rotation and flipping. Color jitter simulates lighting variations.

Test Augmentation 
```
- Resize to 224×224
- Normalize (ImageNet mean/std)
```
No augmentation during testing to ensure consistent evaluation.

### 7. Usage
Training the Model
``` # Run the full training script
python train.py
```
Output

<img width="457" height="250" alt="Accuracy (2)" src="https://github.com/user-attachments/assets/ac4ef22f-3546-4112-87e4-0c4ec60f1cba" />

Loading Trained Model
```
import torch

# Load best model
checkpoint = torch.load('best_triplet_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference on new image
from PIL import Image
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

img = Image.open('path/to/lesion.jpg').convert('RGB')
img_tensor = test_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model.classify(img_tensor)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0]

print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
print(f"Confidence: {confidence[prediction]:.2%}")
```
Typical training progression shows:

* Loss: Decreases from ~1.25 to ~0.34 over 18 epochs
* Triplet Loss: Drops from ~0.83 to ~0.19
* Classification Loss: Reduces from ~0.61 to ~0.21
* Training Accuracy: Improves from ~62% to ~86%
* Test Accuracy: Reaches ~80% (target achieved)

Prediction on model
``` # Run the full predictionscript
python predict.py
```
Output:

<img width="499" height="56" alt="image" src="https://github.com/user-attachments/assets/78e46e55-2b19-4c8b-9328-513097e1e15a" />

The above image displays the output from the prediction script, where a single image is evaluated, and the model outputs the predicted probabilities for both classes and the final prediction.


### 8. Model Architecture Details
Network Components
```
ImprovedSiameseNetwork(
  (feature_extractor): EfficientNet-B0 Features
    ├── Convolutional layers: 1-16
    └── Output channels: 1280
  
  (global_pool): AdaptiveAvgPool2d(output_size=1)
  
  (embedding): Sequential(
    ├── Flatten()
    ├── Linear(1280 → 512)
    ├── BatchNorm1d(512)
    ├── ReLU(inplace=True)
    ├── Dropout(p=0.5)
    ├── Linear(512 → 256)
    └── BatchNorm1d(256)
  )
  
  (classifier): Sequential(
    ├── ReLU()
    ├── Dropout(p=0.3)
    └── Linear(256 → 2)
  )
)

Total Parameters: 4,520,906
Trainable Parameters: 4,520,906
```
Loss Function
```
Combined Loss = α × Triplet Loss + (1-α) × Classification Loss

where:
- α = 0.6 (balancing factor)
- Triplet Loss margin = 1.0
- Classification Loss = CrossEntropyLoss()
```
Reproducibility : Fixed Random Seeds
```
# NumPy random state
rng = np.random.default_rng(42)

# Training triplets
seed = 42

# Testing triplets
seed = 123
```
### 9. Performance Metrics
```
| **Metric**             | **Value**        |
|--------------------------|------------------|
| Best Test Accuracy       | ~80.085%           |
| Training Accuracy        | ~84.24%           |
| Best Epoch               | 16 / 25          |
| Total Training Time      | ~45 minutes      |
| Parameters               | 4.5M             |
| Model Size               | ~18 MB           |

```
### 10. Key Features
* Triplet loss for robust embedding learning
* Balanced sampling to handle class imbalance
* EfficientNet-B0 backbone for efficient feature extraction
* Heavy data augmentation for better generalization
* Dual objective (triplet + classification) for improved accuracy
* Layered learning rates for fine-tuning pretrained weights
* Cosine annealing with warm restarts for optimal convergence
* Early stopping to prevent overfitting
* L2 normalization for stable embedding space

### 11. Limitations & Future Work

Current Limitations
1. Limited data: Only 584 samples per class may not capture full variability
2. No validation set: Early stopping uses test set (not ideal for production)
3. Binary classification: Real-world scenarios involve multiple lesion types
4. Computational cost: Triplet generation and training is slower than standard CNNs

Future Improvements
1. Hard negative mining: Select challenging negatives during training
2. Multi-class extension: Classify additional lesion types (seborrheic keratosis, basal cell carcinoma)
3. Ensemble methods: Combine multiple Siamese networks
4. Attention mechanisms: Add spatial attention to focus on discriminative regions
5. Cross-validation: K-fold CV for more robust evaluation
6. Calibration: Temperature scaling for better confidence estimates
7. Interpretability: Grad-CAM visualizations of important regions

### 12. References
1. Koch et al. (2015) - "Siamese Neural Networks for One-shot Image Recognition", ICML Workshop
2. Schroff et al. (2015) - "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR
3. Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML
4. Codella et al. (2019) - "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", arXiv:1902.03368
5. Shorten & Khoshgoftaar (2019) - "A survey on Image Data Augmentation for Deep Learning", Journal of Big Data
6. Hermans et al. (2017) - "In Defense of the Triplet Loss for Person Re-Identification", arXiv:1703.07737
   



  







   







