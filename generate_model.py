"""
Quick script to generate a model file so the app can run
This creates an untrained VGG16 model with random weights
"""
import torch
from torchvision import models
import os

# Create model directory if it doesn't exist (relative path for portability)
model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

# Create VGG16 model with 4 output classes
print("Creating VGG16 model...")
model = models.vgg16(pretrained=True)  # Load pretrained weights as base
model.classifier[6] = torch.nn.Linear(4096, 4)  # Modify final layer for 4 classes

# Save the model
model_path = os.path.join(model_dir, '40.pth')
torch.save(model.state_dict(), model_path)

file_size = os.path.getsize(model_path) / (1024**2)
print(f"✅ Model file created successfully!")
print(f"✅ Location: {model_path}")
print(f"✅ Size: {file_size:.2f} MB")
print("\n⚠️  Note: This is a base model with random classifier weights.")
print("For best results, train the model using 'waste classifier.ipynb'")
