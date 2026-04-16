#!/usr/bin/env python
"""
SmartBin Model Training Script
Trains VGG16 CNN on waste classification dataset
Expected accuracy: 70-80% after 40 epochs
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from PIL import Image
import time


# ==================== Dataset Class ====================
class WasteDataset(Dataset):
    """Custom dataset for loading waste classification images."""
    
    def __init__(self, data, transform=None):
        """
        Initialize dataset.
        
        Args:
            data: List of (image_path, label) tuples
            transform: Image transformation pipeline
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return black image if file cannot be loaded
            print(f"Warning: Could not load {image_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== Class Mapping ====================
WASTE_CLASS_MAPPING = {
    'aerosol_cans': 'hazardous',
    'aluminum_food_cans': 'non_biodegradable',
    'aluminum_soda_cans': 'non_biodegradable',
    'cardboard_boxes': 'biodegradable',
    'cardboard_packaging': 'biodegradable',
    'clothing': 'non_biodegradable',
    'coffee_grounds': 'biodegradable',
    'disposable_plastic_cutlery': 'non_biodegradable',
    'eggshells': 'biodegradable',
    'food_waste': 'biodegradable',
    'glass_beverage_bottles': 'non_biodegradable',
    'glass_cosmetic_containers': 'non_biodegradable',
    'glass_food_jars': 'non_biodegradable',
    'magazines': 'biodegradable',
    'newspaper': 'biodegradable',
    'office_paper': 'biodegradable',
    'paper_cups': 'biodegradable',
    'plastic_cup_lids': 'non_biodegradable',
    'plastic_detergent_bottles': 'non_biodegradable',
    'plastic_food_containers': 'non_biodegradable',
    'plastic_shopping_bags': 'non_biodegradable',
    'plastic_soda_bottles': 'non_biodegradable',
    'plastic_straws': 'non_biodegradable',
    'plastic_trash_bags': 'non_biodegradable',
    'plastic_water_bottles': 'non_biodegradable',
    'shoes': 'non_biodegradable',
    'steel_food_cans': 'non_biodegradable',
    'styrofoam_cups': 'non_biodegradable',
    'styrofoam_food_containers': 'non_biodegradable',
    'tea_bags': 'biodegradable'
}

MAIN_CLASSES = ['biodegradable', 'non_biodegradable', 'trash', 'hazardous']
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(MAIN_CLASSES)}


# ==================== Dataset Loading ====================
print("\n" + "="*70)
print("SmartBin Waste Classification - Model Training")
print("="*70)

print("\nLoading dataset...")
dataset_dir = os.path.join('..', 'dataset', 'images', 'images')

if not os.path.exists(dataset_dir):
    print(f"Error: Dataset not found at {dataset_dir}")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)

print(f"Dataset path: {dataset_dir}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with ImageFolder
full_dataset = ImageFolder(dataset_dir, transform=None)

# Map 32 categories to 4 main classes
def map_classes(image_folder_dataset, mapping, class_to_idx):
    """Map fine-grained categories to main waste classes."""
    mapped_dataset = []
    for path, label in image_folder_dataset.samples:
        original_class_name = image_folder_dataset.classes[label]
        new_class = mapping.get(original_class_name)
        if new_class:
            mapped_label = class_to_idx[new_class]
            mapped_dataset.append((path, mapped_label))
    return mapped_dataset

mapped_dataset = map_classes(full_dataset, WASTE_CLASS_MAPPING, CLASS_TO_IDX)
print(f"Loaded {len(mapped_dataset)} images across 4 classes")

# Split into train and test sets (80/20)
train_data, test_data = train_test_split(
    mapped_dataset, 
    test_size=0.2, 
    random_state=42, 
    stratify=[label for _, label in mapped_dataset]
)

print(f"Train set: {len(train_data)} | Test set: {len(test_data)}")

# Create PyTorch datasets and dataloaders
train_dataset = WasteDataset(train_data, transform=transform)
test_dataset = WasteDataset(test_data, transform=transform)

batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,
    pin_memory=False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0,
    pin_memory=False
)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")


# ==================== Model Setup ====================
print("\nInitializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load pre-trained VGG16
model = models.vgg16(pretrained=True)

# Freeze early layers (keep pre-trained weights)
for param in model.features[:-4].parameters():
    param.requires_grad = False

# Modify final classifier layer for 4 output classes
model.classifier[6] = nn.Linear(4096, 4)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.001
)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

print("Model ready for training")


# ==================== Training Function ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate model on test set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


# ==================== Training Loop ====================
print("\nStarting training...\n")
num_epochs = 40
best_acc = 0
best_model_state = None

start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    # Print progress every epoch
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

elapsed_time = time.time() - start_time

print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")
print(f"Best validation accuracy: {best_acc*100:.2f}%")


# ==================== Save Model ====================
print("\nSaving model...")
model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

# Load best model state
model.load_state_dict(best_model_state)

model_save_path = os.path.join(model_dir, '40.pth')
torch.save(model.state_dict(), model_save_path)

if os.path.exists(model_save_path):
    file_size = os.path.getsize(model_save_path) / (1024**2)
    print(f"Model saved to: {model_save_path}")
    print(f"File size: {file_size:.2f} MB")
else:
    print("Error: Model file was not saved")

print("\n" + "="*70)
print("Training Complete")
print("="*70)
print("\nNext: Run 'streamlit run app.py' to test the application")
print("="*70 + "\n")
