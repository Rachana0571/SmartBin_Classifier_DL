import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the mapping of waste categories to the 4 main classes
waste_class_mapping = {
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

main_classes = ['biodegradable', 'non_biodegradable', 'trash', 'hazardous']
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(main_classes)}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class
class WasteDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# Load dataset
print("Loading dataset...")
dataset_dir = r'./dataset'

if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory not found at {dataset_dir}")
    print("Make sure the dataset is in the correct location.")
    exit(1)

# Collect all image paths and labels
mapped_dataset = []
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    new_class = waste_class_mapping.get(category, None)
    if not new_class:
        print(f"Warning: No mapping found for {category}, skipping...")
        continue
    
    mapped_label = class_to_idx[new_class]
    
    # Load from both 'default' and 'real_world' folders
    for subfolder in ['default', 'real_world']:
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            for img_file in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_file)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mapped_dataset.append((img_path, mapped_label))

print(f"Total images found: {len(mapped_dataset)}")

if len(mapped_dataset) == 0:
    print("Error: No images found in dataset directory!")
    exit(1)

# Split dataset
train_data, test_data = train_test_split(
    mapped_dataset, test_size=0.2, random_state=42, 
    stratify=[label for _, label in mapped_dataset]
)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Create dataloaders
batch_size = 32
train_dataset = WasteDataset(train_data, transform=transform)
test_dataset = WasteDataset(test_data, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Load VGG16 model
print("Loading VGG16 model...")
model = models.vgg16(pretrained=True)

# Freeze early layers
for param in model.features[:-4].parameters():
    param.requires_grad = False

# Modify classifier
model.classifier[6] = nn.Linear(4096, 4)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Training loop
num_epochs = 40
best_loss = float('inf')

print("\nStarting training...")
for epoch in range(num_epochs):
    # Train
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

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(test_loader.dataset)
    val_acc = val_correct / val_total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), './model/40.pth')
        print(f"✓ Model saved!")

print("\n✓ Training complete! Model saved to ./model/40.pth")
