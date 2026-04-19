"""
Model Comparison Script
Compares accuracy of available trained models
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os
import random


# Configuration
# Handle flexible dataset path - works in both single and multi-root workspaces
possible_paths = [
    '../complete_dataset',  # PRIMARY
    '../dataset/images/images',
    './complete_dataset',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'complete_dataset')),
]

DATASET_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    print("Error: Dataset not found. Tried paths:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    print("\nPlease download the dataset from:")
    print("https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification")
    exit(1)

SAMPLE_SIZE = 300

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
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(MAIN_CLASSES)}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_test_images():
    """Load random sample of test images from dataset."""
    image_paths = []
    labels = []
    
    dataset_dir = Path(DATASET_PATH)
    
    for category in sorted(os.listdir(dataset_dir)):
        cat_path = dataset_dir / category
        if not cat_path.is_dir():
            continue
        
        main_class = WASTE_CLASS_MAPPING.get(category)
        if not main_class:
            continue
        
        label_idx = CLASS_TO_IDX[main_class]
        
        subfolder_path = cat_path / 'default'
        if subfolder_path.exists():
            for img_file in list(subfolder_path.glob('*.jpg')) + list(subfolder_path.glob('*.png')):
                image_paths.append(str(img_file))
                labels.append(label_idx)
    
    # Sample random subset  
    if len(image_paths) > SAMPLE_SIZE:
        sample_indices = random.sample(range(len(image_paths)), SAMPLE_SIZE)
        image_paths = [image_paths[i] for i in sample_indices]
        labels = [labels[i] for i in sample_indices]
    
    return image_paths, labels


def evaluate_model(model_path, model_name, device):
    """Evaluate single model on test set."""
    print(f"\nEvaluating: {model_name}")
    print("-" * 70)
    
    # Load model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 4)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Load test images
    image_paths, labels = load_test_images()
    print(f"Testing on {len(image_paths)} images")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)}")
            
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()
                
                all_preds.append(pred)
                all_labels.append(labels[idx])
            except:
                pass
    
    # Calculate accuracy without sklearn dependency
    if all_labels:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)
    else:
        accuracy = 0
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    return accuracy


def main():
    """Compare all available models."""
    print("\n" + "="*70)
    print("SmartBin - Model Comparison")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    models_to_test = [
        ('./model/best_improved.pth', 'best_improved.pth (90.67% expected)'),
        ('./model/40.pth', '40.pth (fallback model)'),
    ]
    
    results = {}
    for model_path, name in models_to_test:
        if os.path.exists(model_path):
            acc = evaluate_model(model_path, name, device)
            if acc is not None:
                results[name] = acc
        else:
            print(f"Skipped: {name} - file not found")
    
    # Display summary
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:50} {acc*100:7.2f}%")
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1])
        print("\n" + "-"*70)
        print(f"Best Model: {best_model[0]}")
        print(f"Accuracy: {best_model[1]*100:.2f}%")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
