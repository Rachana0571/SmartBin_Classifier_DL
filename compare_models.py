"""
Test BOTH model files to find which one has good accuracy
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os
import random
from sklearn.metrics import accuracy_score

# Configuration
DATASET_PATH = '../dataset/images/images'
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
        
        for subfolder in ['default']:
            subfolder_path = cat_path / subfolder
            if subfolder_path.exists():
                for img_file in list(subfolder_path.glob('*.jpg')) + list(subfolder_path.glob('*.png')):
                    image_paths.append(str(img_file))
                    labels.append(label_idx)
    
    if len(image_paths) > SAMPLE_SIZE:
        sample_indices = random.sample(range(len(image_paths)), SAMPLE_SIZE)
        image_paths = [image_paths[i] for i in sample_indices]
        labels = [labels[i] for i in sample_indices]
    
    return image_paths, labels

def test_model(model_path, model_name):
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}\n")
    
    device = torch.device('cpu')
    
    # Load model
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 4)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from: {model_path}\n")
    except Exception as e:
        print(f"❌ Failed to load: {e}\n")
        return 0
    
    model = model.to(device)
    model.eval()
    
    # Load test images
    image_paths, labels = load_test_images()
    print(f"Testing on {len(image_paths)} images...\n")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images")
            
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()
                
                all_preds.append(pred)
                all_labels.append(labels[idx])
            except:
                pass
    
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    
    print(f"\n✅ ACCURACY: {accuracy*100:.2f}%\n")
    
    return accuracy

# Test both models
print("\n" + "="*70)
print("COMPARING MODEL FILES")
print("="*70)

models_to_test = [
    ('./model/40.pth', '40.pth (Current)'),
    ('./model/best_improved.pth', 'best_improved.pth (Previous)'),
]

results = {}
for model_path, name in models_to_test:
    if os.path.exists(model_path):
        acc = test_model(model_path, name)
        results[name] = acc
    else:
        print(f"❌ {name} NOT FOUND")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    status = "🟢 GOOD" if acc >= 0.70 else "🟡 OK" if acc >= 0.60 else "🔴 BAD"
    print(f"{name:40} → {acc*100:6.2f}% {status}")

# Find best model
if results:
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n✅ BEST MODEL: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]*100:.2f}%")
    
    if best_model[1] >= 0.70:
        print(f"\n✅ This model is PERFECT for your review!")
        print(f"   Your faculty will be satisfied with {best_model[1]*100:.1f}% accuracy")
    else:
        print(f"\n⚠️  Need to improve accuracy")

print("\n" + "="*70 + "\n")
