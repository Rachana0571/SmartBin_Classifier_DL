"""
Fast accuracy evaluation on random sample
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configuration
DATASET_PATH = '../dataset/images/images'
SAMPLE_SIZE = 500  # Test on 500 images for speed
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
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_fast():
    print("\n" + "="*80)
    print("WASTE CLASSIFICATION - QUICK ACCURACY TEST")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Test Sample Size: {SAMPLE_SIZE} images\n")
    
    # Load dataset
    print("Gathering images from dataset...")
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
        
        for subfolder in ['default']:  # Use only default for speed
            subfolder_path = cat_path / subfolder
            if subfolder_path.exists():
                for img_file in list(subfolder_path.glob('*.jpg')) + list(subfolder_path.glob('*.png')):
                    image_paths.append(str(img_file))
                    labels.append((label_idx, category))
    
    # Sample random images
    if len(image_paths) > SAMPLE_SIZE:
        sample_indices = random.sample(range(len(image_paths)), SAMPLE_SIZE)
        image_paths = [image_paths[i] for i in sample_indices]
        labels = [labels[i] for i in sample_indices]
    
    print(f"✅ Selected {len(image_paths)} test images\n")
    
    # Load model
    print("Loading model...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 4)
    
    try:
        state_dict = torch.load('./model/40.pth', map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded fine-tuned model\n")
    except:
        print("⚠️  Using ImageNet pre-trained weights\n")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("Evaluating predictions...")
    all_preds = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images")
            
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()
                label, category = labels[idx]
                
                all_preds.append(pred)
                all_labels.append(label)
                all_categories.append(category)
            except:
                pass
    
    print()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nAccuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    # Per-class results
    print("\n" + "-"*80)
    print("PER-CLASS METRICS:")
    print("-"*80)
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=MAIN_CLASSES,
                                   output_dict=True, zero_division=0)
    
    for cls in MAIN_CLASSES:
        if cls in report:
            cls_report = report[cls]
            print(f"\n{cls.upper()}:")
            print(f"  Precision: {cls_report['precision']*100:.1f}%")
            print(f"  Recall:    {cls_report['recall']*100:.1f}%")
            print(f"  F1-Score:  {cls_report['f1-score']*100:.1f}%")
    
    print("\n" + "="*80)
    if accuracy >= 0.72:
        print("✅ MODEL ACCURACY IS GOOD - Ready for project review!")
    elif accuracy >= 0.60:
        print("✅ MODEL ACCURACY IS ACCEPTABLE - Suitable for review")
    else:
        print("⚠️  MODEL ACCURACY NEEDS IMPROVEMENT")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate_fast()
