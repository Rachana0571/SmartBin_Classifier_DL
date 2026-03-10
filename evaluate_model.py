"""
Result & Performance Evaluation Module
Evaluates the trained VGG16 model on the test dataset and generates performance metrics
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class mapping
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
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

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

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n" + "="*70)
print("LOADING DATASET FOR EVALUATION")
print("="*70)

dataset_dir = r'C:\Users\User\Downloads\dataset\images\images'

if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory not found at {dataset_dir}")
    exit(1)

# Collect all image paths and labels
mapped_dataset = []
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    new_class = waste_class_mapping.get(category, None)
    if not new_class:
        continue
    
    mapped_label = class_to_idx[new_class]
    
    for subfolder in ['default', 'real_world']:
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            for img_file in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_file)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mapped_dataset.append((img_path, mapped_label))

print(f"✓ Total images found: {len(mapped_dataset)}")

# Split dataset (same as training)
train_data, test_data = train_test_split(
    mapped_dataset, test_size=0.2, random_state=42,
    stratify=[label for _, label in mapped_dataset]
)

print(f"✓ Train samples: {len(train_data)}")
print(f"✓ Test samples: {len(test_data)}")

# Create test dataloader
batch_size = 32
test_dataset = WasteDataset(test_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

model_path = './model/40.pth'
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 4)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from {model_path}")
except FileNotFoundError:
    print(f"✗ Model file not found at {model_path}")
    exit(1)

model.to(device)
model.eval()

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

# Overall Accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✓ Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# Precision, Recall, F1-Score (per class)
precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\nMicro-Average Metrics:")
print(f"  Precision: {precision_micro:.4f}")
print(f"  Recall:    {recall_micro:.4f}")
print(f"  F1-Score:  {f1_micro:.4f}")

print(f"\nWeighted-Average Metrics:")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall:    {recall_weighted:.4f}")
print(f"  F1-Score:  {f1_weighted:.4f}")

# Per-class metrics
print(f"\n" + "-"*70)
print("PER-CLASS METRICS")
print("-"*70)

precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

for i, class_name in enumerate(main_classes):
    print(f"\n{class_name.upper()}:")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print(f"  Recall:    {recall_per_class[i]:.4f}")
    print(f"  F1-Score:  {f1_per_class[i]:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print(f"\n" + "-"*70)
print("CONFUSION MATRIX")
print("-"*70)
print("\nPredicted →")
print("Actual ↓     " + "  ".join([f"{cls:15}" for cls in main_classes]))
for i, class_name in enumerate(main_classes):
    print(f"{class_name:12} " + "  ".join([f"{cm[i][j]:15}" for j in range(len(main_classes))]))

# Classification Report
print(f"\n" + "-"*70)
print("DETAILED CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(all_labels, all_preds, target_names=main_classes, zero_division=0))

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Create output directory for plots
os.makedirs('./evaluation_results', exist_ok=True)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=main_classes, yticklabels=main_classes,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Waste Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('./evaluation_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 2. Per-Class Metrics Comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(main_classes))
width = 0.25

bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
bars2 = ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Waste Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(main_classes, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('./evaluation_results/per_class_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: per_class_metrics.png")
plt.close()

# 3. Accuracy Distribution
class_counts = np.bincount(all_labels, minlength=len(main_classes))
correct_per_class = []
for i in range(len(main_classes)):
    if class_counts[i] > 0:
        class_accuracy = np.sum((all_labels == i) & (all_preds == i)) / class_counts[i]
    else:
        class_accuracy = 0
    correct_per_class.append(class_accuracy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
ax1.bar(main_classes, class_counts, color='skyblue', alpha=0.8, edgecolor='navy')
ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax1.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=15)
for i, v in enumerate(class_counts):
    ax1.text(i, v, str(int(v)), ha='center', va='bottom', fontweight='bold')

# Per-class accuracy
colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in correct_per_class]
ax2.bar(main_classes, correct_per_class, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.tick_params(axis='x', rotation=15)
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Acceptable (60%)')
ax2.legend()
for i, v in enumerate(correct_per_class):
    ax2.text(i, v, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('./evaluation_results/accuracy_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: accuracy_analysis.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("EVALUATION SUMMARY REPORT")
print("="*70)

summary_text = f"""
WASTE CLASSIFICATION MODEL - PERFORMANCE EVALUATION REPORT
{'='*70}

1. DATASET INFORMATION
   - Total Test Samples: {len(test_data)}
   - Number of Classes: {len(main_classes)}
   - Classes: {', '.join(main_classes)}

2. OVERALL PERFORMANCE
   - Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)
   - Precision (Weighted): {precision_weighted:.4f}
   - Recall (Weighted): {recall_weighted:.4f}
   - F1-Score (Weighted): {f1_weighted:.4f}

3. PER-CLASS PERFORMANCE
"""

for i, class_name in enumerate(main_classes):
    summary_text += f"""
   {class_name.upper()}:
     - Precision: {precision_per_class[i]:.4f}
     - Recall:    {recall_per_class[i]:.4f}
     - F1-Score:  {f1_per_class[i]:.4f}
     - Test Samples: {class_counts[i]}
"""

summary_text += f"""

4. KEY FINDINGS
   - Best performing class: {main_classes[np.argmax(f1_per_class)]} (F1: {np.max(f1_per_class):.4f})
   - Worst performing class: {main_classes[np.argmin(f1_per_class)]} (F1: {np.min(f1_per_class):.4f})
   - Model Status: {'✓ Production Ready' if overall_accuracy >= 0.8 else '⚠ Needs Improvement'}

5. OUTPUT FILES GENERATED
   - Confusion Matrix: ./evaluation_results/confusion_matrix.png
   - Per-Class Metrics: ./evaluation_results/per_class_metrics.png
   - Accuracy Analysis: ./evaluation_results/accuracy_analysis.png
   - Summary Report: ./evaluation_results/evaluation_summary.txt

{'='*70}
"""

print(summary_text)

# Save summary report
with open('./evaluation_results/evaluation_summary.txt', 'w') as f:
    f.write(summary_text)
print("\n✓ Saved: evaluation_summary.txt")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\nAll results saved to: ./evaluation_results/")
