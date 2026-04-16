"""
Quick Sanity Check
Tests model loading and inference on sample images
"""

from main import load_model, predict_image_class
from PIL import Image
import random
from pathlib import Path
import os


def quick_test():
    """Run quick test on 5 sample images."""
    print("\n" + "="*70)
    print("SmartBin - Quick Sanity Check")
    print("="*70 + "\n")

    # Load model
    print("Loading model...")
    model = load_model()

    if model is None:
        print("Error: Failed to load model")
        exit(1)

    print("Model loaded successfully\n")

    # Define waste categories
    WASTE_CLASS_MAPPING = {
        'aerosol_cans': 'Hazardous',
        'aluminum_food_cans': 'Non-Biodegradable',
        'cardboard_boxes': 'Biodegradable',
        'glass_beverage_bottles': 'Non-Biodegradable',
        'food_waste': 'Biodegradable',
    }

    # Find test images
    dataset_path = Path('../dataset/images/images')
    test_images = []

    for category, expected_class in WASTE_CLASS_MAPPING.items():
        cat_path = dataset_path / category / 'default'
        if cat_path.exists():
            images = list(cat_path.glob('*.jpg')) + list(cat_path.glob('*.png'))
            if images:
                test_images.append((str(images[0]), category, expected_class))

    if not test_images:
        print("Error: Could not find test images")
        return

    # Run predictions
    print("Testing predictions:\n")
    print(f"{'Category':<25} {'Expected':<20} {'Predicted':<20} {'Result':<10}")
    print("-" * 75)

    correct = 0
    total = len(test_images)

    for img_path, category, expected in test_images:
        try:
            image = Image.open(img_path).convert('RGB')
            prediction = predict_image_class(image, model)
            
            # Check if prediction matches expected class
            is_correct = expected.lower() in prediction.lower()
            result = "PASS" if is_correct else "FAIL"
            
            print(f"{category:<25} {expected:<20} {prediction:<20} {result:<10}")
            
            if is_correct:
                correct += 1
        except Exception as e:
            print(f"Error testing {category}: {e}")

    print("-" * 75)
    print(f"\nResults: {correct}/{total} correct ({(correct/total)*100:.0f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    quick_test()
