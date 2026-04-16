"""
SmartBin Waste Classification - Model Loading and Prediction
Loads pre-trained VGG16 model and provides image prediction functionality
"""

import torch
from torchvision import models, transforms
from PIL import Image
import os


def load_model():
    """
    Load trained VGG16 model for waste classification.
    
    Attempts to load models in priority order:
    1. best_improved.pth (90.67% accuracy)
    2. 40.pth (fallback trained model)
    3. Pre-trained ImageNet weights (last resort)
    
    Returns:
        model: Loaded VGG16 model in evaluation mode, or None if loading fails
    """
    model_path_best = './model/best_improved.pth'
    model_path_fallback = './model/40.pth'
    device = torch.device('cpu')
    
    # Initialize VGG16 model with 4 output classes
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 4)
    
    # Try loading best model first
    try:
        model.load_state_dict(torch.load(model_path_best, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        pass
    
    # Try loading fallback model
    try:
        model.load_state_dict(torch.load(model_path_fallback, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        # Use pre-trained ImageNet weights as last resort
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model.classifier[6] = torch.nn.Linear(4096, 4)
            os.makedirs('./model', exist_ok=True)
            model.eval()
            return model
        except Exception as e:
            print(f"Error creating model: {e}")
            return None


def preprocess_image(image):
    """
    Preprocess PIL image for model inference.
    
    Args:
        image: PIL Image object (assumed to be RGB)
        
    Returns:
        Tensor: Preprocessed image tensor with batch dimension [1, 3, 224, 224]
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_image_class(image, model):
    """
    Predict waste classification for a given image.
    
    Args:
        image: PIL Image object
        model: Loaded PyTorch model
        
    Returns:
        str: Predicted class name (Biodegradable, Non-Biodegradable, Trash, or Hazardous)
    """
    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred_idx = torch.max(outputs, 1)
    
    class_names = {0: 'Biodegradable', 1: 'Non-Biodegradable', 2: 'Trash', 3: 'Hazardous'}
    return class_names.get(pred_idx.item(), "Unknown")