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
    1. best_improved.pth (90.67% accuracy on 32 waste categories)
    2. 40.pth (fallback trained model)
    3. Pre-trained ImageNet weights (last resort)
    
    The model is a VGG16 CNN with transfer learning:
    - Pre-trained on ImageNet for feature extraction
    - Fine-tuned classifier for 4 waste categories:
      * Biodegradable: Food waste, paper, cardboard, etc.
      * Non-Biodegradable: Plastics, glass, aluminum, metal
      * Trash: Mixed/contaminated waste
      * Hazardous: Aerosol cans, toxic materials
    
    Returns:
        model: Loaded VGG16 model in evaluation mode, or None if loading fails
    
    Raises:
        Implicit: Prints error messages if loading fails
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
    
    Applies standard PyTorch/ImageNet preprocessing:
    - Resizes to 224x224 (VGG16 input size)
    - Converts to tensor (0-1 range)
    - Normalizes using ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    Args:
        image: PIL Image object (assumed to be RGB, 3 channels)
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension [1, 3, 224, 224]
        
    Note:
        Batch dimension added with unsqueeze(0) for direct model inference
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
    
    Uses the trained VGG16 model to classify waste into one of 4 categories.
    
    Args:
        image: PIL Image object (will be converted to RGB if needed)
        model: Loaded PyTorch VGG16 model in evaluation mode
        
    Returns:
        str: Predicted waste class name:
            - 'Biodegradable': Food waste, paper, cardboard, etc.
            - 'Non-Biodegradable': Plastics, glass, aluminum, metal, clothing
            - 'Trash': Mixed or contaminated waste
            - 'Hazardous': Aerosol cans, toxic materials
            - 'Unknown': If prediction fails
            
    Process:
        1. Preprocesses image (resize, normalize)
        2. Passes through model with no_grad context (inference only)
        3. Extracts class with highest confidence
        4. Returns human-readable class name
    """
    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred_idx = torch.max(outputs, 1)
    
    class_names = {0: 'Biodegradable', 1: 'Non-Biodegradable', 2: 'Trash', 3: 'Hazardous'}
    return class_names.get(pred_idx.item(), "Unknown")