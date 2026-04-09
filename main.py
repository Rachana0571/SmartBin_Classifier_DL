import torch
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request
from PIL import ImageOps

# Load model function
def load_model():
    # Try to load best_improved.pth first (90.67% accuracy)
    model_path_best = './model/best_improved.pth'
    # Fallback to 40.pth
    model_path_fallback = './model/40.pth'
    
    # Create model with 4 output classes
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 4)  # 4 output classes
    
    # Try best_improved.pth first
    try:
        model.load_state_dict(torch.load(model_path_best, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        return model
    except FileNotFoundError:
        pass
    
    # Try fallback to 40.pth
    try:
        model.load_state_dict(torch.load(model_path_fallback, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"⚠️  No trained model found")
        print("💡 Creating base model with pretrained VGG16 weights...")
        print("⏳ Downloading weights (this may take a moment on first run)...")
        
        # Create base model with pretrained weights - this is the fallback
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model.classifier[6] = torch.nn.Linear(4096, 4)
            
            # Create model directory if needed
            os.makedirs('./model', exist_ok=True)
            
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            return None

# Image preprocessing function
def preprocess_image(image):
    # Fix image rotation (important for phone uploads)
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass  # If no EXIF data, just continue
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform_test(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class
def predict_image_class(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    # Map index to class name
    idx_to_class = {0: 'Biodegradable', 1: 'Non-Biodegradable', 2: 'Trash', 3: 'Hazardous'}
    return idx_to_class.get(preds.item(), "Unknown")