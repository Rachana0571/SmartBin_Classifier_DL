import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load model function
def load_model():
    model_path = './model/40.pth'
    
    # Create model with 4 output classes
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 4)  # 4 output classes
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"⚠️  Model file not found at {model_path}")
        print("💡 Creating base model with pretrained VGG16 weights...")
        print("⏳ Downloading weights (this may take a moment on first run)...")
        
        # Create base model with pretrained weights
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model.classifier[6] = torch.nn.Linear(4096, 4)
            
            # Create model directory if needed
            os.makedirs('./model', exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved to {model_path}")
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            return None

# Image preprocessing function
def preprocess_image(image):
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