"""
Grad-CAM Visualization Demo
PyTorch-based Gradient Class Activation Mapping for waste classification model
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from main import load_model, preprocess_image

# Load the trained model
print("Loading model...")
model = load_model()
if model is None:
    print("Error: Could not load model")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def grad_cam(model, image_tensor, target_layer="features.28"):
    """
    Compute Grad-CAM heatmap for a given image
    """
    # Register hook to capture activations
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activations['features'] = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        gradients['features'] = grad_output[0].detach()
    
    # Get the target layer
    target_layer_module = dict(model.named_modules())[target_layer]
    target_layer_module.register_forward_hook(forward_hook)
    target_layer_module.register_full_backward_hook(backward_hook)
    
    # Forward pass
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    
    logits = model(image_tensor)
    predicted_class = logits.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    loss = logits[0, predicted_class[0]]
    loss.backward()
    
    # Compute Grad-CAM
    activations_val = activations['features'][0].cpu()
    gradients_val = gradients['features'][0].cpu()
    
    # Weight activations by gradients
    weights = gradients_val.mean(dim=(1, 2))
    cam = (weights.view(-1, 1, 1) * activations_val).sum(dim=0)
    cam = F.relu(cam)
    
    # Normalize
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    
    return cam.numpy(), predicted_class.item()

def visualize_gradcam(image_path):
    """
    Load image, compute Grad-CAM, and visualize
    """
    # Read and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img.resize((224, 224)))
    img_tensor = preprocess_image(img)
    
    # Compute Grad-CAM
    print(f"Computing Grad-CAM for {image_path}...")
    cam, pred_class = grad_cam(model, img_tensor, target_layer="features.28")
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Blend original image with heatmap
    overlay = cv2.addWeighted(img_bgr, 0.7, heatmap, 0.3, 0)
    
    # Class mapping
    idx_to_class = {0: 'Biodegradable', 1: 'Non-Biodegradable', 2: 'Trash', 3: 'Hazardous'}
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Overlay (Predicted: {idx_to_class[pred_class]})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_result.png', dpi=100, bbox_inches='tight')
    print(f"✅ Grad-CAM visualization saved as 'gradcam_result.png'")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your test image path
    test_image_path = "test.jpg"
    try:
        visualize_gradcam(test_image_path)
    except FileNotFoundError:
        print(f"Error: Test image not found at {test_image_path}")
        print("Please provide a valid image path")