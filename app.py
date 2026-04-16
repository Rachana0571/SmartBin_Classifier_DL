"""
SmartBin Waste Classification - Streamlit Web Application
Web interface for waste classification using trained VGG16 CNN model
Supports image upload and webcam capture
"""

import streamlit as st
from PIL import Image
from main import load_model, predict_image_class


st.set_page_config(
    page_title="♻️ Waste Classification",
    page_icon="♻️",
    layout="wide"
)


@st.cache_resource
def get_model():
    """Load and cache the trained model once."""
    return load_model()


# Initialize model
model = get_model()
if model is None:
    st.error("Error: Could not load model. Please verify model files exist.")
    st.stop()


st.title("♻️ Waste Classification")
st.write("""
**Automated waste classification using Deep Learning**

Upload or capture an image of waste to classify it into one of four categories:
**Biodegradable**, **Non-Biodegradable**, **Trash**, or **Hazardous**
""")

col1, col2 = st.columns(2)

# Image upload section
with col1:
    st.subheader("📤 Upload Image")
    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            predicted_class = predict_image_class(image, model)
            st.success(f"**Predicted Class:** {predicted_class}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Camera capture section
with col2:
    st.subheader("📷 Capture Photo")
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert('RGB')
            st.image(image, caption="Captured Image", use_container_width=True)
            
            predicted_class = predict_image_class(image, model)
            st.success(f"**Predicted Class:** {predicted_class}")
        except Exception as e:
            st.error(f"Error processing camera image: {e}")


# Information section
st.divider()
st.header("About This Project")
st.write("""
**Objective:** Automate waste classification to improve recycling and waste management efficiency.

**Technology:** 
- Convolutional Neural Network (CNN) using VGG16 architecture
- Transfer learning with ImageNet pre-trained weights
- 90.67% classification accuracy

**Classes:**
- **Biodegradable:** Organic waste (food, paper, cardboard, etc.)
- **Non-Biodegradable:** Plastics, metals, glass
- **Trash:** Mixed/unusable materials
- **Hazardous:** Dangerous materials (chemicals, batteries, etc.)

**Dataset:** 15,000+ waste images across 32 categories, mapped to 4 main classes
""")