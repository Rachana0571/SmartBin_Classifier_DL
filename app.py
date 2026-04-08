import streamlit as st
from PIL import Image
import numpy as np
import time
from main import load_model, predict_image_class

st.set_page_config(page_title="♻️ Waste Classification", page_icon="♻️", layout="wide")

# Cache the CNN model
@st.cache_resource
def get_model():
    return load_model()

# ✅ Load once
model = get_model()
if model is None:
    st.error("Model not loaded. Please check your model file.")
    st.stop()

st.title("♻️ Waste Classification CNN Convolutional Neural Network")

st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select classification mode:", [
    "Upload/Capture Image",
])

# ------------------- Upload/Capture Image -------------------
if mode == "Upload/Capture Image":
    st.write("Upload or capture an image of waste, and the model will classify it into one of four categories.")
    uploaded_image = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])
    camera_image = st.camera_input("📷 Take a photo")

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            predicted_class = predict_image_class(image, model)
            st.balloons()
            st.success(f"✅ **Predicted Class:** *{predicted_class}*")
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert('RGB')
            st.image(image, caption="Captured Image", use_container_width=True)
            predicted_class = predict_image_class(image, model)
            st.balloons()
            st.success(f"✅ **Predicted Class:** *{predicted_class}*")
        except Exception as e:
            st.error(f"❌ Error processing the camera image: {e}")

# ------------------- Introduction -------------------
st.header("Introduction")
st.write("""
This project utilizes a Convolutional Neural Network (CNN) to classify waste into four categories: 
**Biodegradable, Non-Biodegradable, Trash, or Hazardous**. 
The model is designed to assist in waste management by automating the classification process.
""")