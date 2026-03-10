import streamlit as st
import cv2
from PIL import Image
import numpy as np
from main import load_model, predict_image_class
import time

st.title("Real-Time Waste Classifier")

# Load model
model = load_model()
if model is None:
    st.error("Model not loaded. Please check your model file.")
    st.stop()

# OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot open camera")
    st.stop()

stframe = st.empty()  # placeholder for video frames

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Convert BGR → RGB → PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Prediction
    predicted_class = predict_image_class(pil_image, model)

    # Add text overlay
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show in Streamlit (convert BGR→RGB for display)
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Add slight delay (Streamlit loop refresh)
    time.sleep(0.05)