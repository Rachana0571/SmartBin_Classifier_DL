import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from main import load_model, predict_image_class

st.set_page_config(page_title="♻️ Waste Classification", page_icon="♻️", layout="wide")

# Cache the CNN model
@st.cache_resource
def get_model():
    return load_model()

# Cache YOLO model
@st.cache_resource
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("./model/yolov8n.pt")   # or your trained model path e.g. "model/best.pt"

# ✅ Load once
model = get_model()
if model is None:
    st.error("Model not loaded. Please check your model file.")
    st.stop()

yolo_model = get_yolo_model()

st.title("♻️ Waste Classification CNN Convolutional Neural Network")

st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select classification mode:", [
    "Upload/Capture Image",
    "Real-Time Webcam",
    "Object Detection",
    "Real-Time Detection & Classification"
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

# ------------------- Real-Time Webcam -------------------
elif mode == "Real-Time Webcam":
    st.write("## Real-Time Waste Detection and Classification")

    start = st.button("▶️ Start Camera")
    stop = st.button("⏹️ Stop Camera")

    stframe = st.empty()
    class_placeholder = st.empty()

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    if start:
        st.session_state.run_webcam = True
    if stop:
        st.session_state.run_webcam = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open camera")
    else:
        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Run detection
            results = yolo_model(pil_image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names

            for box, cls in zip(boxes, classes):
                x_min, y_min, x_max, y_max = box.astype(int)
                label = class_names[cls]
                # Crop detected object for CNN classification
                cropped_obj = pil_image.crop((x_min, y_min, x_max, y_max))
                obj_class = predict_image_class(cropped_obj, model)
                combined_label = f"{label} ({obj_class})"
                color = (0, 255, 0) if obj_class == "Biodegradable" else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, combined_label, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Classification
            predicted_class = predict_image_class(pil_image, model)
            cv2.putText(frame, f"Class: {predicted_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            class_placeholder.success(f"Detected Class: {predicted_class}")

            time.sleep(0.05)

        cap.release()
# ------------------- Object Detection -------------------
elif mode == "Object Detection":
    st.write("## Object Detection")
    uploaded_image = st.file_uploader("📤 Upload an image for detection", type=['jpg', 'jpeg', 'png'])
    camera_image = st.camera_input("📷 Take a photo for detection")

    def run_object_detection(image):
        try:
            results = yolo_model(image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            img_np = np.array(image)
            for box, cls in zip(boxes, classes):
                x_min, y_min, x_max, y_max = box.astype(int)
                label = class_names[cls]
                color = (0,255,0) if label == "Biodegradable" else (0,0,255)
                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img_np, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return img_np
        except Exception as e:
            st.error(f"Detection error: {e}")
            return None

    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
        detected_img = run_object_detection(image)
        if detected_img is not None:
            st.image(detected_img, caption="Detected Objects", use_container_width=True)

    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        st.image(image, caption="Captured Image", use_container_width=True)
        detected_img = run_object_detection(image)
        if detected_img is not None:
            st.image(detected_img, caption="Detected Objects", use_container_width=True)

# ------------------- Real-Time Detection & Classification -------------------
elif mode == "Real-Time Detection & Classification":
    st.write("""
    ## Real-Time Detection & Classification
    This mode runs both object detection and overall waste classification on each frame/image.
    """)
    uploaded_image = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])
    camera_image = st.camera_input("📷 Take a photo")
    run_webcam = st.button("Start Webcam Detection & Classification")
    stop_webcam = st.button("Stop Webcam")
    stframe = st.empty()

    def run_object_detection_and_classification(image):
        detected_img = None
        labels = []
        try:
            results = yolo_model(image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            img_np = np.array(image)
            for box, cls in zip(boxes, classes):
                x_min, y_min, x_max, y_max = box.astype(int)
                label = class_names[cls]
                # Crop detected object for CNN classification
                cropped_obj = image.crop((x_min, y_min, x_max, y_max))
                obj_class = predict_image_class(cropped_obj, model)
                combined_label = f"{label} ({obj_class})"
                labels.append(combined_label)
                color = (0,255,0) if obj_class == "Biodegradable" else (0,0,255)
                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img_np, combined_label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            detected_img = img_np
        except Exception as e:
            st.error(f"Detection error: {e}")

        # Classification
        try:
            predicted_class = predict_image_class(image, model)
        except Exception as e:
            predicted_class = None
            st.error(f"Classification error: {e}")
        return detected_img, predicted_class, labels

    # Upload or capture
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
        detected_img, predicted_class, labels = run_object_detection_and_classification(image)
        if detected_img is not None:
            st.image(detected_img, caption="Detected Objects & Classification", use_container_width=True)
        if predicted_class:
            st.success(f"✅ **Overall Predicted Class:** *{predicted_class}*")
        if labels:
            st.info(f"Detected objects: {', '.join(labels)}")

    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        st.image(image, caption="Captured Image", use_container_width=True)
        detected_img, predicted_class, labels = run_object_detection_and_classification(image)
        if detected_img is not None:
            st.image(detected_img, caption="Detected Objects & Classification", use_container_width=True)
        if predicted_class:
            st.success(f"✅ **Overall Predicted Class:** *{predicted_class}*")
        if labels:
            st.info(f"Detected objects: {', '.join(labels)}")

    # Webcam loop
    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                detected_img, predicted_class, labels = run_object_detection_and_classification(pil_image)
                if detected_img is not None:
                    stframe.image(detected_img, channels="RGB")
                if predicted_class:
                    stframe.success(f"✅ **Overall Predicted Class:** *{predicted_class}*")
                if labels:
                    stframe.info(f"Detected objects: {', '.join(labels)}")
                time.sleep(0.05)
                if stop_webcam:
                    break
            cap.release()

# ------------------- Introduction -------------------
st.header("Introduction")
st.write("""
This project utilizes a Convolutional Neural Network (CNN) to classify waste into four categories: 
**Biodegradable, Non-Biodegradable, Trash, or Hazardous**. 
The model is designed to assist in waste management by automating the classification process.
""")