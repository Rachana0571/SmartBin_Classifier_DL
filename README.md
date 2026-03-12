# ♻️ SmartBin Classifier - Intelligent Waste Classification System

A deep learning-powered application that automatically classifies waste into four categories: **Biodegradable**, **Non-Biodegradable**, **Trash**, or **Hazardous**. This project leverages a Convolutional Neural Network (CNN) built on the VGG16 architecture to assist in intelligent waste management and recycling.

---

## 📋 Project Overview

This application provides an end-to-end solution for waste classification:
- **32 waste categories** mapped to **4 main classes** for practical waste management
- **VGG16-based CNN** with transfer learning for high accuracy
- **Multiple interfaces**: Web app (Streamlit), command-line, real-time webcam processing
- **Cross-platform compatibility** with proper path handling
- **Visualization tools** including Grad-CAM for model interpretability

### Key Classes
- 🟢 **Biodegradable**: Food waste, paper, cardboard, tea bags, eggshells, coffee grounds
- 🔵 **Non-Biodegradable**: Plastics (bottles, bags, containers), glass, aluminum, steel, clothing, shoes
- 🟡 **Trash**: Mixed or contaminated waste
- 🔴 **Hazardous**: Aerosol cans, toxic materials

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Image Upload** | Upload images for instant classification |
| **Webcam Support** | Real-time waste classification from live video feed |
| **Object Detection** | YOLO-based localization of waste items |
| **Model Interpretability** | Grad-CAM visualization showing decision areas |
| **High Accuracy** | Transfer learning from ImageNet-pretrained weights |
| **User-Friendly UI** | Streamlit web interface with intuitive controls |

---

## 📊 Dataset

**Source**: [Kaggle - Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

- **32 waste categories** with dual imaging modes:
  - `default/`: Clean, controlled environment images
  - `real_world/`: Photos in natural conditions
- **Total Classes**: Mapped to 4 primary categories
- **Train-Test Split**: 80% training, 20% testing

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB+ free disk space (for model & dependencies)

### Step 1: Clone or Download Repository
```bash
cd SmartBin_Classifier_DL
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
The dataset folder should be placed alongside this project:
```
parent_folder/
├── SmartBin_Classifier_DL/     # This project
├── dataset/                     # Download from Kaggle
│   └── images/
│       └── images/
│           ├── aerosol_cans/
│           ├── aluminum_food_cans/
│           └── ... (30 more categories)
```

**Download dataset from Kaggle:**
```bash
kaggle datasets download -d alistairking/recyclable-and-household-waste-classification
unzip recyclable-and-household-waste-classification.zip
```

---

## 🚀 Usage

### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

**Features in Web App:**
- 📤 Upload/Capture Image
- 📷 Real-Time Webcam Classification
- 🎯 Object Detection Mode
- 🔍 Real-Time Detection & Classification

### Option 2: Command-Line Prediction
```bash
python realtime_classify.py
```

### Option 3: Train Your Own Model
Open and run the Jupyter notebook at your own pace:
```
waste classifier.ipynb
```

---

## 📁 Project Structure

```
SmartBin_Classifier_DL/
├── app.py                    # Streamlit web application
├── main.py                   # Core functions (load_model, predict_image_class)
├── train_model.py            # Model training script
├── evaluate_model.py         # Model evaluation & metrics
├── generate_model.py         # Generate base model
├── realtime_classify.py      # Real-time classification interface
├── demo.py                   # Grad-CAM visualization demo
├── waste classifier.ipynb    # Jupyter notebook for training
├── requirements.txt          # Python dependencies
├── model/                    # Directory for trained models
│   └── 40.pth               # Pre-trained VGG16 model weights
└── README.md
```

---

## 🧠 Model Architecture

**Base Model**: VGG16 (pretrained on ImageNet)

```
Input (224×224×3)
    ↓
Feature Extraction (Frozen: features.0-26)
    ↓
Fine-tuning (Trainable: features.27-30)
    ↓
Global Average Pooling
    ↓
Classifier (4096 → 4096 → 4)
    ↓
Output (4 classes)
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 40
- Learning Rate Scheduler: ReduceLROnPlateau

---

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate per class
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision & recall
- **Confusion Matrix**: Per-class performance breakdown

Run evaluation:
```bash
python evaluate_model.py
```

---

## 🔍 Model Interpretability

The `demo.py` script provides Grad-CAM visualization showing which image regions influenced the model's decision.

```bash
python demo.py
```

This creates a side-by-side comparison of:
1. Original image
2. Activation heatmap
3. Overlaid prediction

---

## 💡 How It Works

1. **Image Input**: User uploads or captures an image
2. **Preprocessing**: Resize to 224×224 and normalize
3. **Feature Extraction**: VGG16 features layer processes the image
4. **Classification**: Fully connected layers output 4 class probabilities
5. **Result Display**: Shows predicted class with confidence

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Module not found: sklearn` | Run `pip install scikit-learn` |
| `Model file not found` | Ensure `model/40.pth` exists or run `python generate_model.py` |
| `Dataset not found` | Place dataset folder in parent directory of this project |
| `CUDA out of memory` | Reduce batch size in scripts (line 32 in train_model.py: `batch_size = 16`) |
| Webcam not working | Check if another app is using the camera; restart the app |

---

## 📚 Technologies Used

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch 2.6.0, TorchVision 0.21.0 |
| **Model Architecture** | VGG16 (Transfer Learning) |
| **Data Processing** | Scikit-learn, NumPy, Pandas |
| **Web Framework** | Streamlit 1.42.0 |
| **Object Detection** | YOLOv8 (optional) |
| **Visualization** | Matplotlib, OpenCV, Plotly |
| **Notebook Environment** | Jupyter |

---

## 📝 Training & Evaluation

### To Train from Scratch
1. Open `waste classifier.ipynb` in Jupyter
2. Follow the cells sequentially
3. Model will be saved as `model/40.pth`

### To Evaluate a Trained Model
```bash
python evaluate_model.py
```

This generates:
- Classification report
- Confusion matrix
- ROC curves
- Per-class metrics

---

## 🎓 Learning Outcomes

By studying this project, you'll learn:
- ✅ Transfer learning with pretrained models
- ✅ CNN architecture and fine-tuning
- ✅ Image preprocessing & augmentation
- ✅ Model evaluation & metrics
- ✅ Building interactive ML applications
- ✅ Real-time inference & visualization

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Copyright © 2025** | Built as an educational project for intelligent waste management

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Improve documentation
- Submit pull requests with enhancements

---

## 📧 Contact & Support

For questions or suggestions regarding this project, please reach out or open an issue in the repository.

**Last Updated**: March 2025  
**Status**: ✅ Production Ready
