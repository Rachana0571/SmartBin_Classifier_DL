"""
SmartBin Waste Classification - Streamlit Web Application
Web interface for waste classification using trained VGG16 CNN model
Supports image upload and webcam capture
"""

import streamlit as st
from PIL import Image
from main import load_model, predict_image_class
import io


st.set_page_config(
    page_title="♻️ Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# Mobile-responsive CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        padding: 1rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    .stImage {
        max-width: 100%;
    }
    @media (max-width: 640px) {
        [data-testid="stColumn"] {
            width: 100% !important;
        }
        h1 {
            font-size: 1.5rem;
        }
        h2 {
            font-size: 1.2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


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
**Biodegradable** • **Non-Biodegradable** • **Trash** • **Hazardous**
""")

# Use expanders for mobile-friendly layout
with st.expander("📤 Upload Image", expanded=True):
    
    # Show upload button if no image is uploaded
    if not st.session_state.uploaded_image:
        if st.button("📤 Upload Image", use_container_width=True, key="upload_trigger_btn"):
            st.session_state.show_uploader = True
            st.rerun()
    
    # Show file uploader only if button was clicked
    if st.session_state.show_uploader:
        col_upload, col_close = st.columns([3, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key=f"uploader_{st.session_state.upload_key}")
        
        with col_close:
            if st.button("❌", key="close_uploader_btn", help="Close uploader"):
                st.session_state.show_uploader = False
                st.session_state.uploaded_image = None
                st.session_state.predicted_class = None
                st.session_state.last_file_name = None
                st.session_state.upload_key += 1
                st.rerun()
        
        # Process uploaded file
        if uploaded_file is not None:
            st.session_state.last_file_name = uploaded_file.name
            
            # Check file extension
            valid_extensions = ['jpg', 'jpeg', 'png']
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext not in valid_extensions:
                st.error(f"❌ **Unsupported file format:** .{file_ext}\n\nPlease upload: JPG, JPEG, or PNG")
                st.session_state.uploaded_image = None
                st.session_state.predicted_class = None
            else:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.session_state.uploaded_image = image
                    st.session_state.show_uploader = False
                    
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    with st.spinner("🔍 Analyzing image..."):
                        predicted_class = predict_image_class(image, model)
                        st.session_state.predicted_class = predicted_class
                    
                    st.success(f"✅ **Predicted Class:** {predicted_class}")
                except Exception as e:
                    st.error(f"❌ **Error processing image:**\n{str(e)}\n\nPlease try another image.")
                    st.session_state.uploaded_image = None
                    st.session_state.predicted_class = None
    
    # Show result and clear button if image is uploaded
    if st.session_state.uploaded_image and st.session_state.predicted_class:
        col_result, col_clear = st.columns([3, 1])
        
        with col_result:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
            st.success(f"✅ **Predicted Class:** {st.session_state.predicted_class}")
        
        with col_clear:
            st.write("")
            st.write("")
            if st.button("🔄 Clear", key="clear_btn", use_container_width=True):
                st.session_state.uploaded_image = None
                st.session_state.predicted_class = None
                st.session_state.last_file_name = None
                st.session_state.upload_key += 1
                st.session_state.show_uploader = False
                st.rerun()

with st.expander("📷 Capture Photo", expanded=False):
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert('RGB')
            st.image(image, caption="Captured Image", use_container_width=True)
            
            with st.spinner("🔍 Analyzing image..."):
                predicted_class = predict_image_class(image, model)
            
            st.success(f"✅ **Predicted Class:** {predicted_class}")
        except Exception as e:
            st.error(f"❌ Error processing camera image: {str(e)}")


# Information section
st.divider()

with st.expander("ℹ️ About This Project", expanded=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Classes:**")
        st.write("""
        🟢 **Biodegradable**  
        Organic waste (food, paper, cardboard)
        
        🔵 **Non-Biodegradable**  
        Plastics, metals, glass
        
        🟡 **Trash**  
        Mixed/unusable materials
        
        🔴 **Hazardous**  
        Dangerous materials (chemicals, batteries)
        """)
    
    with col2:
        st.write("**Technology:**")
        st.write("""
        • CNN using VGG16 architecture
        • Transfer learning with ImageNet
        • 90.67% accuracy
        • 15,000+ training images
        • 32 categories → 4 classes
        """)

st.caption("♻️ SmartBin: Intelligent Waste Classification System | Powered by Deep Learning")