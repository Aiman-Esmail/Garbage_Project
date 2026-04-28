import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Garbage Classifier",
    page_icon="🗑️",
    layout="centered"
)

st.title("🗑️ Garbage Classifier")
st.write("Upload an image to classify garbage items")

# Load model
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Keras model"""
    model_path = os.path.join(os.path.dirname(__file__), "model", "Garbage_Classifier_Final_95.h5")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image for model
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image to match model input requirements"""
    if img.size != target_size:
        img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Make prediction
def predict_garbage_class(img_array, model):
    """Make prediction using the model"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return predicted_class, confidence, predictions[0]

# Load the model
model = load_trained_model()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "gif", "bmp"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image_obj = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_obj, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        img_array = preprocess_image(image_obj)
        predicted_class, confidence, all_predictions = predict_garbage_class(img_array, model)
        
        # Get class labels (adjust these based on your model's training classes)
        # Common garbage classes: recyclable, organic, metal, paper, plastic, etc.
        class_labels = [
            "Cardboard",
            "Glass",
            "Metal",
            "Paper",
            "Plastic",
            "Trash"
        ]
        
        # Ensure we have enough labels
        if len(all_predictions) > len(class_labels):
            class_labels = [f"Class {i}" for i in range(len(all_predictions))]
        
        with col2:
            st.subheader("Classification Result")
            st.metric("Predicted Class", class_labels[predicted_class])
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            
            # Show all class probabilities
            st.subheader("Class Probabilities")
            for i, (label, prob) in enumerate(zip(class_labels[:len(all_predictions)], all_predictions)):
                st.write(f"{label}: {prob * 100:.2f}%")
else:
    st.error("Failed to load the model. Please check the model file path.")
