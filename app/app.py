import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

# --- App Configuration ---
st.set_page_config(
    page_title="Car Classifier",
    page_icon="🚗",
    layout="centered"
)

# --- Constants ---
IMG_SIZE = 224
MODEL_PATH = "/Users/tushar04master/Documents/car-classifier/models/checkpoints/best_model.keras"
META_CSV_PATH = "/Users/tushar04master/Documents/car-classifier/data/devkit/cars_meta.csv" 

@st.cache_resource
def load_model_and_meta():
    """
    Loads the trained Keras model and the class names from the CSV file.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        meta_df = pd.read_csv(META_CSV_PATH)
        # The dataset classes are 1-indexed, but Python lists are 0-indexed.
        # We create a list where the index matches (class_id - 1).
        class_names = meta_df['class_name'].tolist()
        return model, class_names
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please make sure '{MODEL_PATH}' and '{META_CSV_PATH}' are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model or metadata: {e}")
        st.stop()

# --- Image Preprocessing and Prediction ---
def predict(model, image, class_names):
    """
    Takes an uploaded image, preprocesses it, and returns the predicted class name and confidence.
    """
    # 1. Preprocess the image to match model's input requirements
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch

    # 2. Make a prediction
    predictions = model.predict(img_array)
    
    # 3. Decode the prediction
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class_name, confidence

# --- Main App Interface ---
st.title("🚗 Stanford Cars Classifier")
st.markdown("Upload an image of a car, and the model will predict its make, model, and year.")

# Load the model and class names
model, class_names = load_model_and_meta()

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Create a button to trigger the classification
    if st.button('Classify Car', use_container_width=True):
        with st.spinner('Classifying...'):
            # Make prediction
            predicted_class, confidence = predict(model, image, class_names)
            
            # Display the result
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

else:
    st.info("Please upload an image file to get started.")

st.markdown("---")
st.markdown("Built with a MobileNetV2 model fine-tuned on the Stanford Cars dataset.")
