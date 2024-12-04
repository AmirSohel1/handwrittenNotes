import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Load trained generator model
@st.cache_resource
def load_generator_model():
    # Replace with the path to your trained generator model
    model = load_model("generator_model.h5")
    return model

generator = load_generator_model()

# Image preprocessing function
def preprocess_image(uploaded_image):
    image = np.array(uploaded_image.convert("L"))  # Convert to grayscale
    resized = cv2.resize(image, (128, 128))  # Resize to model input size
    normalized = resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(normalized, axis=(0, -1))  # Add batch and channel dims

# Generate handwriting using the generator
def generate_handwriting(generator, num_images=1):
    noise = np.random.randn(num_images, 100)  # Random noise
    generated_images = generator.predict(noise)
    return generated_images.squeeze()

# Streamlit UI
st.title("Handwriting Generator")
st.write("Upload an image to preprocess and generate handwriting!")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    uploaded_image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(uploaded_image)
    st.write("Preprocessed Image:")
    st.image(preprocessed_image.squeeze(), caption="Preprocessed", use_column_width=True, clamp=True)

    # Generate handwriting
    st.write("Generated Handwriting:")
    generated_images = generate_handwriting(generator, num_images=5)

    # Display generated images
    for i, img in enumerate(generated_images):
        st.image(img, caption=f"Generated Image {i+1}", use_column_width=True, clamp=True)
