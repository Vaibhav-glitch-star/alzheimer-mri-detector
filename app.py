import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = load_model("alzheimer_classifier.h5")

# Define class labels
classes = ['Alzheimer', 'Normal']

# Streamlit UI
st.title("Alzheimer Detection from MRI Brain Image")
st.write("Upload a brain MRI image (JPEG or PNG) to detect Alzheimer's disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess the image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    label = classes[np.argmax(prediction)]

    # Output
    st.subheader("Prediction:")
    st.success(f"The MRI image is classified as: **{label}**")
