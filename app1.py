import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("model.h5")  # Make sure model.h5 is in the same folder

# Title
st.title("🐶 Dog vs 🐱 Cat Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.resize(img, (256, 256))  # Resize
    img = img / 255.0  # Normalize (if you trained with normalization)
    img_reshaped = np.expand_dims(img, axis=0)  # Reshape for prediction

    # Predict
    prediction = model.predict(img_reshaped)

    # Show result
    if prediction[0][0] > 0.5:
        st.success("Prediction: 🐶 Dog")
    else:
        st.success("Prediction: 🐱 Cat")
