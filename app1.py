# app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

@st.cache_resource
def load_model(model_path="cat_dog_model.h5", model_url=None):
    # If model file not present and model_url provided, download it
    if not os.path.exists(model_path) and model_url:
        import requests
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = tf.keras.models.load_model(model_path)
    return model

# If you put model in repo, leave model_url=None.
# If your model is hosted elsewhere (public link), set model_url to that raw link.
MODEL_URL = "https://colab.research.google.com/drive/1a_abPgFE8g_JdBXsi0piANP0epXO5Atj?usp=sharing"  # e.g. "https://huggingface.co/username/repo/resolve/main/cat_dog_model.h5"
model = load_model(model_path="cat_dog_model.h5", model_url=MODEL_URL)

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it's a Cat or a Dog.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_batch)[0][0]
    if pred > 0.5:
        label = "Dog 🐶"
        confidence = pred
    else:
        label = "Cat 🐱"
        confidence = 1 - pred

    st.markdown(f"### Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
