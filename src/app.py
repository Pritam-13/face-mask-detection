import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("😷 Face Mask Detection")
st.write("Upload an image to check if a person is wearing a face mask.")

@st.cache_resource
def load_mask_model():
    return load_model("model/mask_detector.h5")

model = load_mask_model()
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_preprocessed = preprocess_input(img_to_array(img_resized))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    (mask, noMask) = model.predict(img_batch)[0]
    label = "Mask ✅" if mask > noMask else "No Mask ❌"
    confidence = max(mask, noMask) * 100

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")