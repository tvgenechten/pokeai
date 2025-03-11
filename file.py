import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model
import time

# Streamlit-app titel
st.title("🎥 Live & Upload Afbeeldingsclassificatie met Keras")

# Laad model en labels
@st.cache_resource
def load_keras_model():
    return load_model("keras_model.h5", compile=False)

@st.cache_resource
def load_labels():
    return [line.strip().split(" ", 1)[1] for line in open("labels.txt", "r").readlines()]

model = load_keras_model()
class_names = load_labels()

# Sidebar opties
st.sidebar.subheader("📡 Kies invoermethode")
input_option = st.sidebar.radio("Selecteer:", ("📂 Upload Afbeelding", "🎥 Live Webcam"))

# Afbeeldingsgrootte voor model
size = (224, 224)

def predict_image(image):
    """Verwerkt een afbeelding en doet een voorspelling."""
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Model voorspelling
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Upload afbeelding
if input_option == "📂 Upload Afbeelding":
    uploaded_file = st.file_uploader("Upload een afbeelding...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Geüploade afbeelding", use_container_width=True)

        # Voorspelling tonen
        class_name, confidence_score = predict_image(image)
        st.subheader("Resultaat 📊")
        st.write(f"**Klasse:** {class_name}")
        st.write(f"**Vertrouwensscore:** {confidence_score:.2%}")

# Live Webcam (st.camera_input wordt gebruikt in plaats van OpenCV)
elif input_option == "🎥 Live Webcam":
    st.subheader("📸 Neem een foto met je webcam")
    webcam_image = st.camera_input("Klik op de knop om een foto te maken")

    if webcam_image is not None:
        image = Image.open(webcam_image).convert("RGB")
        st.image(image, caption="📷 Opgenomen afbeelding", use_column_width=True)

        # Voorspelling tonen
        class_name, confidence_score = predict_image(image)
        st.subheader("Resultaat 📊")
        st.write(f"**Klasse:** {class_name}")
        st.write(f"**Vertrouwensscore:** {confidence_score:.2%}")
