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
        st.image(image, caption="📷 Geüploade afbeelding", use_column_width=True)

        # Voorspelling tonen
        class_name, confidence_score = predict_image(image)
        st.subheader("Resultaat 📊")
        st.write(f"**Klasse:** {class_name}")
        st.write(f"**Vertrouwensscore:** {confidence_score:.2%}")

# Live Webcam
elif input_option == "🎥 Live Webcam":
    run_webcam = st.sidebar.toggle("Start Webcam", value=False)

    if run_webcam:
        cap = cv2.VideoCapture(0)  # Open webcam
        stframe = st.empty()  # Video feed container

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Kan geen webcambeelden ophalen")
                break

            # Verwerk frame voor model
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            class_name, confidence_score = predict_image(image)

            # Toon class label op video feed
            cv2.putText(frame, f"{class_name} ({confidence_score:.2%})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Toon video stream
            stframe.image(frame, channels="BGR", use_column_width=True)

            # Slaap kort om CPU-belasting te verminderen
            time.sleep(0.1)

        cap.release()
