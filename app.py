import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite Model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="mango_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

st.title("🥭 Mango Leaf Disease Detector")
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Diagnose Leaf Health"):
        # TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)

        st.metric(label="Diagnosis", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")