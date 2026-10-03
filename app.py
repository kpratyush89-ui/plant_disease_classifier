import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Mango Leaf Disease Detector",
    page_icon="🥭",
    layout="centered"
)

# Title & Subtitle
st.title("🥭 Mango Leaf Disease Diagnostic System")
st.write("Upload a clear photograph of a mango leaf to detect potential diseases using deep learning.")

# TFLite Model Loader with Cache
@st.cache_resource
def load_tflite_model():
    model_path = "mango_disease_model.tflite"
    if not os.path.exists(model_path):
        st.error(f"Model file missing! Please upload '{model_path}' to your project directory.")
        st.stop()
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

with st.spinner("Loading TFLite AI Model..."):
    interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class Labels
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# File Uploader
uploaded_file = st.file_uploader("Choose a mango leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Inference Button
    if st.button("🔍 Diagnose Leaf Health"):
        with st.spinner("Analyzing leaf features with TFLite..."):
            # Set tensor and invoke interpreter
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()
            raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

            # Apply Softmax normalization if output is raw logits
            if raw_output.max() > 1.0 or raw_output.min() < 0.0:
                score = tf.nn.softmax(raw_output).numpy()
            else:
                score = raw_output

            # Top 2 Predictions Extraction
            top2_indices = np.argsort(score)[-2:][::-1]
            top1_class = CLASS_NAMES[top2_indices[0]]
            top1_prob = float(score[top2_indices[0]] * 100)
            top2_class = CLASS_NAMES[top2_indices[1]]
            top2_prob = float(score[top2_indices[1]] * 100)

        st.divider()
        st.subheader("Diagnostic Results")

        # Primary Output
        if top1_class == "Healthy":
            st.success(f"**Primary Diagnosis:** {top1_class}")
        else:
            st.error(f"**Primary Diagnosis:** {top1_class}")

        st.metric(label="Primary Confidence", value=f"{top1_prob:.2f}%")
        st.write(f"**Secondary Possibility:** {top2_class} ({top2_prob:.2f}%)")

        # Low Confidence Warning Trigger
        if top1_prob < 70.0:
            st.warning("⚠ **Low Confidence Detection:** The leaf may have early-stage/faint symptoms or non-pathogenic physical damage. Consider cropping tightly around the affected discolored area and re-uploading.")

        # Full Probability Breakdown
        st.subheader("Full Disease Probability Breakdown")
        for idx, name in enumerate(CLASS_NAMES):
            prob = float(score[idx] * 100)
            st.write(f"**{name}**: {prob:.2f}%")
            st.progress(int(prob))
    
    
