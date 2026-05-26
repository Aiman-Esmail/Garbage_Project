import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
username = os.getenv("KAGGLE_USERNAME")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Garbage Classifier",
    page_icon="♻️",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
IMG_SIZE = (224, 224)

MODEL_PATHS = [
    "model/garbage_classifier_saved",   # SavedModel (preferred)
    "model/Garbage_Classifier_Final_95.h5",  # H5 fallback
]

RECYCLING_TIPS = {
    "Cardboard": "♻️ Flatten boxes before recycling. Keep dry!",
    "Glass":     "🍶 Rinse bottles. Remove lids before recycling.",
    "Metal":     "🥫 Crush cans to save space. Rinse food residue.",
    "Paper":     "📄 Keep paper dry and clean for recycling.",
    "Plastic":   "🧴 Check the resin code. Rinse containers.",
    "Trash":     "🗑️ This item goes to general waste.",
}

# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                return model, path
            except Exception as e:
                st.warning(f"Could not load `{path}`: {e}")
    return None, None

# ── Image Preprocessor ────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Predictor ─────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    tensor = preprocess(image)
    preds = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]) * 100, preds

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Status")
    model, model_path = load_model()

    if model:
        st.success("✅ Model loaded")
        st.caption(f"Source: `{model_path}`")
    else:
        st.error("❌ No model found")
        st.info("Run `python convert_model.py` to convert your H5 model.")

    st.divider()
    st.markdown("**Classes**")
    for c in CLASSES:
        st.markdown(f"- {c}")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("♻️ AI Garbage Classifier")
st.caption("Upload an image and the AI will classify the type of garbage.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.error("⚠️ Model not available. Please check the sidebar for instructions.")
        st.stop()

    with st.spinner("Analyzing image..."):
        label, confidence, probs = predict(model, img)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏷️ Prediction", label)
    with col2:
        st.metric("🎯 Confidence", f"{confidence:.1f}%")

    st.info(RECYCLING_TIPS[label])

    # ── Probability Chart ─────────────────────────────────────────────────────
    st.write("### 📊 Confidence per Category")
    prob_df = pd.DataFrame({
        "Category": CLASSES,
        "Score": probs
    }).sort_values("Score", ascending=False)

    st.bar_chart(prob_df.set_index("Category")["Score"])

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with st.expander("🔬 Technical Details"):
        st.dataframe(
            prob_df.style.format({"Score": "{:.4f}"}),
            use_container_width=True
        )
        st.caption(f"Model path: `{model_path}`")
        st.caption(f"Input shape: {IMG_SIZE[0]}×{IMG_SIZE[1]}×3")
