import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
 
# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
 
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Garbage Classifier",
    page_icon="♻️",
    layout="centered"
)
 
# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = [
    'Battery', 'Biological', 'Brown-Glass', 'Cardboard',
    'Clothes', 'Green-Glass', 'Metal', 'Paper',
    'Plastic', 'Shoes', 'Trash', 'White-Glass'
]
IMG_SIZE = (128, 128)
 
RECYCLING_TIPS = {
    "Battery":     "🔋 Take to a designated battery recycling point. Never bin it!",
    "Biological":  "🌱 Compost organic waste or use a bio bin.",
    "Brown-Glass": "🍺 Rinse and place in the brown glass bin.",
    "Cardboard":   "📦 Flatten boxes before recycling. Keep dry!",
    "Clothes":     "👕 Donate if reusable, or take to a textile recycling point.",
    "Green-Glass": "🍾 Rinse and place in the green glass bin.",
    "Metal":       "🥫 Crush cans to save space. Rinse food residue.",
    "Paper":       "📄 Keep paper dry and clean for recycling.",
    "Plastic":     "🧴 Check the resin code. Rinse containers.",
    "Shoes":       "👟 Donate if wearable, or drop at a shoe recycling point.",
    "Trash":       "🗑️ This item goes to general waste.",
    "White-Glass": "🥛 Rinse and place in the white/clear glass bin.",
}
 
# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    # Try garbage_saved_model with saved_model API
    if os.path.exists("model/garbage_saved_model"):
        try:
            model = tf.saved_model.load("model/garbage_saved_model")
            infer = model.signatures["serving_default"]
            return ("saved_model", infer), "model/garbage_saved_model"
        except Exception as e:
            st.warning(f"saved_model load failed: {e}")
 
    # Try keras models
    for path in ["model/garbage_classifier_saved", "model/Garbage_Classifier_Final_95.h5"]:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                return ("keras", model), path
            except Exception as e:
                st.warning(f"Could not load `{path}`: {e}")
 
    return None, None
 
# ── Image Preprocessor ────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
 
# ── Predictor ─────────────────────────────────────────────────────────────────
def predict(model_tuple, image: Image.Image):
    tensor = preprocess(image)
    model_type, model = model_tuple
 
    if model_type == "saved_model":
        input_tensor = tf.constant(tensor, dtype=tf.float32)
        result = model(input_tensor)
        preds = list(result.values())[0].numpy()[0]
    else:
        preds = model.predict(tensor, verbose=0)[0]
 
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]) * 100, preds
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Status")
    model_tuple, model_path = load_model()
 
    if model_tuple:
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
    st.image(img, caption="Uploaded Image", use_container_width=True)
 
    if model_tuple is None:
        st.error("⚠️ Model not available. Please check the sidebar for instructions.")
        st.stop()
 
    with st.spinner("Analyzing image..."):
        label, confidence, probs = predict(model_tuple, img)
 
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