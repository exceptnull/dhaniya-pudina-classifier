import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Page config
st.set_page_config(
    page_title="Dhaniya-Pudina Classifier",
    page_icon="🌿",
    layout="centered"
)

# Load model once and cache it
@st.cache_resource
def load_model():
    return load_learner('model.pkl')

learn = load_model()

st.title("🌿 Dhaniya - Pudina Classifier")
st.write("Upload a photo of Dhaniya (Coriander) or Pudina (Mint) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open with Pillow
    img = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        try:
            # Directly pass Pillow image to fastai
            pred, pred_idx, probs = learn.predict(img)

            st.success(f"Prediction: {pred}")
            st.write(f"Confidence: {probs[pred_idx]:.2%}")

            st.write("Probabilities:")
            for i, cat in enumerate(learn.dls.vocab):
                st.write(f"- {cat}: {probs[i]:.2%}")
        except Exception as e:
            st.error(f"Error during classification: {e}")

