import streamlit as st
from fastai.vision.all import *
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Dhaniya-Pudina Classifier",
    page_icon="🌿",
    layout="centered"
)

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return load_learner('model.pkl')

learn = load_model()

# UI
st.title("🌿 Dhaniya - Pudina Classifier")
st.write("Upload a photo of Dhaniya (Coriander) or Pudina (Mint) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                # Create fastai image from the uploaded file
                img = PILImage.create(uploaded_file)

                # Get prediction
                pred, pred_idx, probs = learn.predict(img)

                # Display results
                st.success(f"**Prediction: {pred}**")
                st.write(f"**Confidence: {probs[pred_idx]:.2%}**")

                # Show probabilities for each class
                st.write("### Probabilities:")
                for i, cat in enumerate(learn.dls.vocab):
                    st.write(f"- {cat}: {probs[i]:.2%}")

            except Exception as e:
                st.error(f"Error during classification: {e}")
                st.write("Please try uploading a different image.")

