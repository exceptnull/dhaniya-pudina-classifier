import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib

# Page config
st.set_page_config(
    page_title="Dhaniya-Pudina Classifier",
    page_icon="🌿",
    layout="centered"
)

# Load model (cached to avoid reloading)
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
    st.image(image, caption='Uploaded Image', width='stretch')
    
    # Predict button
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                # Create a new dataloader with the uploaded image
                dl = learn.dls.test_dl([image], with_labels=False)
                
                # Get prediction
                preds, _ = learn.get_preds(dl=dl)
                pred_idx = preds.argmax()
                pred = learn.dls.vocab[pred_idx]
                
                # Display results
                st.success(f"**Prediction: {pred}**")
                st.write(f"**Confidence: {preds[0][pred_idx]:.2%}**")
                
                # Show probabilities
                st.write("### Probabilities:")
                for i, cat in enumerate(learn.dls.vocab):
                    st.write(f"- {cat}: {preds[0][i]:.2%}")

            except Exception as e:
                st.error(e)
