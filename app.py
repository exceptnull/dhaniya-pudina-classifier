import streamlit as st
from fastai.vision.all import *
from PIL import Image
import io
import pathlib

# Page config
st.set_page_config(
    page_title="Dhaniya-Pudina Classifier",
    page_icon="🌿",
    layout="centered"
)

# Load model
learn = load_learner('model.pkl')

# UI
st.title("🌿 Dhaniya - Pudina Classifier")
st.write("Upload a photo of Dhaniya (Coriander) or Pudina (Mint) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To buffer image for prediction
    img = PILImage.create(bytes_data)


    # Display image
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption='Uploaded Image', width='stretch')
    
    # Predict button
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                # Get prediction
                pred, pred_idx, probs = learn.predict(img)
                
                # Display results
                st.success(f"**Prediction: {pred}**")
                st.write(f"**Confidence: {probs[pred_idx]:.2%}**")
                
                # Show probabilities
                st.write("### Probabilities:")
                for i, cat in enumerate(learn.dls.vocab):
                    st.write(f"- {cat}: {probs[i]:.2%}")

            except Exception as e:
                st.error(e)
