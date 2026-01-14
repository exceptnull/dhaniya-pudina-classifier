import streamlit as st
from fastai.vision.all import *
from PIL import Image

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
categories = ('Dhaniya (Coriander)', 'Pudina (Mint)')

# UI
st.title("🌿 Dhaniya - Pudina Classifier")
st.write("Upload a photo of Dhaniya (Coriander) or Pudina (Mint) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict button
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            # Convert to fastai format
            try:
                img = PILImage.create(uploaded_file)
                pred, idx, probs = learn.predict(img)
                
                # Display results
                st.success(f"**Prediction: {pred}**")
                st.write(f"**Confidence: {probs[idx]:.2%}**")
                
                # Show probabilities
                st.write("### Probabilities:")
                for i, cat in enumerate(categories):
                    st.write(f"- {cat}: {probs[i]:.2%}")
            except Exception as e:
                st.error(e)