#-----Run This Code With ``streamlit run main_streamlit.py`` To Run The Web App That Implements This Classification Model--------
#-----To Download the Model used in this app, horses_humans_classifier.keras, visit this link: https://drive.google.com/file/d/1iTWmN8e_aKr-ud8zxiZk92_3UIdolRvU/view?usp=sharing#

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def image_preprocessing(file_path: str):

    img = Image.open(file_path).convert('RGB').resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.title("Human & Horse Classifier Using Transfer Learning")
st.write("Upload an image to classify it into one of the following classes:")
st.write("1. Human")
st.write("2. Horse")

model = load_model('horses_humans_classifier.keras')

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png"])

if uploaded_file is not None:
    with st.spinner("Processing..."):

        preprocessed_img = image_preprocessing(uploaded_file)
        prediction = model.predict(preprocessed_img)

        img = Image.open(uploaded_file)
        st.image(img)
        label = "Human" if prediction > 0.5 else "Horse"
        st.markdown(f"<div style='text-align: center; font-size: 2rem; margin-top: 10px;'>Predicted Label: {label}</div>", unsafe_allow_html=True)