#-----Run This Code With ``streamlit run main_streamlit.py`` To Run The Web App That Implements This Classification Model--------
#-----To Download the Model used in this app, horses_humans_classifier.keras, visit this link: https://drive.google.com/file/d/1iTWmN8e_aKr-ud8zxiZk92_3UIdolRvU/view?usp=sharing#

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import urllib.request

def image_preprocessing(file_path: str):

    img = Image.open(file_path).convert('RGB').resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def download_model():
    url = "https://github.com/HyderDeve/TransferLearning/releases/download/v1.0/horses_humans_classifier.keras"
    destination = "horses_humans_classifier.keras"
    progress_bar = st.progress(0)
    status_text = st.empty()

    with urllib.request.urlopen(url) as response, open(destination, "wb") as file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded_size = 0

        while True:
            chunk = response.read(8192)
            if not chunk:
                break

            file.write(chunk)
            downloaded_size += len(chunk)

            if total_size > 0:
                progress = int((downloaded_size / total_size) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Downloading model... {progress}%")

    progress_bar.progress(100)
    status_text.text("Download complete")

st.title("Human & Horse Classifier Using Transfer Learning")
st.write("Upload an image to classify it into one of the following classes:")
st.write("1. Human")
st.write("2. Horse")

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png"])

model = None

try:
    with st.spinner('Loading Model...'):
        model = load_model('horses_humans_classifier.keras')
        st.success('Model loaded successfully.')

except Exception as e:
    st.write(f'''Error: Model not found {e}
                    \n Download Model!!!''')

if st.button('Download Model'):
    with st.spinner('Downloading model...'):
        download_model()
    st.success('Model downloaded successfully. Reload the app to use it.')



if uploaded_file is not None:
    
    try:
        with st.spinner("Processing..."):

            preprocessed_img = image_preprocessing(uploaded_file)
            prediction = model.predict(preprocessed_img)
            img = Image.open(uploaded_file)
            st.image(img)
            label = "Human" if prediction > 0.5 else "Horse"
            st.markdown(f"<div style='text-align: center; font-size: 2rem; margin-top: 10px;'>Predicted Label: {label}</div>", unsafe_allow_html=True)
    
    except Exception as e:
        raise st.write(f'Error: Model not found {e}\n Download Model')
