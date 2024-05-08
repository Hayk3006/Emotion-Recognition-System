# streamlit_app.py
pip install soundfile

import streamlit as st
import numpy as np
import soundfile as sf
from PIL import Image
import io
import os

# Import your existing function
from your_existing_module import process_image_and_audio

# Streamlit interface
st.title("Multimodal Emotion Detection")

# Image capture/upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Save image to a path or process in memory
    image_path = "/tmp/uploaded_image.jpg"
    image.save(image_path)

# Audio capture/upload
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_audio is not None:
    audio_data, samplerate = sf.read(io.BytesIO(uploaded_audio.read()))
    # Save audio to a path or process in memory
    audio_path = "/tmp/uploaded_audio.wav"
    sf.write(audio_path, audio_data, samplerate)

# Process if both image and audio are uploaded
if uploaded_image and uploaded_audio:
    st.write("Processing...")
    process_image_and_audio(image_path, audio_path)
    st.write("Processing completed.")

