import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pickle
import cv2
from base64 import b64decode

# Load the voice recognition model from JSON file
def load_voice_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

# Load data using pickle (for scalers and encoders)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)
    return np.squeeze(mfccs.T) if not flatten else np.ravel(mfccs.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    features = np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return features

def get_predict_feat(path, scaler):
    data, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, s_rate)
    features = np.reshape(features, newshape=(1, -1))
    scaled_features = scaler.transform(features)
    scaled_features = np.expand_dims(scaled_features, axis=2)
    return scaled_features

# Predict emotion from voice using extracted features
def voice_prediction(model, audio_path, scaler):
    scaled_features = get_predict_feat(audio_path, scaler)
    predictions = model.predict(scaled_features)
    y_pred = np.argmax(predictions, axis=1)
    emotions = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fear', 6: 'Disgust', 7: 'Surprise'}
    return emotions[y_pred[0]]

# Process image and audio paths
def process_image_and_audio(face_emotion_model, voice_model, scaler, image_path, audio_path):
    face_emotion = predict_emotion(image_path, face_emotion_model)
    voice_emotion = voice_prediction(voice_model, audio_path, scaler)

    result = {
        "Face Emotion": face_emotion,
        "Voice Emotion": voice_emotion
    }
    return result

# Streamlit user interface
st.title("Emotion Detection from Image and Audio")
st.header("Upload your image and audio files")

# Load models
face_model = load_face_model()
voice_model = load_voice_model('path_to_json_model', 'path_to_model_weights')
scaler2 = load_pickle('path_to_scaler.pickle')

uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if st.button("Analyze Emotions"):
    if uploaded_image is not None and uploaded_audio is not None:
        # Save uploaded files to a temporary location
        image_path = f"temp_{uploaded_image.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())
        audio_path = f"temp_{uploaded_audio.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())

        # Process emotions
        results = process_image_and_audio(face_model, voice_model, scaler2, image_path, audio_path)
        st.write(f"Detected Face Emotion: {results['Face Emotion']}")
        st.write(f"Detected Voice Emotion: {results['Voice Emotion']}")
    else:
        st.warning("Please upload both image and audio files.")
