import streamlit as st
import zipfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
from PIL import Image
import cv2
import numpy as np
import librosa
import pickle
from base64 import b64decode
from IPython.display import display, Javascript, Image as IPythonImage

# Load and prepare the face recognition model
def load_face_model():
    model = load_model('/content/drive/MyDrive/Model/face_recognition.h5', compile=False)
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Predict emotion from an image
def predict_emotion(image_path, model):
    image = Image.open(image_path)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    score = tf.nn.softmax(predictions[0])
    return class_labels[np.argmax(score)]

# Load the voice recognition model from JSON file
def load_voice_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    print("Loaded voice model from disk")
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

# Main processing function to analyze both image and audio
def process_image_and_audio(image_path, audio_path):
    face_model = load_face_model()
    face_emotion = predict_emotion(image_path, face_model)
    st.write(f"Detected Face Emotion: {face_emotion}")

    voice_model = load_voice_model('/content/Models/Speech Recognition/Model/CNN_model.json', '/content/Models/Speech Recognition/Model/CNN_model_weights.h5')
    scaler2 = load_pickle('/content/drive/MyDrive/Model/scaler2.pickle')

    voice_emotion = voice_prediction(voice_model, audio_path, scaler2)
    st.write(f"Detected Voice Emotion: {voice_emotion}")

    if face_emotion == 'Happy' and voice_emotion in ['Happy', 'Calm']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Sad' and voice_emotion in ['Sad', 'Calm']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Angry' and voice_emotion in ['Angry', 'Fear']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Fear' and voice_emotion in ['Fear', 'Surprise']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Surprise' and voice_emotion in ['Surprise', 'Happy']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Disgust' and voice_emotion in ['Disgust', 'Angry']:
        st.write("Both face and voice are consistent and positive.")
    elif face_emotion == 'Neutral' and voice_emotion in ['Neutral', 'Calm']:
        st.write("Both face and voice are consistent and positive.")
    else:
        st.write("Face and voice emotions do not align well or are not consistent.")

# Streamlit app
def main():
    st.title("Emotion Analysis")

    image_path = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    audio_path = st.file_uploader("Upload Audio", type=['wav'])

    if st.button("Process"):
        if image_path is not None and audio_path is not None:
            process_image_and_audio(image_path, audio_path)
        else:
            st.write("Please upload both image and audio files.")

if __name__ == "__main__":
    main()