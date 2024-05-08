# Emotion-Recognition-System

# Multimodal Emotion Recognition System

Welcome to the Multimodal Emotion Recognition System! This project combines facial and voice emotion recognition using state-of-the-art deep learning techniques. By leveraging facial expressions and vocal tones, the system can accurately identify a range of human emotions in real time.

## Features

- **Facial Emotion Recognition**
  - Uses a convolutional neural network (CNN) based on EfficientNetB7 to classify emotions from facial expressions.
- **Voice Emotion Recognition**
  - Employs a CNN-LSTM model to analyze audio recordings and determine emotional states from vocal features.
- **Real-Time Camera and Microphone Integration**
  - Captures facial images and voice recordings directly, allowing for live emotion recognition.
- **Simple and Hybrid Integration**
  - Offers two modes of combining facial and voice data for emotion prediction:
  - **Simple Mode**
    - Each model works independently to classify emotions.
  - **Hybrid Mode**
    - Combines the outputs of both models to provide a unified emotion prediction.

## Datasets and Models

- **Datasets**
  - [Facial Emotion Detection (FER)](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
  - [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

- **Pre-Trained Models**
  - [Pre-Trained Models (Google Drive)](https://drive.google.com/drive/folders/1LzEb-L8a8PGiAU6_3DXEdiRYccSr2A--?usp=share_link)

## Results

- **Accuracy**
  - Both facial and voice models demonstrated high accuracy in classifying emotions.
- **Confusion Matrices**
  - Confusion matrices for both models provide insights into model performance and areas for improvement.

## Future Work

1. **Expand Emotion Categories** - Include more subtle and culturally diverse emotional expressions.
2. **Improve Model Fusion** - Develop smarter algorithms to combine the strengths of both facial and vocal data.
3. **Real-World Applications** - Integrate this system into real-world applications such as mental health monitoring and educational tools.

## Acknowledgments

Special thanks to the open-source community and dataset contributors who made this project possible!
