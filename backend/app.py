import tensorflow as tf
import librosa
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# app = Flask(__name__)

# Load the saved model (update this path to match your local system)
# MODEL_PATH = "model/cnn_melspecs_model.h5"
# MODEL_PATH = "model/audio_deepfake_model.h5"
MODEL_PATH = "model/new_cnn_mfcc_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess audio file and extract MFCC features
def preprocess_audio(file_path, max_length=500):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Pad or trim the MFCCs
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]

        return mfccs
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None

# Function to make a prediction
# def predict_audio(file_path, model, max_length=500):
#     mfcc_features = preprocess_audio(file_path, max_length)
#     if mfcc_features is None:
#         return None, None

#     # Reshape for the model
#     mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
#     mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension

#     # Predict using the model
#     prediction = model.predict(mfcc_features)
#     # confidence = prediction[0][0]
#     predicted_label = "Bonafide (Real)" if prediction[0][0] < 0.5 else "Spoofed (Fake)"
#     # Determine label
#     # label = 'Real' if confidence < 0.5 else 'Fake'
#     # confidence_level = confidence if label == 'Fake' else 1 - confidence
#     print("Predicted label ",predicted_label)
#     # return label, confidence_level
#     return predicted_label
def predict_audio(file_path, model, max_length=500):
    mfcc_features = preprocess_audio(file_path, max_length)
    
    if mfcc_features is None:
        print("Error: preprocess_audio() returned None.")
        return None

    print("MFCC shape before expansion:", mfcc_features.shape)  # Debugging
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension
    print("MFCC shape after expansion:", mfcc_features.shape)  # Debugging

    # Predict using the model
    prediction = model.predict(mfcc_features)
    print("Raw Model Prediction:", prediction)  # Debugging

    predicted_label = "Bonafide (Real)" if prediction[0][0] < 0.5 else "Spoofed (Fake)"
    return predicted_label


# Flask route to handle file uploads and predictions
@app.route('/detect', methods=['POST'])
def detect():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)

    # Make prediction
    # label, confidence = predict_audio(temp_path, model)
    label= predict_audio(temp_path, model)
    # Remove temporary file
    os.remove(temp_path)

    if label:
        # return jsonify({'prediction': label, 'confidence': round(float(confidence), 2)})
        return jsonify({'prediction': label})
    else:
        return jsonify({'error': 'Failed to process the audio'}), 500

if __name__ == '__main__':
    app.run(debug=True)
