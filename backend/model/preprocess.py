import librosa
import numpy as np
import io

def preprocess_audio(audio_file):
    """Preprocesses audio file for model input."""
    y, sr = librosa.load(audio_file, sr=16000)  
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  
    mfccs = np.mean(mfccs, axis=1)  
    return np.expand_dims(mfccs, axis=0)  
