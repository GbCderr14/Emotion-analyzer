import os
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from flask import Flask, render_template, request, jsonify
import joblib
import soundfile
from python_speech_features import mfcc , logfbank

app = Flask(__name__)

# Load your trained model
# model = joblib.load(r'C:\Users\Lenovo\OneDrive\Desktop\ai proj\Speech_Emotion_Detection\Emotion_Voice_Detection_Model.pkl')
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your trained model using a relative path
model_path = os.path.join(current_dir, 'Emotion_Voice_Detection_Model.pkl')
model = joblib.load(model_path)
# Define a route for the index page
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/js/app.js")
def js():
    return render_template("js/app.js")
@app.route("/js/recorder.js")
def rec_js():
    return render_template("js/recorder.js")

# Define a route for processing audio file uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'})

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if audio_file:
        # Save the uploaded audio file to a temporary location
        current_dir = os.path.dirname(os.path.abspath(__file__))

        uploads_dir = os.path.join(current_dir, 'uploads')
        # uploads_dir = os.path.abspath(r'C:\Users\Lenovo\OneDrive\Desktop\ai proj\Speech_Emotion_Detection\uploads')
        audio_path = os.path.join(uploads_dir, audio_file.filename)
        audio_file.save(audio_path)

        signal , rate = librosa.load(audio_path, sr=16000)
        mask = envelope(signal,rate, 0.0005)
        fname=uploads_dir+"\\audio.wav"
        wavfile.write(filename= fname, rate=rate,data=signal[mask])

        ans =[]
        features = extract_features(audio_path, mfcc=True, chroma=True, mel=True)
        print(features.size)
        ans.append(features)
        ans = np.array(ans)
        # data.shape
        ans = ans[0]
        # Use the trained model to make predictions
        prediction = model.predict([features])

        # Delete the temporary audio file
        # os.remove(audio_path)

        # Return the prediction to the frontend
        return jsonify({'prediction': prediction[0]})
def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def extract_features(file_name, mfcc, chroma, mel):
    # Implement feature extraction from the audio data here
    # You can use the same feature extraction code you mentioned in a previous question
    print(file_name)
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs.reshape(-1)))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma.reshape(-1)))
        if mel:
            # mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result=np.hstack((result, mel.reshape(-1)))
        # print(result)
    return result

if __name__ == '__main__':
    app.run(debug=True)
