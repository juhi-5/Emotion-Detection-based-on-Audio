import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
import pyaudio
import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import librosa

# Load gender detection model
def load_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

emotion_model = load_model('emotion_model_json1.json', 'Emotion_Model1.h5')
gender_model = load_model('gender_model_json1.json', 'Gender_Model1.h5')

# Record audio
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "temp.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    process_audio(WAVE_OUTPUT_FILENAME)

# Upload audio
def upload_audio():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_audio(file_path)

# Process audio
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17).T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)

    # Predict gender
    gender_pred = gender_model.predict(mfccs)
    if np.argmax(gender_pred) == 0:  # Assuming '0' corresponds to female
        # Predict emotion
        emotion_pred = emotion_model.predict(mfccs)
        emotion = np.argmax(emotion_pred)
        emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'pleasant surprise', 6: 'sad'}
        emotion_label.config(text=f"Detected Emotion: {emotion_dict[emotion]}")
    else:
        messagebox.showerror("Please upload a female voice.")

# Create GUI
root = tk.Tk()
root.title("Emotion Detection from Voice")
root.geometry("600x600")
root.configure(background='#CDCDCD')
heading = Label(root,text='Emotion Detector based on Audio',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()

record_button = tk.Button(root, text="Record Voice", command=record_audio)
record_button.pack(pady=10)

upload_button = tk.Button(root, text="Upload Voice", command=upload_audio)
upload_button.pack(pady=10)

emotion_label = tk.Label(root, text="Detected Emotion")
emotion_label.pack(pady=10)

root.mainloop()

root = tk.Tk()
