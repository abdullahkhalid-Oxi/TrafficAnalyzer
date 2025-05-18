from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import numpy as np
import tensorflow_hub as hub

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')
print(model)

# Preprocess audio
def preprocess_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = librosa.util.fix_length(waveform, size=16000)
    print("Waveform shape:", waveform.shape, "Sample rate:", sr)
    return waveform

# Classify audio
def classify_audio(file_path):
    waveform = preprocess_audio(file_path)
    scores, _, _ = model(waveform)
    predicted_idx = np.argmax(scores, axis=-1)[0]
    intensity = np.mean(np.abs(waveform))
    if predicted_idx == 387:  # Traffic noise, roadway noise
        level = "Low" if intensity < 0.01 else "Moderate" if intensity < 0.25 else "Severe"
        print(f"Predicted: Traffic Jam, Blockage: {level}")
    elif predicted_idx == 318:  # Ambulance (siren)
        level = "Low" if intensity < 0.01 else "Medium" if intensity < 0.3 else "High"
        print(f"Predicted: Ambulance, Urgency: {level}")
    elif predicted_idx == 297:  # Crash, bang, smash
        level = "Mild" if intensity < 0.02 else "Severe"
        print(f"Predicted: Accident, Severity: {level}")
    elif predicted_idx == 375:  # Car horn
        level = "Light" if intensity < 0.01 else "Heavy"
        print(f"Predicted: Horns, Traffic Rush: {level}")
    else:
        print("No relevant sound detected.")
        print(predicted_idx)

print(model)
# Extract audio from video and classify
video = VideoFileClip('carhorn.mp4')
audio = video.audio
audio.write_audiofile('extracted_audio8.wav')
classify_audio('extracted_audio8.wav')