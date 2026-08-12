from speaker_classifier_modules.model import VoiceClassifier
from speaker_classifier_modules.data import create_dataloaders, split_data
from speaker_classifier_modules.config import DATA_DIR, BEST_MODEL
import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
# test and run ffmpeg to convert m4a to wav
import subprocess
import os
from pathlib import Path
import io
from PIL import Image

SAMPLE_RATE = 48000  # Define the sample rate for audio processing
WINDOW_SECONDS = 2
STEP_SECONDS = 2
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS  # Calculate the number of samples in the window
STEP_SAMPLES = SAMPLE_RATE * STEP_SECONDS  # Calculate the number of samples to step for each window
print("sample rate: ", SAMPLE_RATE)
print("window samples: ", WINDOW_SAMPLES)
print("step samples: ", STEP_SAMPLES)
def record_live():
    buffer = np.empty((0,), dtype=np.float32)  # Initialize an empty buffer for audio data

    while True:
        audio = sd.rec(STEP_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')

        sd.wait()     # Record audio for the step duration
        audio =  audio[:, 0]  # Convert to mono by taking the first channel
        buffer = np.concatenate((buffer, audio))  # Append the new audio to the buffer

        if len(buffer) >= WINDOW_SAMPLES:
            window = buffer[:WINDOW_SAMPLES]  # Take the first WINDOW_SAMPLES from the buffer
            
            yield window  # Yield the current window for processing
            buffer = buffer[STEP_SAMPLES:]

def plot_spectrogram(audio_data, sample_rate, output_path):
    # Convert stereo/multi-channel audio to mono for spectrogram plotting
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Generate the spectrogram
    # frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)
    frequencies, times, Sxx = spectrogram(
    audio_data,
    fs=sample_rate,
    window='hann',
    nperseg=4096//4,
    noverlap=3840//4, #3072,
    scaling='spectrum',
    mode='magnitude'
                        )
    buffer = io.BytesIO()
    # Plot the spectrogram
    plt.figure(figsize=(20, 6))
    # output_path = os.path.join(
    #     r"D:\DSAI-teaching\brahmansh",
    #     os.path.splitext(os.path.basename(wav_path))[0] + ".png"
    # )

    plt.imsave(
        buffer,
        20 * np.log10(Sxx + 1e-12),
        origin="lower",
        cmap="viridis"      # optional
    )
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return image





for audio in record_live():
    # Here you can process the audio window, e.g., convert it to a spectrogram and feed it to the model
    # For demonstration, we will just print the shape of the audio window
    print(f"Received audio window of shape: {audio.shape}")
    plot_spectrogram(audio, SAMPLE_RATE, "temp_spectrogram.png")
    break