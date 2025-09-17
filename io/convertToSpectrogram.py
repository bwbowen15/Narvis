import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import time




def wav_to_spectogram(wav_file: str):
    sample_rate, data = wavfile.read(wav_file)

    # check if its mono or sterio and only use one channel
    if len(data.shape) > 1:
        data = data[:,0]

    frequencies, times, Sxx = spectrogram(data, fs=sample_rate)

    print("spectogram matrix")
    print(Sxx)

    timestamp = int(time.time() * 1000)
    save_path = f"./io/spectrogram_{timestamp}.npy"
    np.save(save_path, Sxx)
    print(save_path)



wav_to_spectogram("C:/Users/Brian Bowen/Narvis/io/recording.wav")
