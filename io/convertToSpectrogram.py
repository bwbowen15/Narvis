import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import time
import os

def wav_to_spectrogram(data, sample_rate):
    frequencies, times, Sxx = spectrogram(data, fs=sample_rate)
    return Sxx


# def wav_to_spectogram(wav_file: str):
#     # sample_rate, data = wavfile.read(wav_file)

#     # # check if its mono or sterio and only use one channel
#     # if len(data.shape) > 1:
#     #     data = data[:,0]

#     frequencies, times, Sxx = spectrogram(data, fs=sample_rate)

#     # print("spectogram matrix")
#     # print(Sxx)

#     # timestamp = int(time.time() * 1000)
#     # save_path = f"./io/spectrogram_{timestamp}.npy"
#     # np.save(save_path, Sxx)
#     # print(save_path)

#change folder name based on the word you are training
def process_folder(folder_path: str, label: str, save_dir= "spectrograms/zero"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for wav_file in os.listdir(folder_path):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(folder_path, wav_file)
        sample_rate, data = wavfile.read(wav_path)

        if len(data.shape) > 1:
            data = data[:,0]
        
        Sxx = wav_to_spectrogram(data, sample_rate)

        char_label = list(label.lower())

        #save with label
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(save_dir, f"{label}_{timestamp}.npz")
        np.savez_compressed(save_path, spectrogram=Sxx, label=char_label)




#change based on the word
folder_path = "C:/Users/Brian Bowen/Narvis/io/speech_commands_v0.02/zero"
label = "zero"
process_folder(folder_path,label)
print("done")
