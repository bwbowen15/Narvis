import os
import numpy as np


# 1. imports
# 2. data loading
# 3. model def
# 4. training Loop
# 5. save model periodically
# 6. posibly eval / prediction

folder = "spectrograms"

def load_spectrograms(FLDR: str):
    #convert letters to numbers then back to letters
    #note for me. this is because it cant think with letters it uses numbers mapped to the letters to learn then translates it back to letters to show me the words
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for i,c in enumerate(chars)}


    #load all the spectros
    spectrograms = []
    labels = []

    for root, dirs, files in os.walk(FLDR):
        for file in files:
            if file.endswith(".npz"):
                path = os.path.join(root, file)
                data = np.load(path)
                spec = data["spectrogram"]
                label_chars = data["label"]

                # convert to numbers
                label_ints = [char2idx[c] for c in label_chars]
                spectrograms.append(spec)
                labels.append(label_ints)

    max_time = max(spec.shape[1] for spec in spectrograms)
    padded_specs = []
    for spec in spectrograms:
        pad_width = max_time - spec.shape[1]
        padded_spec = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
        padded_specs.append(padded_spec)

    # For labels: pad with a special index (here we use len(chars) as padding index)
    max_label_len = max(len(lbl) for lbl in labels)
    padded_labels = []
    for lbl in labels:
        pad_width = max_label_len - len(lbl)
        padded_lbl = lbl + [len(chars)]*pad_width
        padded_labels.append(padded_lbl)

    # Convert to numpy arrays
    X = np.array(padded_specs)  # Shape: (num_samples, freq_bins, time_steps)
    y = np.array(padded_labels) # Shape: (num_samples, max_label_len)