# model.py
import torch
import torch.nn as nn

class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_chars):
        super(SpeechRecognitionModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))

        # Recurrent layer
        self.gru = nn.GRU(
            input_size=64,       
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected output layer (+1 for CTC blank token)
        self.fc = nn.Linear(128*2, num_chars+1)

    def forward(self, x):
        # x: (batch, freq_bins, time_steps)
        x = x.unsqueeze(1)  # add channel dim → (batch, 1, freq, time)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))

        # reshape for GRU: collapse freq dimension
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1*f)  # (batch, time, features)

        x, _ = self.gru(x)  # GRU output
        x = self.fc(x)       # (batch, time, num_chars+1)

        return x
