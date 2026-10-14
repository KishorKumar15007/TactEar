"""Inference architecture compatible with official PANNs Cnn14_16k weights.

Architecture verified against qiuqiangkong/audioset_tagging_cnn models.py,
Cnn14_16k. No training augmentation or network acquisition runs here.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.bn2 = nn.BatchNorm2d(outgoing)
    def forward(self, x, pool):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool)

class CNN14(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=512, hop_length=160,
            win_length=512, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=16000, n_fft=512,
            n_mels=64, fmin=50, fmax=8000, ref=1., amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6): setattr(self, f"conv_block{i+1}", Block(widths[i], widths[i+1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)
    def forward(self, waveform):
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, (2, 2) if i<6 else (1, 1))
        x = x.mean(dim=3)
        x = x.amax(dim=2) + x.mean(dim=2)
        embedding = F.relu(self.fc1(x))
        return {"clipwise_output": torch.sigmoid(self.fc_audioset(embedding)), "embedding": embedding}
