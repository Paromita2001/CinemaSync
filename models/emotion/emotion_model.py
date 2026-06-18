import torch
import torch.nn as nn

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


class EmotionMLP(nn.Module):
    """
    3-layer MLP that classifies 6 emotions from 84-dim audio features.
    Features: 40 MFCC means + 40 MFCC stds + pitch + energy + ZCR + spectral centroid
    """
    def __init__(self, input_dim=84, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, features_tensor):
        self.eval()
        with torch.no_grad():
            logits = self.forward(features_tensor)
            idx = logits.argmax(dim=1).item()
        return IDX2EMOTION[idx]
