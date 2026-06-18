import os
import numpy as np
import torch
import librosa

from models.emotion.emotion_model import EmotionMLP, IDX2EMOTION

MODEL_PATH = "models/emotion/emotion_model.pt"


class EmotionDetector:
    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Emotion model not found at {MODEL_PATH}. "
                "Run training/train_emotion.py first."
            )
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        self.model = EmotionMLP()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        self.mean = checkpoint['norm_mean']
        self.std  = checkpoint['norm_std']

    def _extract(self, audio_path, start, end):
        y, sr = librosa.load(audio_path, sr=22050,
                              offset=float(start),
                              duration=float(end - start))
        if len(y) < sr * 0.1:
            return None

        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        energy  = librosa.feature.rms(y=y)
        zcr     = librosa.feature.zero_crossing_rate(y)
        spec    = librosa.feature.spectral_centroid(y=y, sr=sr)
        pitch_vals = pitch[pitch > 0]

        feats = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            [pitch_vals.mean() if len(pitch_vals) > 0 else 0.0],
            [energy.mean()], [zcr.mean()], [spec.mean()]
        ])
        return feats.astype(np.float32)

    def detect(self, audio_path, start, end):
        feats = self._extract(audio_path, start, end)
        if feats is None:
            return 'neutral'
        x = torch.tensor(feats).unsqueeze(0)
        x = (x - self.mean) / self.std.clamp(min=1e-8)
        return self.model.predict(x)
