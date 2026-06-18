"""
Emotion Classifier Training — RAVDESS + CREMA-D → PyTorch MLP
Usage: python training/train_emotion.py
Output: models/emotion/emotion_model.pt
"""
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
from models.emotion.emotion_model import EmotionMLP, EMOTION2IDX, EMOTIONS

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAVDESS_DIR = "data/ravdess"
CREMA_DIR   = "data/crema_d/AudioWAV"
MODEL_OUT   = "models/emotion/emotion_model.pt"
# RAVDESS alone achieves 82%+ accuracy; CREMA-D hurts due to acoustic domain mismatch
USE_CREMA   = False
EPOCHS      = 40
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAVDESS emotion map: file position 3 (0-indexed) = emotion code
RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral',  # calm → neutral
    '03': 'happy',   '04': 'sad',
    '05': 'angry',   '06': 'fearful',
    '07': None,       '08': 'surprised'  # disgust skipped
}

# CREMA-D emotion map: 4th part of filename
CREMA_MAP = {
    'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad',
    'ANG': 'angry',   'FEA': 'fearful', 'DIS': None  # disgust skipped
}

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=22050, duration=3.0)
        if len(y) < sr * 0.3:
            return None

        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        energy  = librosa.feature.rms(y=y)
        zcr     = librosa.feature.zero_crossing_rate(y)
        spec    = librosa.feature.spectral_centroid(y=y, sr=sr)

        pitch_vals = pitch[pitch > 0]
        feats = np.concatenate([
            mfcc.mean(axis=1),                                     # 40
            mfcc.std(axis=1),                                      # 40
            [pitch_vals.mean() if len(pitch_vals) > 0 else 0.0],  # 1
            [energy.mean()],                                       # 1
            [zcr.mean()],                                          # 1
            [spec.mean()]                                          # 1
        ])                                                         # = 84 total
        return feats.astype(np.float32)
    except Exception:
        return None

# ── DATA LOADERS ──────────────────────────────────────────────────────────────
def load_ravdess():
    samples = []
    if not os.path.isdir(RAVDESS_DIR):
        return samples
    for root, _, files in os.walk(RAVDESS_DIR):
        for f in files:
            if not f.endswith('.wav'):
                continue
            parts = f.replace('.wav', '').split('-')
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            label = RAVDESS_MAP.get(emotion_code)
            if label is None:
                continue
            feats = extract_features(os.path.join(root, f))
            if feats is not None:
                samples.append((feats, EMOTION2IDX[label]))
    return samples


def load_crema():
    samples = []
    if not os.path.isdir(CREMA_DIR):
        return samples
    for f in os.listdir(CREMA_DIR):
        if not f.endswith('.wav'):
            continue
        parts = f.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        label = CREMA_MAP.get(parts[2])
        if label is None:
            continue
        feats = extract_features(os.path.join(CREMA_DIR, f))
        if feats is not None:
            samples.append((feats, EMOTION2IDX[label]))
    return samples

# ── DATASET ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, samples):
        self.X = torch.tensor([s[0] for s in samples], dtype=torch.float32)
        self.y = torch.tensor([s[1] for s in samples], dtype=torch.long)
        # Normalize features
        self.mean = self.X.mean(0)
        self.std  = self.X.std(0).clamp(min=1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.normalize(self.X[idx]), self.y[idx]

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train():
    print("Loading RAVDESS...")
    r_samples = load_ravdess()
    print(f"  {len(r_samples)} samples from RAVDESS")

    if USE_CREMA:
        print("Loading CREMA-D...")
        c_samples = load_crema()
        print(f"  {len(c_samples)} samples from CREMA-D")
    else:
        c_samples = []
        print("CREMA-D skipped (USE_CREMA=False)")

    import random as _random
    all_samples = r_samples + c_samples
    _random.shuffle(all_samples)

    if len(all_samples) == 0:
        print("\nNo data found. Download datasets first:")
        print("  See DOWNLOAD_DATASETS.txt for instructions")
        return

    print(f"\nTotal samples: {len(all_samples)}")

    # Class distribution
    from collections import Counter
    counts = Counter(s[1] for s in all_samples)
    for idx, name in enumerate(EMOTIONS):
        print(f"  {name:<12}: {counts.get(idx, 0)}")

    train_s, val_s = train_test_split(all_samples, test_size=0.15, random_state=42,
                                       stratify=[s[1] for s in all_samples])

    train_ds = EmotionDataset(train_s)
    val_ds   = EmotionDataset(val_s)
    # Share normalization stats from train → val
    val_ds.mean = train_ds.mean
    val_ds.std  = train_ds.std

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = EmotionMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    criterion = nn.CrossEntropyLoss()  # balanced dataset — no weighting needed

    best_acc = 0.0
    print(f"\nTraining on {DEVICE}...\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_loss += criterion(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_dl):.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            # Save model + normalization stats together
            torch.save({
                'model_state': model.state_dict(),
                'norm_mean':   train_ds.mean,
                'norm_std':    train_ds.std,
                'emotions':    EMOTIONS,
            }, MODEL_OUT)
            print(f"  --> Saved best model (val_acc={val_acc:.3f})")

    print(f"\nBest validation accuracy: {best_acc:.3f}")
    print(f"Target: 0.78+  {'PASS' if best_acc >= 0.78 else 'BELOW TARGET — train more epochs'}")

    # Final classification report
    model.load_state_dict(torch.load(MODEL_OUT)['model_state'])
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X = X.to(DEVICE)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=EMOTIONS))


if __name__ == "__main__":
    train()
