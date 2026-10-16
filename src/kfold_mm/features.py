"""193-dimensional handcrafted acoustic features (Section 3.1 of the paper).

    40 MFCC  + 12 chroma + 128 mel-spectrogram + 7 spectral contrast + 6 Tonnetz = 193

Each descriptor is averaged over time, which gives one fixed-length vector per utterance.
Only needed if you start from raw .wav files; the CSVs in ``data/`` already contain the vectors.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def extract_features(path: str, sr: int | None = 22050) -> np.ndarray:
    import librosa  # imported lazily so the rest of the package works without librosa

    y, sr = librosa.load(path, sr=sr)
    stft = np.abs(librosa.stft(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)              # 40
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)             # 12
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)                # 128
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)     # 7
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)  # 6
    return np.hstack([mfcc, chroma, mel, contrast, tonnetz]).astype("float32")         # 193


def build_feature_table(wav_paths: Iterable[str], labels: Iterable[int], out_csv: str) -> pd.DataFrame:
    """Extract features for every file and save them in the same layout as ``data/audio_features_*.csv``
    (columns ``0..192`` followed by ``label``)."""
    rows = []
    for p, lab in zip(wav_paths, labels):
        rows.append(np.r_[extract_features(p), lab])
    df = pd.DataFrame(rows, columns=[str(i) for i in range(193)] + ["label"])
    df["label"] = df["label"].astype(int)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
