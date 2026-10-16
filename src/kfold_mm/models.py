"""The two architectures used in the paper (Sections 3.6 and 3.7), shared by both modalities.

CNN  : Conv1D(256,k5) -> Conv1D(128,k5) -> Dropout(0.1) -> MaxPool(8)
       -> 3 x Conv1D(128,k5) -> Dropout(0.2) -> Conv1D(128,k5) -> Flatten -> Dropout(0.2) -> Dense SoftMax
LSTM : LSTM(512) -> Dropout(0.2) -> Dense 512 -> 256 -> 128 -> 64 (ReLU, Dropout 0.2) -> SoftMax

For audio the input is the 193-dimensional standardised feature vector:
    CNN  sees it as a sequence of 193 scalars   (193, 1)  -- convolution along the feature axis
    LSTM sees it as one 193-channel time step    (1, 193)  -- see README, "Notes on the LSTM input"
For text the input is the (max_len, 300) sequence of pre-trained GloVe vectors of the cleaned utterance.
"""
from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def _optimizer(name: str, lr: float):
    if name.lower() == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
    return keras.optimizers.Adam(learning_rate=lr)


def _cnn_stack(x, n_classes: int):
    """1-D CNN of the paper (audio: 193 features treated as a 193-step sequence)."""
    x = layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling1D(pool_size=8)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def _lstm_stack(x, n_classes: int, units: int = 512, dropout: float = 0.2):
    x = layers.LSTM(units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def build_model(arch: str, modality: str, n_classes: int, *, n_features: int = 193, max_len: int = 500,
                embed_dim: int = 300,
                optimizer: str = "adam", lr: float = 1e-3) -> keras.Model:
    arch, modality = arch.lower(), modality.lower()
    if modality == "audio":
        if arch == "cnn":
            inp = layers.Input(shape=(n_features, 1))
            out = _cnn_stack(inp, n_classes)
        else:
            inp = layers.Input(shape=(1, n_features))
            out = _lstm_stack(inp, n_classes)
    else:
        inp = layers.Input(shape=(max_len, embed_dim))   # (500, 300) GloVe sequence, Section 3.2
        out = _cnn_stack(inp, n_classes) if arch == "cnn" else _lstm_stack(inp, n_classes)
    model = keras.Model(inp, out, name=f"{modality}_{arch}")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=_optimizer(optimizer, lr),
                  metrics=["accuracy"])
    return model


def reshape_for(arch: str, modality: str, X: np.ndarray) -> np.ndarray:
    """Give the feature matrix the input shape the model expects."""
    if modality == "text":
        return X                                 # already (n, max_len, 300)
    return X.reshape(-1, X.shape[1], 1) if arch == "cnn" else X.reshape(-1, 1, X.shape[1])
