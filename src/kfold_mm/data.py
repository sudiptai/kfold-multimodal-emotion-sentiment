"""Dataset loading, SMOTE balancing and the held-out split (Sections 3.3, 3.4 and 4 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import EMOTION_CLASSES, SENTIMENT_CLASSES

DATA_FILES = {
    ("audio", "emotion"): "data/audio_features_emotion.csv",
    ("audio", "sentiment"): "data/audio_features_sentiment.csv",
    ("text", "emotion"): "data/text_emotion.csv",
    ("text", "sentiment"): "data/text_sentiment.csv",
}


def _smote(X, y, seed):
    """SMOTE with k_neighbors capped by the smallest class (the 'very negative' sentiment class has only 6 utterances)."""
    k = max(1, min(5, int(np.bincount(y).min()) - 1))
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray   # original CSV row (-1 = synthetic SMOTE sample)
    idx_test: np.ndarray
    class_names: list[str]
    extra: dict             # e.g. word_index / embedding matrix for text


def load_raw(modality: str, task: str, root: str = "."):
    df = pd.read_csv(f"{root}/{DATA_FILES[(modality, task)]}")
    if modality == "audio":
        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].values.astype(int)
    else:
        text_col = "tweet" if "tweet" in df.columns else "text"
        label_col = "class" if "class" in df.columns else "label"
        X = df[text_col].astype(str).values
        y = df[label_col].values.astype(int)
    names = EMOTION_CLASSES if task == "emotion" else SENTIMENT_CLASSES
    n = int(y.max()) + 1
    names = names[:n] if n <= len(names) else [str(i) for i in range(n)]
    return X, y, names


def make_split(modality: str, task: str, root: str = ".", seed: int = 0, test_size: float = 0.2,
               smote_before_split: bool = True, glove_path: str | None = None, max_len: int = 500) -> Split:
    """Reproduce the data protocol of the paper.

    smote_before_split=True  : SMOTE on the whole set, then 20 % held-out test split (protocol of the paper;
                               e.g. audio emotion 1395 -> 4130 -> 826 test, audio sentiment 1395 -> 3666 -> 734 test)
    smote_before_split=False : split the real utterances first and oversample only the training part
                               (leak-free variant discussed in the Limitations paragraph)
    """
    X, y, names = load_raw(modality, task, root)
    extra = {}
    if modality == "text":
        from .text import encode, glove_matrix
        ids, word_index, _ = encode(X, max_len=max_len)
        emb = glove_matrix(word_index, glove_path)
        if emb is None:                       # fallback: random (fixed) embedding so the code still runs
            rng = np.random.RandomState(seed)
            emb = rng.normal(0, 0.3, size=(len(word_index) + 1, 300)).astype("float32"); emb[0] = 0
            extra["glove"] = False
        else:
            extra["glove"] = True
        # each utterance -> (max_len, 300) GloVe sequence (Section 3.2); SMOTE/split operate on these
        # continuous features, flattened to max_len*300 (word indices are NOT interpolated).
        X = emb[ids].reshape(len(ids), -1).astype("float32")
        extra["word_index"] = word_index
        extra["max_len"] = max_len
        extra["embed_dim"] = emb.shape[1]
        del emb

    idx = np.arange(len(X))
    if smote_before_split:
        X_res, y_res = _smote(X, y, seed)
        src = np.r_[idx, np.full(len(X_res) - len(X), -1)]
        X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(X_res, y_res, src, test_size=test_size,
                                                              random_state=seed)
    else:
        X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(X, y, idx, test_size=test_size,
                                                              random_state=seed, stratify=y)
        X_tr, y_tr = _smote(X_tr, y_tr, seed)
        i_tr = np.r_[i_tr, np.full(len(X_tr) - len(i_tr), -1)]

    if modality == "audio":
        scaler = StandardScaler().fit(X_tr)              # fit on training data only
        X_tr = scaler.transform(X_tr).astype("float32")
        X_te = scaler.transform(X_te).astype("float32")
        extra["scaler"] = scaler
    else:
        d = extra["embed_dim"]
        X_tr = X_tr.reshape(len(X_tr), max_len, d).astype("float32")
        X_te = X_te.reshape(len(X_te), max_len, d).astype("float32")

    return Split(X_tr, X_te, np.asarray(y_tr), np.asarray(y_te), i_tr, i_te, names, extra)
