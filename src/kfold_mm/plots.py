"""Figures: confusion matrix (Figs. 6-13 of the paper) and learning curves."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_png: str, title: str = "Confusion matrix"):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label", fontsize=9); ax.set_xlabel("Predicted label", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_curves(histories: list[dict], out_prefix: str):
    for key, name in (("loss", "loss"), ("accuracy", "accuracy")):
        tr = sum([h[key] for h in histories], [])
        va = sum([h["val_" + key] for h in histories], [])
        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
        ax.plot(tr, label="train"); ax.plot(va, label="validation")
        ax.set_title(f"model {name}"); ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(f"{out_prefix}_{name}.png"); plt.close(fig)
