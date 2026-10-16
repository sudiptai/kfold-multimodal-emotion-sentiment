"""K-fold deep learning model (Algorithm 1 of the paper).

1. the balanced data are divided into a training set and a held-out test set (data.make_split)
2. K-fold cross-validation is run INSIDE the training set with a fresh model per fold
3. a final model is trained on the whole training set
4. the final model is evaluated ONCE on the held-out test set -> classification report + confusion matrix
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from .data import make_split
from .models import build_model, reshape_for
from .plots import plot_confusion_matrix, plot_curves


def _callbacks(fixed_epochs: bool = False):
    if fixed_epochs:
        return []
    return [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)]


def run_experiment(modality: str, task: str, arch: str, *, root: str = ".", out_root: str = "results",
                   folds: int = 5, epochs: int = 20, final_epochs: int = 40, batch_size: int = 32,
                   seed: int = 0, optimizer: str = "adam", lr: float = 1e-3, smote_before_split: bool = True,
                   glove_path: str | None = None, max_len: int = 500, verbose: int = 2,
                   fixed_epochs: bool = False) -> dict:
    np.random.seed(seed); tf.random.set_seed(seed)
    name = f"{modality}_{task}_{arch}"
    out = os.path.join(out_root, name); os.makedirs(out, exist_ok=True)
    t0 = time.time()

    sp = make_split(modality, task, root=root, seed=seed, smote_before_split=smote_before_split,
                    glove_path=glove_path, max_len=max_len)
    n_classes = len(sp.class_names)
    Xtr, Xte = reshape_for(arch, modality, sp.X_train), reshape_for(arch, modality, sp.X_test)
    print(f"[{name}] train {Xtr.shape}  test {Xte.shape}  classes {sp.class_names}")

    pd.DataFrame({"csv_row": sp.idx_train, "label": sp.y_train}).to_csv(f"{out}/split_train.csv", index=False)
    pd.DataFrame({"csv_row": sp.idx_test, "label": sp.y_test}).to_csv(f"{out}/split_test.csv", index=False)

    kw = dict(n_features=sp.X_train.shape[1] if modality == "audio" else 193,
              max_len=sp.extra.get("max_len", max_len), embed_dim=sp.extra.get("embed_dim", 300),
              optimizer=optimizer, lr=lr)

    # ---- K-fold cross-validation inside the training set ----
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_acc, histories = [], []
    for k, (tr, va) in enumerate(kf.split(Xtr, sp.y_train), 1):
        m = build_model(arch, modality, n_classes, **kw)
        h = m.fit(Xtr[tr], sp.y_train[tr], validation_data=(Xtr[va], sp.y_train[va]), epochs=epochs,
                  batch_size=batch_size, verbose=verbose, callbacks=_callbacks(fixed_epochs))
        histories.append(h.history)
        fold_acc.append(float(max(h.history["val_accuracy"])))
        print(f"[{name}] fold {k}/{folds} best val acc = {fold_acc[-1]:.4f}")
        keras.backend.clear_session()

    # ---- final model on the full training set, evaluated once on the held-out split ----
    model = build_model(arch, modality, n_classes, **kw)
    h = model.fit(Xtr, sp.y_train, validation_split=0.1, epochs=final_epochs, batch_size=batch_size,
                  verbose=verbose, callbacks=_callbacks())
    histories.append(h.history)
    model.save(f"{out}/model.keras")

    y_pred = model.predict(Xte, verbose=0).argmax(axis=1)
    acc = accuracy_score(sp.y_test, y_pred)
    report = classification_report(sp.y_test, y_pred, target_names=sp.class_names, digits=2, zero_division=0)
    cm = confusion_matrix(sp.y_test, y_pred, labels=range(n_classes))
    print(report); print(cm)

    with open(f"{out}/classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(f"{out}/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, sp.class_names, f"{out}/confusion_matrix.png")
    plot_curves(histories, f"{out}/curves")

    summary = {"experiment": name, "modality": modality, "task": task, "arch": arch,
               "n_train": int(len(sp.y_train)), "n_test": int(len(sp.y_test)), "classes": sp.class_names,
               "kfold_val_acc": fold_acc, "kfold_val_acc_mean": float(np.mean(fold_acc)),
               "kfold_val_acc_std": float(np.std(fold_acc)), "heldout_test_acc": float(acc),
               "correct": int((sp.y_test == y_pred).sum()), "smote_before_split": smote_before_split,
               "optimizer": optimizer, "lr": lr, "epochs_per_fold": epochs, "final_epochs": final_epochs, "fixed_epochs": fixed_epochs,
               "glove": sp.extra.get("glove") if modality == "text" else None,
               "seconds": round(time.time() - t0, 1)}
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] held-out accuracy {acc:.4f} ({summary['correct']}/{summary['n_test']})  "
          f"K-fold {summary['kfold_val_acc_mean']:.4f} +/- {summary['kfold_val_acc_std']:.4f}")
    return summary


def collect_results(out_root: str = "results") -> pd.DataFrame:
    rows = []
    for d in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, d, "summary.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            rows.append({"modality": s["modality"], "task": s["task"], "model": s["arch"].upper(),
                         "K-fold val acc (mean ± std)": f"{s['kfold_val_acc_mean']:.3f} ± {s['kfold_val_acc_std']:.3f}",
                         "held-out test acc": f"{s['heldout_test_acc']:.3f}", "n_test": s["n_test"]})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(os.path.join(out_root, "results_table.csv"), index=False)
        with open(os.path.join(out_root, "results_table.md"), "w") as f:
            f.write(df.to_markdown(index=False))
    return df
