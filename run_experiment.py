#!/usr/bin/env python3
"""Run one (or all) of the eight modality x task x architecture experiments of the paper.

Examples
    python run_experiment.py --modality audio --task emotion --model lstm
    python run_experiment.py --all --glove data/glove.840B.300d.txt
    python run_experiment.py --all --epochs 2 --final-epochs 2        # quick smoke test
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from kfold_mm.train import collect_results, run_experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["audio", "text"])
    p.add_argument("--task", choices=["emotion", "sentiment"])
    p.add_argument("--model", choices=["cnn", "lstm"])
    p.add_argument("--all", action="store_true", help="run all eight combinations")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20, help="epochs per fold")
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--glove", default=os.environ.get("GLOVE_PATH"), help="path to glove.840B.300d.txt")
    p.add_argument("--max-len", type=int, default=500)
    p.add_argument("--leak-free", action="store_true",
                   help="split the real utterances first and apply SMOTE to the training part only")
    p.add_argument("--fixed-epochs", action="store_true",
                   help="train for exactly --epochs / --final-epochs (no early stopping, no LR reduction)")
    p.add_argument("--out", default="results")
    p.add_argument("--verbose", type=int, default=2)
    a = p.parse_args()

    if a.all:
        combos = itertools.product(["audio", "text"], ["emotion", "sentiment"], ["cnn", "lstm"])
    elif a.modality and a.task and a.model:
        combos = [(a.modality, a.task, a.model)]
    else:
        p.error("give --modality, --task and --model, or --all")

    for modality, task, arch in combos:
        run_experiment(modality, task, arch, out_root=a.out, folds=a.folds, epochs=a.epochs,
                       final_epochs=a.final_epochs, batch_size=a.batch_size, seed=a.seed, optimizer=a.optimizer,
                       lr=a.lr, smote_before_split=not a.leak_free, glove_path=a.glove, max_len=a.max_len,
                       verbose=a.verbose, fixed_epochs=a.fixed_epochs)
    print("\n" + collect_results(a.out).to_string(index=False))


if __name__ == "__main__":
    main()
