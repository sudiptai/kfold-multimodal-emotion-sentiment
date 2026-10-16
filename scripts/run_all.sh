#!/usr/bin/env bash
# Reproduces all eight experiments of the paper (Tables 2-10). Set GLOVE_PATH to use pre-trained GloVe.
set -e
cd "$(dirname "$0")/.."
python run_experiment.py --all --glove "${GLOVE_PATH:-data/glove.840B.300d.txt}" "$@"
