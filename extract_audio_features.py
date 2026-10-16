#!/usr/bin/env python3
"""Extract the 193-d acoustic feature vectors from raw .wav files (only needed to rebuild data/audio_features_*.csv).

    python extract_audio_features.py --manifest my_files.csv --out data/audio_features_emotion.csv

``my_files.csv`` must have the columns ``path`` (wav file) and ``label`` (integer class).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from kfold_mm.features import build_feature_table  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = pd.read_csv(a.manifest)
    build_feature_table(m["path"], m["label"], a.out)
    print("wrote", a.out)
