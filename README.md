# Deep Multimodal K-fold Model for Emotion and Sentiment Analysis in Figurative Language

Code and data accompanying the paper

> *Deep Multimodal K-fold Model for Emotion and Sentiment Analysis in Figurative Language*,
> submitted to the **International Journal of Speech Technology** (Springer), 2026.

The framework trains eight single-task deep models on a 1,354-utterance majority-vote subset of
**CMU-MOSEI**: two modalities (audio, text) × two tasks (emotion, sentiment) × two architectures (1-D CNN, LSTM),
each evaluated with K-fold cross-validation inside the training set and once on a held-out test split
(Algorithm 1 and Fig. 5 of the paper).

```
A  audio feature extraction (193-d)  ─┐      ┌─ C  CNN  ─┐      ┌─ E  emotion
                                      ├──────┤           ├──────┤
B  text feature extraction (GloVe)   ─┘      └─ D  LSTM ─┘      └─ F  sentiment
```

## Repository layout

| Path | Content |
|---|---|
| `run_experiment.py` | command-line entry point: one experiment or all eight |
| `src/kfold_mm/features.py` | 193-d acoustic features: 40 MFCC + 12 chroma + 128 mel + 7 contrast + 6 Tonnetz (Sec. 3.1) |
| `src/kfold_mm/text.py` | cleaning, tokenisation, stop-word removal, lemmatisation, integer encoding, GloVe-300d lookup; each utterance becomes a (500, 300) GloVe sequence (Sec. 3.2) |
| `src/kfold_mm/data.py` | SMOTE balancing and the 80/20 held-out split (Sec. 3.3–3.4, Sec. 4) |
| `src/kfold_mm/models.py` | the CNN (Conv1D 256/128 → pool 8 → 4×Conv1D 128 → Flatten → SoftMax) and LSTM (512 → 512/256/128/64 → SoftMax) architectures (Sec. 3.6–3.7) |
| `src/kfold_mm/train.py` | K-fold cross-validation, final model, held-out evaluation, reports and figures (Sec. 3.8, Algorithm 1) |
| `src/kfold_mm/plots.py` | confusion-matrix and learning-curve figures (Figs. 6–13) |
| `extract_audio_features.py` | rebuild the audio feature CSVs from `.wav` files |
| `data/` | the four feature/label files used in the paper (see `data/README.md`) |
| `notebooks/kfold_lstm_audio_emotion.ipynb` | Google-Colab notebook for the audio-emotion LSTM experiment (Table 3 / Fig. 7) |
| `results/` | one sub-folder per experiment is written here when you run the code |
| `scripts/` | `download_glove.sh`, `run_all.sh` |

## Installation

```bash
git clone https://github.com/<user>/kfold-multimodal-emotion-sentiment.git
cd kfold-multimodal-emotion-sentiment
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
bash scripts/download_glove.sh        # glove.840B.300d.txt (~5.6 GB) – optional, see "GloVe" below
```

Tested with Python 3.10–3.12 and TensorFlow 2.15–2.21. A GPU is not required; on Google Colab (T4) a single
experiment takes a few minutes.

## Running the experiments

```bash
# one experiment: <modality> x <task> x <model>
python run_experiment.py --modality audio --task emotion   --model lstm
python run_experiment.py --modality text  --task sentiment --model cnn --glove data/glove.840B.300d.txt

# all eight experiments of the paper (Tables 2–10)
bash scripts/run_all.sh

# quick smoke test (~2 min on CPU)
python run_experiment.py --all --folds 2 --epochs 1 --final-epochs 1
```

Each run writes to `results/<modality>_<task>_<model>/`:

| file | content |
|---|---|
| `classification_report.txt` | per-class precision / recall / F1 / support on the held-out split (the format of Tables 2–9) |
| `confusion_matrix.csv`, `confusion_matrix.png` | held-out confusion matrix (Figs. 6–13) |
| `curves_loss.png`, `curves_accuracy.png` | learning curves (folds + final model) |
| `split_train.csv`, `split_test.csv` | row index of every sample in the original CSV (`-1` = synthetic SMOTE sample) |
| `summary.json` | per-fold validation accuracies (mean ± std), held-out accuracy, configuration |
| `model.keras` | the final trained model |

After a run, `results/results_table.md` gathers all experiments in one table.

Useful options: `--folds` (default 5), `--epochs` per fold (20), `--final-epochs` (40), `--optimizer adam|rmsprop`,
`--lr`, `--seed`, `--leak-free` (see *Evaluation protocol*), `--max-len` (500).

## Evaluation protocol

Exactly as stated in Section 4 of the paper:

1. the data are balanced with SMOTE (minority oversampling) and divided into a training set and a **held-out test set (20 %)**;
2. **K-fold cross-validation** (K = 5 by default) is run inside the training set with a fresh model per fold —
   the per-fold validation accuracies are reported as mean ± std in `summary.json`;
3. a final model is trained on the whole training set and evaluated **once** on the held-out split.

Every accuracy, precision, recall and F1 value in the paper is a held-out-split value, not a fold average.

With the paper's protocol SMOTE is applied before the split, which gives the test-set sizes used in the paper
(audio/text emotion: 5 × 826 = 4,130 samples → 826 test; audio sentiment: 6 × 611 = 3,666 → 734 test).
Because synthetic samples are interpolations of real ones, this protocol can inflate accuracy — a limitation
acknowledged in the paper. `--leak-free` splits the *real* utterances first and oversamples the training part only;
use it to obtain a conservative estimate.

## Results reported in the paper

Held-out test accuracy (%), Table 10 of the paper. These are the numbers reported in the manuscript; a new run
of this code produces its own `results/results_table.md`, which will differ slightly because of random
initialisation and library versions.

| Model | CNN | LSTM |
|---|---:|---:|
| K-fold deep learning **text emotion** | 85 | 99 |
| K-fold deep learning **audio emotion** | 85 | 88 |
| K-fold deep learning **text sentiment** | 94 | 93 |
| K-fold deep learning **audio sentiment** | 79 | 78 |

Per-class reports (Tables 2–9) and confusion matrices (Figs. 6–13) are in the paper.

`results/` ships the output of the four **audio** experiments produced with this code (seed 0, one shared held-out
split per task; reports, confusion matrices, `summary.json`, split files). The audio-emotion runs are the ones used
for Tables 2–3 / Figs. 6–7 of the revised paper:

| experiment | K-fold val acc (mean ± std) | held-out acc | paper |
|---|---:|---:|---:|
| audio emotion CNN   | 0.874 ± 0.012 | 0.85 (701/826) | 85 |
| audio emotion LSTM  | 0.871 ± 0.009 | 0.88 (725/826) | 88 |
| audio sentiment CNN | 0.771 ± 0.013 | 0.74 (540/734) | 79 |
| audio sentiment LSTM| 0.773 ± 0.011 | 0.75 (553/734) | 78 |

The four text experiments require the 5.6 GB GloVe file; `notebooks/run_text_experiments_colab.ipynb` runs them on
Google Colab in ~20 min, and `bash scripts/run_all.sh` regenerates all eight locally (`results/results_table.md`).

## Data

`data/` contains the feature/label files for the 1,354 majority-vote CMU-MOSEI utterances used in the paper
(see `data/README.md` for the label codes). The raw audio/video is **not** redistributed; download CMU-MOSEI from the
[CMU MultiComp Lab](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) and use `extract_audio_features.py`
to rebuild the acoustic feature files.

## GloVe and the text representation

Following Section 3.2, every cleaned utterance is converted into a (500, 300) matrix of pre-trained
`glove.840B.300d` vectors **before** SMOTE and the split; SMOTE therefore interpolates continuous GloVe sequences
(150,000-d), never word indices, and the CNN/LSTM consume the (500, 300) sequences directly. This needs about
6–8 GB of RAM for the emotion task (fine on Colab); use `--max-len 300` on smaller machines.

If the GloVe file is not supplied (`--glove` or `GLOVE_PATH`), a fixed random embedding table is used so the code
still runs end to end, but this is *not* the configuration of the paper; `summary.json` records `"glove": false`.

## Notes on the LSTM input

The 193 acoustic descriptors are not a time series. The LSTM therefore receives the standardised feature
vector as a single 193-channel step, `(1, 193)`; feeding it as 193 scalar steps `(193, 1)` trains ~30× slower and
stays near chance with the paper's optimiser settings. The CNN convolves along the feature axis, `(193, 1)`,
as described in Section 3.6. Features are standardised with statistics of the training split only.

## Citation

```bibtex
@article{kfold_multimodal_2026,
  title   = {Deep Multimodal K-fold Model for Emotion and Sentiment Analysis in Figurative Language},
  journal = {International Journal of Speech Technology},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT — see `LICENSE`. CMU-MOSEI is subject to its own terms of use.
