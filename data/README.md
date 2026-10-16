# Data files

All files are derived from the CMU-MOSEI utterances selected by majority vote (Section 3.4 of the paper).
Raw audio and video are not redistributed; see the main README for how to rebuild the acoustic features.

| file | rows | columns | label column | classes |
|---|---:|---|---|---|
| `audio_features_emotion.csv`   | 1,395 | `0`…`192` (193 acoustic features), `label` | `label` | 0 happiness · 1 sadness · 2 anger · 3 fear · 4 disgust |
| `audio_features_sentiment.csv` | 1,395 | `0`…`192`, `label` | `label` | 0 very positive · 1 positive · 2 slight positive · 3 neutral · 4 negative · 5 very negative |
| `text_emotion.csv`             | 1,395 | `tweet` (transcript), `class` | `class` | same codes as audio emotion |
| `text_sentiment.csv`           | 1,395 | `text` (transcript), `label` | `label` | same codes as audio sentiment |

Class distributions before SMOTE

* emotion (audio and text): 826 / 357 / 126 / 69 / 17
* sentiment (audio and text): 611 / 445 / 169 / 128 / 36 / 6

The 193 acoustic features are, in order: 40 MFCC, 12 chroma, 128 mel-spectrogram bands, 7 spectral-contrast
bands and 6 Tonnetz dimensions, each averaged over the utterance (`src/kfold_mm/features.py`).

Row order differs between the audio and the text files, so rows cannot be joined across modalities by index.

`split_train.csv` / `split_test.csv` written by every run record which rows of these files form the training
and the held-out test set of that run (`csv_row = -1` marks a synthetic SMOTE sample).
