"""Deep Multimodal K-fold Model for Emotion and Sentiment Analysis in Figurative Language.

Package layout
--------------
features.py   193-dimensional handcrafted acoustic feature extraction (MFCC, chroma, mel, contrast, Tonnetz)
text.py       text cleaning, tokenisation, stop-word removal, lemmatisation, integer encoding, GloVe matrix
data.py       dataset loading, SMOTE balancing and the held-out train/test split
models.py     the two architectures used in the paper: 1-D CNN and LSTM
train.py      K-fold cross-validation inside the training set, final model, held-out evaluation
plots.py      confusion matrix and learning-curve figures
"""

__version__ = "1.0.0"

EMOTION_CLASSES = ["happiness", "sadness", "anger", "fear", "disgust"]
SENTIMENT_CLASSES = ["very positive", "positive", "slight positive", "neutral", "negative", "very negative"]
