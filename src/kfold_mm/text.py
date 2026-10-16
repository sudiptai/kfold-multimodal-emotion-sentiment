"""Textual feature extraction (Section 3.2 of the paper).

clean -> tokenise -> remove punctuation and stop words -> lemmatise -> words to integers
-> pad to MAX_LEN (500) -> 300-d pre-trained GloVe embedding matrix.
"""
from __future__ import annotations

import os
import re
import string
from typing import Iterable

import numpy as np

MAX_LEN = 500          # maximum sequence length used in the paper
EMBED_DIM = 300        # GloVe 300-d vectors

_EXTRA = {"''", "``", "rt", "https", "’", "“", "”", "​", "--", "n't", "'s", "...", "//t.c"}

_FALLBACK_STOPS = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own s same she should so some such t than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with you your yours yourself yourselves""".split())

_NLTK_OK: bool | None = None


def _nltk_ready() -> bool:
    """Make sure the NLTK resources are present; return False (-> regex fallback) if they cannot be obtained."""
    global _NLTK_OK
    if _NLTK_OK is not None:
        return _NLTK_OK
    try:
        import nltk
    except ImportError:
        _NLTK_OK = False
        return False
    ok = True
    for res, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                nltk.data.find(path)
            except Exception:
                ok = False
    if not ok:
        print("WARNING: NLTK data not available - using regex tokeniser and built-in stop list (no lemmatisation). "
              "Run  python -c \"import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]\"")
    _NLTK_OK = ok
    return ok


def clean_tokens(sentence: str) -> list[str]:
    """Tokenise, lower-case, drop punctuation and stop words, lemmatise (as in the paper).

    Falls back to a regex tokeniser + built-in stop list (no lemmatisation) when NLTK data is unavailable.
    """
    if _nltk_ready():
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        words = word_tokenize(str(sentence))
        words = [w.lower() for w in words if w not in set(string.punctuation)]
        stops = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        words = [lem.lemmatize(w) for w in words if w not in stops and w not in _EXTRA]
    else:
        words = [w for w in re.findall(r"[a-zA-Z@']+", str(sentence).lower())
                 if w not in _FALLBACK_STOPS and w not in _EXTRA]
    return [w for w in words if re.fullmatch(r"[a-zA-Z@]+", w)]


def encode(sentences: Iterable[str], max_len: int = MAX_LEN):
    """Return (padded integer matrix, word_index, tokenizer)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokens = [clean_tokens(s) for s in sentences]
    tok = Tokenizer()
    tok.fit_on_texts(tokens)
    seqs = tok.texts_to_sequences(tokens)
    X = pad_sequences(seqs, maxlen=max_len)
    return X, tok.word_index, tok


def glove_matrix(word_index: dict, glove_path: str | None, dim: int = EMBED_DIM) -> np.ndarray | None:
    """Build the (vocab_size, 300) embedding matrix from a GloVe text file.

    Returns None when ``glove_path`` is missing, in which case the models fall back to a
    trainable randomly-initialised embedding layer (documented in the README).
    """
    if not glove_path or not os.path.exists(glove_path):
        if glove_path:
            print(f"WARNING: GloVe file not found at {glove_path} - using a trainable random embedding instead")
        return None
    vocab = len(word_index) + 1
    matrix = np.zeros((vocab, dim), dtype="float32")
    hits = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word, vec = parts[0], parts[1:]
            if word in word_index and len(vec) == dim:
                matrix[word_index[word]] = np.asarray(vec, dtype="float32")
                hits += 1
    print(f"GloVe: {hits}/{vocab - 1} vocabulary words found in {os.path.basename(glove_path)}")
    return matrix
