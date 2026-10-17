# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Caption fields of Stage 1 (train_jax/alps_jax/data/tokenize.py), with the pure-Python ALPS tokenizer.

Every field of length L is  [bos] w_1 .. w_k [eos] [PAD] .. [PAD] [CLS]: [bos] + WordPiece ids + [eos], truncated
or right-padded with 0 to L - 1 ids (the last forced to [eos] when the caption fills the field), then [CLS] in slot
L - 1 (also in the caption-decoder field, where it is a prediction target).

    labels       [80]  the synthetic caption (one element at random if it is a list)   -> caption decoder
    itc_labels1  [32]  the web caption (the first element if it is a list)             -> contrastive positive 1
    itc_labels2  [32]  one sentence of the synthetic caption, uniformly at random      -> contrastive positive 2

The ALPS runs tokenized with TensorFlow Text's BertTokenizer; `alps.tokenizer.ALPSTokenizer` re-implements it
(its test against TF Text lives in the inference package's tests).
"""
from __future__ import annotations

import re
from typing import List, Sequence, Union

import numpy as np

from alps.tokenizer import DEFAULT_VOCAB, ALPSTokenizer

_ASCII_WS = " \t\n\v\f\r"
_SPLIT = re.compile(r"[.!]+")


def _as_list(x: Union[str, bytes, Sequence]) -> List[str]:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, str):
        return [x]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in x]


def random_element(x, rng: np.random.Generator) -> str:
    """A string or a list of strings -> one of them uniformly at random ('' for an empty list)."""
    xs = _as_list(x) + [""]
    n = max(len(xs) - 1, 1)
    return xs[int(rng.integers(0, n))]


def sentence_candidates(dense) -> List[str]:
    """All parts joined by ' ', '[.!]+' replaced by '|', split on '|', ASCII-whitespace stripped, empties dropped."""
    text = " ".join(_as_list(dense))
    parts = [p.strip(_ASCII_WS) for p in _SPLIT.sub("|", text).split("|")]
    return [p for p in parts if len(p.encode("utf-8")) > 0]


def random_sentence(dense, rng: np.random.Generator) -> str:
    """One sentence of the synthetic caption uniformly at random; the whole joined text if there is none."""
    parts = sentence_candidates(dense)
    if not parts:
        return " ".join(_as_list(dense))
    return parts[int(rng.integers(0, len(parts)))]


class Stage1Text:
    """Builds the three token fields of one example."""

    def __init__(self, vocab_path: str = DEFAULT_VOCAB, itc_len: int = 32, caption_len: int = 80):
        self.itc = ALPSTokenizer(vocab_path, context_length=itc_len)
        self.cap = ALPSTokenizer(vocab_path, context_length=caption_len)

    def __call__(self, web_caption, synthetic_caption, rng: np.random.Generator):
        web = _as_list(web_caption)
        if not web:
            raise ValueError("empty web caption")
        return {"labels": np.asarray(self.cap.encode(random_element(synthetic_caption, rng)), np.int32),
                "itc_labels1": np.asarray(self.itc.encode(web[0]), np.int32),
                "itc_labels2": np.asarray(self.itc.encode(random_sentence(synthetic_caption, rng)), np.int32)}
