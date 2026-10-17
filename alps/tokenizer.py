# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer of the ALPS text encoder.

ALPS was trained with TensorFlow Text's `BertTokenizer(vocab, lower_case=True)` on the BERT-base
uncased WordPiece vocabulary, in which the two unused slots 1 and 2 were renamed [bos] and [eos]
(`assets/bert_base_vocab_bos_eos.txt`). Every text is laid out as

    [bos] w_1 ... w_k [eos] [PAD] ... [PAD] [CLS]          (32 ids in total)

i.e. [bos] + WordPiece ids + [eos], padded with 0 (or truncated, keeping [eos] last) to 31 ids,
followed by a [CLS] summary token in the last slot, which the text encoder reads.

This module is a dependency-free re-implementation of that tokenizer (basic splitting + greedy
WordPiece) with TensorFlow Text's normalization: NFKC case folding, NFD, removal of non-spacing
marks, control / format characters replaced by spaces, splitting on whitespace, punctuation,
ASCII symbols and CJK ideographs. `tests/test_tokenizer.py` compares it with TensorFlow Text.
"""
from __future__ import annotations

import os
import unicodedata
from typing import List, Sequence, Union

__all__ = ["ALPSTokenizer", "DEFAULT_VOCAB"]

DEFAULT_VOCAB = os.path.join(os.path.dirname(__file__), "assets", "bert_base_vocab_bos_eos.txt")

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
               (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))


def _is_split_char(ch: str) -> bool:
    """Characters that TF Text's BERT basic tokenizer keeps as tokens of their own."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True                                    # [!-/] [:-@] [\[-`] [{-~]
    if unicodedata.category(ch).startswith("P"):
        return True                                    # \p{P}
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _normalize(text: str) -> str:
    # case_fold_utf8 (ICU NFKC_Casefold) -> NFD -> drop \p{Mn} -> \p{Cc}|\p{Cf} to spaces
    text = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())
    text = unicodedata.normalize("NFD", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Mn":
            continue
        out.append(" " if cat in ("Cc", "Cf") else ch)
    return "".join(out)


def basic_split(text: str) -> List[str]:
    words, cur = [], []
    for ch in _normalize(text):
        if ch.isspace():
            if cur:
                words.append("".join(cur)); cur = []
        elif _is_split_char(ch):
            if cur:
                words.append("".join(cur)); cur = []
            words.append(ch)
        else:
            cur.append(ch)
    if cur:
        words.append("".join(cur))
    return words


class ALPSTokenizer:
    """`ALPSTokenizer()(["a photo of a dog."])` -> LongTensor [1, 32] for `alps.text.TextEncoder`."""

    def __init__(self, vocab_path: str = DEFAULT_VOCAB, context_length: int = 32,
                 max_bytes_per_word: int = 100):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = f.read().split("\n")
        if vocab and vocab[-1] == "":
            vocab = vocab[:-1]
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.context_length = context_length
        self.max_bytes_per_word = max_bytes_per_word
        self.pad_id = 0
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.bos_id = self.vocab["[bos]"]
        self.eos_id = self.vocab["[eos]"]

    def wordpiece(self, word: str) -> List[int]:
        if len(word.encode("utf-8")) > self.max_bytes_per_word:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, piece = len(word), None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    piece = self.vocab[sub]
                    break
                end -= 1
            if piece is None:
                return [self.unk_id]                   # the whole word is unknown
            ids.append(piece)
            start = end
        return ids

    def wordpiece_ids(self, text: str) -> List[int]:
        """WordPiece ids of `text`, without special tokens (what TF Text's BertTokenizer returns)."""
        ids: List[int] = []
        for w in basic_split(text):
            ids.extend(self.wordpiece(w))
        return ids

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id] + self.wordpiece_ids(text) + [self.eos_id]
        n = self.context_length - 1
        if len(ids) >= n:                              # truncated (or exactly full): [eos] last
            ids = ids[:n - 1] + [self.eos_id]
        else:
            ids = ids + [self.pad_id] * (n - len(ids))
        return ids + [self.cls_id]

    def __call__(self, texts: Union[str, Sequence[str]]):
        """LongTensor [B, 32] (torch is imported lazily, so the tokenizer itself has no torch dependency)."""
        import torch
        if isinstance(texts, str):
            texts = [texts]
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)
