# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Layout of the ALPS token ids, and agreement with the reference tokenizers.

The reference is TensorFlow Text's BertTokenizer(lower_case=True), which the training code used;
the comparison runs when tensorflow_text is installed. HuggingFace's BertTokenizer (lower case)
agrees with it on ordinary text and is used as a second check when transformers is installed.
"""
import pytest

from alps.tokenizer import DEFAULT_VOCAB, ALPSTokenizer, basic_split

TEXTS = [
    "a photo of a dog.", "A photo of a DOG!", "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "open the top drawer of the cabinet", "a bad photo of the tench.", "itap of a great white shark.",
    "a café in Zürich, naïve résumé", "state-of-the-art, x-ray (2026) 3.5x larger; \"quoted\" & co.",
    "an origami cat", "the 1,000 held-out ImageNet images", "a photo of the hard to see grey whale.",
    "two people -- one on the left -- riding bikes...", "ice cream cone", "",
    "a long caption " + "with many repeated words " * 12,
]


@pytest.fixture(scope="module")
def tok():
    return ALPSTokenizer()


def test_layout(tok):
    ids = tok.encode("a photo of a dog.")
    assert len(ids) == 32 and ids[0] == tok.bos_id and ids[-1] == tok.cls_id
    k = ids.index(tok.eos_id)
    assert all(i == 0 for i in ids[k + 1:-1])
    empty = tok.encode("")
    assert empty[:2] == [tok.bos_id, tok.eos_id] and empty[-1] == tok.cls_id
    long = tok.encode("word " * 100)
    assert long[-2] == tok.eos_id and long[-1] == tok.cls_id and 0 not in long
    assert (tok.bos_id, tok.eos_id, tok.cls_id, tok.unk_id) == (1, 2, 101, 100)


def test_basic_split():
    assert basic_split("Hello, World!") == ["hello", ",", "world", "!"]
    assert basic_split("naïve café") == ["naive", "cafe"]


def test_against_huggingface(tok):
    transformers = pytest.importorskip("transformers")
    hf = transformers.BertTokenizer(DEFAULT_VOCAB, do_lower_case=True)
    for t in TEXTS:
        assert tok.wordpiece_ids(t) == hf.convert_tokens_to_ids(hf.tokenize(t)), t


def test_against_tensorflow_text(tok):
    tf_text = pytest.importorskip("tensorflow_text")
    import tensorflow as tf
    ref = tf_text.BertTokenizer(DEFAULT_VOCAB, token_out_type=tf.int32, lower_case=True)
    for t in TEXTS:
        want = ref.tokenize([t]).merge_dims(1, -1).to_list()[0]
        assert tok.wordpiece_ids(t) == want, t
