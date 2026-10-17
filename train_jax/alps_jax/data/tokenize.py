# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Caption tokenization of Stage 1 (TensorFlow Text), as in the original training code.

BERT-base uncased WordPiece (lower case) with slots 1 / 2 renamed [bos] / [eos]. Every field of length L is
    [bos] w_1 .. w_k [eos] [PAD] .. [PAD] [CLS]
i.e. [bos] + WordPiece ids + [eos], truncated or right-padded with 0 to L - 1 ids (the last of them forced to [eos]
when the caption fills the field), then [CLS] in slot L - 1 (also in the caption-decoder field, where it is a target).
"""
import os

import numpy as np
import tensorflow as tf

DEFAULT_VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets",
                             "bert_base_vocab_bos_eos.txt")


def _tokenizer(vocab_path):
    import tensorflow_text
    with tf.io.gfile.GFile(vocab_path) as f:
        vocab = f.read().split("\n")
    ids = {t: vocab.index(t) for t in ("[CLS]", "[bos]", "[eos]")}
    tok = tensorflow_text.BertTokenizer(vocab_path, token_out_type=tf.int32, lower_case=True)
    return tok, ids["[CLS]"], ids["[bos]"], ids["[eos]"]


def _have_tf_text():
    try:
        import tensorflow_text  # noqa: F401
        return True
    except Exception:
        return False


def make_field_tokenizer(vocab_path=DEFAULT_VOCAB, max_len=32, backend="auto"):
    """scalar string -> [max_len] int32 in the layout above.

    backend "tf_text" (the original pipeline), "python" (alps_jax/data/py_tokenizer.py through tf.numpy_function, for
    machines without tensorflow_text; slower) or "auto"."""
    if backend == "python" or (backend == "auto" and not _have_tf_text()):
        from .py_tokenizer import ALPSTokenizer
        py = ALPSTokenizer(vocab_path, context_length=max_len)

        def fn_py(txt):
            out = tf.numpy_function(lambda s: np.asarray(py.encode(s.decode("utf-8")), np.int32), [txt], tf.int32)
            out.set_shape([max_len])
            return out
        return fn_py
    import tensorflow_text
    tok, cls_id, bos_id, eos_id = _tokenizer(vocab_path)

    def fn(txt):
        ids = tok.tokenize(tf.reshape(txt, [1]))
        ids = ids.merge_dims(1, -1) if ids.ragged_rank > 1 else ids
        cnt = tf.shape(ids)[0]
        ids = tf.concat([tf.fill([cnt, 1], bos_id), ids, tf.fill([cnt, 1], eos_id)], axis=1)
        padded, mask = tensorflow_text.pad_model_inputs(ids, max_len - 1)
        padded = tf.cond(tf.equal(mask[0, -1], 1),
                         lambda: tf.concat([padded[:, :-1], tf.fill([cnt, 1], eos_id)], axis=1),
                         lambda: padded)
        return tf.concat([padded, tf.fill([cnt, 1], cls_id)], axis=1)[0]
    return fn


def random_element(x):
    """A scalar string or a list of strings -> one of them, uniformly at random (stateful RNG)."""
    x = tf.reshape(x, [-1])
    x = tf.concat([x, [""]], axis=0)
    n = tf.maximum(tf.shape(x)[0] - 1, 1)
    return x[tf.random.uniform([], 0, n, dtype=tf.int32)]


def random_sentence(dense):
    """Join all parts of the synthetic caption, split on '.' / '!', and pick one non-empty sentence at random."""
    dense = tf.strings.reduce_join(tf.reshape(dense, [-1]), separator=" ")
    parts = tf.strings.split(tf.reshape(tf.strings.regex_replace(dense, "[.!]+", "|"), [1]), sep="|").values
    parts = tf.strings.strip(parts)
    parts = tf.boolean_mask(parts, tf.strings.length(parts) > 0)
    n = tf.shape(parts)[0]
    return tf.cond(n > 0, lambda: parts[tf.random.uniform([], 0, tf.maximum(n, 1), tf.int32)], lambda: dense)


def make_stage1_text_fn(vocab_path=DEFAULT_VOCAB, web_key="txt", synthetic_key="synthetic_caption", itc_len=32,
                        caption_len=80, backend="auto"):
    """Adds labels [80] (the synthetic caption, caption decoder), itc_labels1 [32] (the web caption) and
    itc_labels2 [32] (one sentence of the synthetic caption) to an example."""
    itc_tok = make_field_tokenizer(vocab_path, itc_len, backend)
    cap_tok = make_field_tokenizer(vocab_path, caption_len, backend)

    def fn(ex):
        ex = dict(ex)
        ex["labels"] = cap_tok(random_element(ex[synthetic_key]))
        ex["itc_labels1"] = itc_tok(tf.reshape(ex[web_key], [-1])[0])
        ex["itc_labels2"] = itc_tok(random_sentence(ex[synthetic_key]))
        return ex
    return fn
