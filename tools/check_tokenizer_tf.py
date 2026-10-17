# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare alps.tokenizer with TensorFlow Text's BertTokenizer (the tokenizer of the training code) on a text corpus.

    python tools/check_tokenizer_tf.py --data <clip_benchmark_datasets>      # COCO + Flickr test captions, ImageNet prompts
    python tools/check_tokenizer_tf.py --texts file.txt                      # one text per line

Prints the number of texts whose WordPiece ids differ, and the first differences.
"""
import argparse
import importlib.util
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_tokenizer_module():
    spec = importlib.util.spec_from_file_location("alps_tokenizer", os.path.join(ROOT, "alps", "tokenizer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def corpus(data):
    texts = []
    J = json.load(open(os.path.join(data, "mscoco_captions", "coco_test_karpathy.json")))
    texts += [a["caption"] for a in J["annotations"]]
    with open(os.path.join(data, "flickr30k", "flickr30k_test_karpathy.txt")) as f:
        f.readline()
        texts += [line.strip().split(".jpg,", 1)[1] for line in f if line.strip()]
    lists = json.load(open(os.path.join(ROOT, "alps", "assets", "imagenet_zeroshot.json")))
    for key in ("clip_benchmark", "big_vision"):
        cn, tp = lists[key]["class_names"], lists[key]["templates"]
        texts += [t.replace("{c}", c).replace("{}", c) for c in cn for t in tp]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--texts")
    a = ap.parse_args()
    import tensorflow as tf
    import tensorflow_text
    texts = corpus(a.data) if a.data else [l.rstrip("\n") for l in open(a.texts)]
    tok = load_tokenizer_module().ALPSTokenizer()
    ref = tensorflow_text.BertTokenizer(tok_path := load_tokenizer_module().DEFAULT_VOCAB, token_out_type=tf.int32,
                                        lower_case=True)
    bad = []
    for i in range(0, len(texts), 2048):
        chunk = texts[i:i + 2048]
        want = ref.tokenize(tf.constant(chunk)).merge_dims(1, -1).to_list()
        for t, w in zip(chunk, want):
            if tok.wordpiece_ids(t) != w:
                bad.append((t, w, tok.wordpiece_ids(t)))
    print(f"{len(texts)} texts, {len(bad)} with different WordPiece ids (vocab {os.path.basename(tok_path)})")
    for t, w, g in bad[:10]:
        print(repr(t), "\n  tf_text:", w, "\n  alps:   ", g)


if __name__ == "__main__":
    main()
