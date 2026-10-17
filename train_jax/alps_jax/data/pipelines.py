# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""tf.data input pipelines of both stages. Each host reads its own shard and yields host-local numpy batches.

Data are not part of the release. Expected inputs:

Stage 1 (image-text pairs): TFRecord files of tf.train.Example with
    image              JPEG/PNG bytes
    caption            the web caption (alt text), string
    synthetic_caption  a long synthetic caption, string (or a list of strings)
(the keys are configurable), or a TFDS directory with the same fields, read through tfds.builder_from_directory as in
the original runs (DataComp-1B with the Recap-DataComp-1B LLaVA-LLaMA-3 captions).

Stage 2 (single images): TFRecord files with an `image` field, or with `source_jpg` / `target_jpg` (an image pair, of
which one image is drawn with probability 0.5 per example, as ALPS did with image-editing records). Several sources
can be mixed with sampling weights.
"""
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .tokenize import DEFAULT_VOCAB, make_stage1_text_fn
from .transforms import inception_crop, make_crossview_pair, make_two_view


def _host_files(patterns, process_index, process_count, shuffle=True):
    files = []
    for p in ([patterns] if isinstance(patterns, str) else patterns):
        files += sorted(tf.io.gfile.glob(p))
    if not files:
        raise FileNotFoundError(f"no files match {patterns}")
    ds = tf.data.Dataset.from_tensor_slices(files).shard(process_count, process_index)
    return ds.shuffle(len(files), reshuffle_each_iteration=True).repeat() if shuffle else ds.repeat()


def _decode(img_bytes):
    return tf.io.decode_image(img_bytes, channels=3, expand_animations=False)


def stage1_dataset(*, res, per_host_batch, files=None, tfds_dir=None, tfds_split="full", image_key="image",
                   web_key="caption", synthetic_key="synthetic_caption", vocab_path=DEFAULT_VOCAB, itc_len=32,
                   caption_len=80, shuffle_buffer=250_000, process_index=0, process_count=1,
                   threads=48) -> tf.data.Dataset:
    text_fn = make_stage1_text_fn(vocab_path, web_key, synthetic_key, itc_len, caption_len)
    if tfds_dir:
        import tensorflow_datasets as tfds
        builder = tfds.builder_from_directory(tfds_dir)
        split = tfds.even_splits(tfds_split, process_count)[process_index]
        ds = builder.as_dataset(split=split, shuffle_files=True,
                                read_config=tfds.ReadConfig(skip_prefetch=True, try_autocache=False))
        decode = lambda ex: ex[image_key]                                  # TFDS decodes Image features
    else:
        spec = {image_key: tf.io.FixedLenFeature([], tf.string), web_key: tf.io.VarLenFeature(tf.string),
                synthetic_key: tf.io.VarLenFeature(tf.string)}
        ds = _host_files(files, process_index, process_count).interleave(
            tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        def parse(rec):
            ex = tf.io.parse_single_example(rec, spec)
            return {image_key: ex[image_key], web_key: tf.sparse.to_dense(ex[web_key]),
                    synthetic_key: tf.sparse.to_dense(ex[synthetic_key])}
        ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
        decode = lambda ex: _decode(ex[image_key])

    def prep(ex):
        img = inception_crop(decode(ex), res, area_min=40, method="bilinear", antialias=True)
        out = text_fn({web_key: ex[web_key], synthetic_key: ex[synthetic_key]})
        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        v = make_two_view(tf.cast(img, tf.uint8), res=res, dense_scale=(0.5, 1.0), seed=seed)
        return {**v, "labels": out["labels"], "itc_labels1": out["itc_labels1"], "itc_labels2": out["itc_labels2"]}

    opts = tf.data.Options()
    opts.threading.private_threadpool_size = threads
    ds = ds.with_options(opts).repeat().shuffle(shuffle_buffer).map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.ignore_errors().batch(per_host_batch, drop_remainder=True).prefetch(2)


def stage2_dataset(*, sources: Sequence[Tuple[str, float]], res, per_host_batch, crossview_cfg: Optional[Dict] = None,
                   shuffle_buffer=50_000, process_index=0, process_count=1, threads=48) -> tf.data.Dataset:
    """sources: [(file pattern, sampling weight), ...]."""
    def read(pattern, cycle):
        ds = _host_files(pattern, process_index, process_count).interleave(
            tf.data.TFRecordDataset, cycle_length=cycle, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return ds.map(_parse_stage2, num_parallel_calls=tf.data.AUTOTUNE)

    streams = [read(p, 16 if i == 0 else 8) for i, (p, _) in enumerate(sources)]
    weights = [float(w) for _, w in sources]
    ds = streams[0] if len(streams) == 1 else tf.data.Dataset.sample_from_datasets(
        streams, weights=[w / sum(weights) for w in weights])

    def prep(img_bytes):
        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        v = make_crossview_pair(_decode(img_bytes), res, seed=seed, cfg=crossview_cfg)
        return {k: v[k] for k in ("x_s_rgb", "x_t_rgb", "box_a", "box_b")}

    opts = tf.data.Options()
    opts.threading.private_threadpool_size = threads
    ds = ds.with_options(opts).shuffle(shuffle_buffer).map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.ignore_errors().batch(per_host_batch, drop_remainder=True).prefetch(2)


def _parse_stage2(rec):
    spec = {"image": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "source_jpg": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "target_jpg": tf.io.FixedLenFeature([], tf.string, default_value="")}
    ex = tf.io.parse_single_example(rec, spec)
    pair = tf.where(tf.random.uniform([]) < 0.5, ex["source_jpg"], ex["target_jpg"])
    return tf.where(tf.strings.length(ex["image"]) > 0, ex["image"], pair)


def as_numpy(ds) -> Iterator[Dict[str, np.ndarray]]:
    for batch in ds.as_numpy_iterator():
        yield batch
