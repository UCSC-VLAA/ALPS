# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Input pipelines of both stages as PyTorch IterableDatasets (infinite streams; training counts steps, not epochs).

Data are not part of the release. Accepted inputs ("sources"):

    *.jsonl        one JSON object per line. Stage 1: {"image": <path>, "caption": <web caption, str or list>,
                   "synthetic_caption": <long synthetic caption, str or list>} (keys configurable). Stage 2:
                   {"image": <path>} or an image pair {"source": <path>, "target": <path>}, of which one image is
                   drawn with probability 0.5 per example (as ALPS did with image-editing records). Relative paths
                   are resolved against the directory of the JSONL file.
    *.txt          Stage 2: one image path per line.
    *.tar shards   WebDataset layout: files <key>.<ext> grouped by key. Images: jpg / jpeg / png / webp / bmp.
                   Stage 1 reads the captions from <key>.json (same keys as the JSONL records) or the web caption
                   from <key>.txt; Stage 2 draws one of the record's images uniformly. Patterns may use globs and
                   brace ranges ("shards/{000000..000127}.tar").
    a directory or a glob of image files (Stage 2).

Each (rank, worker) reads its own share of the records (JSONL / lists: every n-th record of a per-pass permutation;
tar: every n-th shard of a per-pass shuffled shard list, plus a shuffle buffer of raw records), decodes with PIL,
applies the transforms of alps_torch.data.transforms with draws from its own numpy generator, and skips records that
fail to decode (the JAX pipeline's ignore_errors). Several Stage-2 sources are mixed per example with the given
weights (tf.data.Dataset.sample_from_datasets).

Differences from the TF pipeline: JPEG decoding by PIL instead of tf.io.decode_image (libjpeg version / DCT
settings may change pixels by a level), different random streams, and shuffling by permutation (map-style sources)
or a smaller shuffle buffer (tar shards; default 2,000 records instead of 250,000 / 50,000).
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import tarfile
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import tf_ops as T
from . import transforms as X
from .text import Stage1Text

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "bmp")
MAX_CONSECUTIVE_FAILURES = 1000          # a stream that only fails (wrong paths, unreadable data) is an error


def _skip(ds, err):
    ds.n_failed += 1
    ds._fail_run = getattr(ds, "_fail_run", 0) + 1
    if ds._fail_run >= MAX_CONSECUTIVE_FAILURES:
        raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive records failed; last error: {err!r}")


def decode_image(data: Union[bytes, str]) -> np.ndarray:
    """bytes or a path -> uint8 [H, W, 3] RGB."""
    from PIL import Image
    im = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def expand_pattern(pattern: Union[str, Sequence[str]]) -> List[str]:
    """Globs and brace ranges {a..b} (zero padded like the bounds), or a list of such patterns."""
    if not isinstance(pattern, str):
        return [f for p in pattern for f in expand_pattern(p)]
    m = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if m:
        a, b = m.group(1), m.group(2)
        width = len(a) if a.startswith("0") else 0
        return [f for i in range(int(a), int(b) + 1)
                for f in expand_pattern(pattern[:m.start()] + str(i).zfill(width) + pattern[m.end():])]
    if any(c in pattern for c in "*?["):
        return sorted(glob.glob(pattern))
    return [pattern]


# ----------------------------------------------------------------------------------------------------- sources
class _Source:
    def records(self, worker: int, n_workers: int, rng: np.random.Generator) -> Iterator[dict]:
        raise NotImplementedError


class ListSource(_Source):
    """A list of records (dicts). Each pass is a fresh permutation shared by all readers (seeded by `seed` and
    the pass number); reader w of n takes every n-th record of it."""

    def __init__(self, records: List[dict]):
        if not records:
            raise ValueError("empty source")
        self.items = records
        self.seed = 0

    def records(self, worker, n_workers, rng):
        n, epoch = len(self.items), 0
        while True:
            if n >= n_workers:
                perm = np.random.default_rng([self.seed, epoch]).permutation(n)[worker::n_workers]
            else:                                        # fewer records than readers: each reader shuffles all
                perm = rng.permutation(n)
            for i in perm:
                yield self.items[int(i)]
            epoch += 1


def _resolve(base: str, p: str) -> str:
    return p if os.path.isabs(p) or "://" in p else os.path.join(base, p)


def jsonl_source(path: str, image_keys=("image", "source", "target")) -> ListSource:
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in image_keys:
                if k in r and isinstance(r[k], str):
                    r[k] = _resolve(base, r[k])
            items.append(r)
    return ListSource(items)


def image_list_source(spec: str) -> ListSource:
    if spec.endswith(".txt"):
        base = os.path.dirname(os.path.abspath(spec))
        with open(spec, encoding="utf-8") as f:
            paths = [_resolve(base, ln.strip()) for ln in f if ln.strip()]
    elif os.path.isdir(spec):
        paths = sorted(p for p in glob.glob(os.path.join(spec, "**", "*"), recursive=True)
                       if p.rsplit(".", 1)[-1].lower() in IMAGE_EXTS)
    else:
        paths = [p for p in expand_pattern(spec) if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(f"no images found for source {spec!r}")
    return ListSource([{"image": p} for p in paths])


class TarSource(_Source):
    """WebDataset-style tar shards: records = {extension: bytes} grouped by key, plus "__key__"."""

    def __init__(self, pattern, shuffle_buffer: int = 2000):
        self.shards = expand_pattern(pattern)
        if not self.shards:
            raise FileNotFoundError(f"no shards match {pattern}")
        self.shuffle_buffer = shuffle_buffer

    @staticmethod
    def iter_shard(path: str) -> Iterator[dict]:
        with tarfile.open(path, "r|*") as tf:
            cur_key, cur = None, {}
            for m in tf:
                if not m.isfile():
                    continue
                name = os.path.basename(m.name)
                if "." not in name:
                    continue
                key, ext = name.split(".", 1)
                key = os.path.join(os.path.dirname(m.name), key)
                if cur_key is not None and key != cur_key:
                    yield dict(cur, __key__=cur_key)
                    cur = {}
                cur_key = key
                cur[ext.lower()] = tf.extractfile(m).read()
            if cur:
                yield dict(cur, __key__=cur_key)

    def records(self, worker, n_workers, rng):
        mine = self.shards[worker::n_workers] if len(self.shards) >= n_workers else self.shards
        buf: List[dict] = []
        cap = self.shuffle_buffer
        while True:
            seen = 0
            for i in rng.permutation(len(mine)):
                for r in self.iter_shard(mine[int(i)]):
                    seen += 1
                    if len(buf) < cap:
                        buf.append(r)
                        continue
                    j = int(rng.integers(len(buf)))
                    out, buf[j] = buf[j], r
                    yield out
            if seen == 0:
                raise RuntimeError(f"no records in {mine[:3]}")
            cap = min(cap, len(buf))                     # less data than the buffer: shuffle within it from now on


def make_source(spec, shuffle_buffer: int = 2000) -> _Source:
    """A .jsonl file, tar shards (a pattern ending in .tar or a list of .tar paths), a .txt list of image paths, a
    directory or a glob of images."""
    if isinstance(spec, _Source):
        return spec
    if isinstance(spec, (list, tuple)):
        if spec and all(str(x).endswith(".tar") for x in spec):
            return TarSource(list(spec), shuffle_buffer)
        raise ValueError("a list source must be a list of .tar shards")
    if spec.endswith(".jsonl"):
        return jsonl_source(spec)
    if spec.endswith(".tar"):
        return TarSource(spec, shuffle_buffer)
    return image_list_source(spec)


def _worker_info(rank: int, world: int):
    info = get_worker_info()
    wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
    return rank * nw + wid, world * nw


# ----------------------------------------------------------------------------------------------------- Stage 1
class Stage1Dataset(IterableDataset):
    """Yields {x_rgb, gx_rgb [res, res, 3] float32, crop_box [4], flip, labels [80], itc_labels1 [32],
    itc_labels2 [32]} examples."""

    def __init__(self, source, res: int, *, seed: int = 0, rank: int = 0, world: int = 1, image_key: str = "image",
                 web_key: str = "caption", synthetic_key: str = "synthetic_caption", itc_len: int = 32,
                 caption_len: int = 80, shuffle_buffer: int = 2000, vocab_path: Optional[str] = None):
        super().__init__()
        self.source = make_source(source, shuffle_buffer)
        self.res, self.seed, self.rank, self.world = res, seed, rank, world
        self.image_key, self.web_key, self.synthetic_key = image_key, web_key, synthetic_key
        self.text = Stage1Text(**({"vocab_path": vocab_path} if vocab_path else {}), itc_len=itc_len,
                               caption_len=caption_len)
        self.n_failed = 0

    def _fields(self, r: dict):
        if "__key__" in r:                                         # tar record
            meta = json.loads(r["json"]) if "json" in r else {}
            img = r.get(self.image_key) or next((v for k, v in r.items() if k != "__key__"
                                                 and k.rsplit(".", 1)[-1] in IMAGE_EXTS), None)
            web = meta.get(self.web_key, r["txt"].decode("utf-8") if "txt" in r else None)
            syn = meta.get(self.synthetic_key)
        else:
            img, web, syn = r[self.image_key], r.get(self.web_key), r.get(self.synthetic_key)
        if img is None or web is None or syn is None:
            raise KeyError("record without image, web caption or synthetic caption")
        return img, web, syn

    def process(self, r: dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        img, web, syn = self._fields(r)
        crop = X.inception_crop(decode_image(img), self.res, rng=T.NumpyBoxRNG(rng))
        out = self.text(web, syn, rng)
        v = X.make_two_view(crop, self.res, X.two_view_draws(rng))
        return {**v, **out}

    def __iter__(self):
        w, n = _worker_info(self.rank, self.world)
        rng = np.random.default_rng([self.seed, w, 1])
        self.source.seed = self.seed
        for r in self.source.records(w, n, rng):
            try:
                ex = self.process(r, rng)
            except Exception as e:                                # undecodable / incomplete record: skip
                _skip(self, e)
                continue
            self._fail_run = 0
            yield ex


# ----------------------------------------------------------------------------------------------------- Stage 2
class Stage2Dataset(IterableDataset):
    """Yields {x_s_rgb, x_t_rgb [res, res, 3] float32, box_a, box_b [3]} cross-view pairs. sources: [(spec, weight)]."""

    def __init__(self, sources: Sequence[Tuple[object, float]], res: int, *, seed: int = 0, rank: int = 0,
                 world: int = 1, crossview_cfg: Optional[dict] = None, shuffle_buffer: int = 2000):
        super().__init__()
        self.sources = [make_source(s, shuffle_buffer) for s, _ in sources]
        w = np.asarray([float(x) for _, x in sources], np.float64)
        self.weights = w / w.sum()
        self.res, self.seed, self.rank, self.world = res, seed, rank, world
        self.cfg = dict(X.CROSSVIEW_DEFAULTS, **(crossview_cfg or {}))
        self.n_failed = 0

    @staticmethod
    def _image(r: dict, rng: np.random.Generator):
        if "__key__" in r:
            imgs = [v for k, v in r.items() if k != "__key__" and k.rsplit(".", 1)[-1] in IMAGE_EXTS]
            if not imgs:
                raise KeyError("tar record without image")
            return imgs[int(rng.integers(len(imgs)))] if len(imgs) > 1 else imgs[0]
        if "image" in r:
            return r["image"]
        return r["source"] if rng.random() < 0.5 else r["target"]       # one image of the pair, p 0.5

    def process(self, r: dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        img = decode_image(self._image(r, rng))
        v = X.make_crossview_pair(img, self.res, X.crossview_draws(rng, int(self.cfg["tries"])), self.cfg)
        return {k: v[k] for k in ("x_s_rgb", "x_t_rgb", "box_a", "box_b")}

    def __iter__(self):
        w, n = _worker_info(self.rank, self.world)
        rng = np.random.default_rng([self.seed, w, 2])
        for s in self.sources:
            s.seed = self.seed
        streams = [s.records(w, n, rng) for s in self.sources]
        while True:
            i = int(rng.choice(len(streams), p=self.weights)) if len(streams) > 1 else 0
            r = next(streams[i])
            try:
                ex = self.process(r, rng)
            except Exception as e:                                # undecodable record: skip
                _skip(self, e)
                continue
            self._fail_run = 0
            yield ex


# --------------------------------------------------------------------------------------------------- batching
_IMAGE_KEYS = ("x_rgb", "gx_rgb", "x_s_rgb", "x_t_rgb")


def collate(examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack examples; images become [B, 3, H, W] float32, token fields int64."""
    out = {}
    for k in examples[0]:
        a = np.stack([e[k] for e in examples])
        t = torch.from_numpy(np.ascontiguousarray(a))
        if k in _IMAGE_KEYS:
            t = t.permute(0, 3, 1, 2).contiguous()
        elif t.dtype == torch.int32:
            t = t.long()
        out[k] = t
    return out


def make_loader(dataset: IterableDataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate, drop_last=True)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **kw)
