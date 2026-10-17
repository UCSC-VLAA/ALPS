# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tiny synthetic datasets in the accepted input formats (JSONL + image files, WebDataset tar shards)."""
import io
import json
import os
import tarfile

import numpy as np

WORDS = "a the red blue small large robot arm drawer bowl plate table cat dog car tree house on in near".split()


def _jpeg(rng, h, w):
    from PIL import Image
    yy, xx = np.mgrid[:h, :w]
    img = np.stack([127 + 100 * np.sin(xx / (3 + c) + rng.random() * 6) * np.cos(yy / (4 + c)) for c in range(3)], -1)
    img = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _captions(rng):
    cap = " ".join(rng.choice(WORDS, int(rng.integers(3, 12))))
    syn = ". ".join(" ".join(rng.choice(WORDS, int(rng.integers(4, 15)))) for _ in range(3)) + "."
    return cap, syn


def stage1_jsonl(root, n=24, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    path = os.path.join(root, "stage1.jsonl")
    with open(path, "w") as f:
        for i in range(n):
            name = f"img/{i:04d}.jpg"
            with open(os.path.join(root, name), "wb") as g:
                g.write(_jpeg(rng, int(rng.integers(40, 90)), int(rng.integers(40, 90))))
            cap, syn = _captions(rng)
            f.write(json.dumps({"image": name, "caption": cap, "synthetic_caption": syn if i % 3 else [syn, syn[::-1]]}) + "\n")
    return path


def stage1_tar(root, n=24, shards=2, seed=1):
    rng = np.random.default_rng(seed)
    paths = []
    for s in range(shards):
        p = os.path.join(root, f"s1_{s:03d}.tar")
        with tarfile.open(p, "w") as tf:
            for i in range(n // shards):
                key = f"{s:03d}{i:05d}"
                cap, syn = _captions(rng)
                for ext, data in (("jpg", _jpeg(rng, int(rng.integers(40, 90)), int(rng.integers(40, 90)))),
                                  ("json", json.dumps({"caption": cap, "synthetic_caption": syn}).encode())):
                    ti = tarfile.TarInfo(f"{key}.{ext}")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
        paths.append(p)
    return os.path.join(root, "s1_{000.." + f"{shards - 1:03d}" + "}.tar")


def stage2_jsonl(root, n=16, seed=2, pairs=True):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "img2"), exist_ok=True)
    path = os.path.join(root, "stage2_pairs.jsonl" if pairs else "stage2.jsonl")
    with open(path, "w") as f:
        for i in range(n):
            names = []
            for j in range(2 if pairs else 1):
                name = f"img2/{i:04d}_{j}.jpg"
                with open(os.path.join(root, name), "wb") as g:
                    g.write(_jpeg(rng, int(rng.integers(40, 90)), int(rng.integers(40, 90))))
                names.append(name)
            f.write(json.dumps({"source": names[0], "target": names[1]} if pairs else {"image": names[0]}) + "\n")
    return path
