# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Image caches of the reconstruction protocol, from the ImageNet-1k VALIDATION set (class folders, 50 files each).

    python build_imagenet_split.py <imagenet>/val <out_dir>

imagenet224_20k.npy: per class (sorted WordNet ids, sorted file names) the first 19 files for decoder training, then the
20th file of every class: the 1,000 held-out evaluation images (the LAST 1,000 entries).
imagenet224_50k.npy: per class the other 49 files for training, then the same 1,000 held-out images (the stronger
decoder of the supplementary control).
Preprocessing: shorter side to 224 (PIL bicubic), central 224 crop, uint8 RGB. A json next to each cache lists the
held-out files.
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def load(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    wn = sorted(os.listdir(root))
    assert len(wn) == 1000, len(wn)
    files = {w: sorted(f for f in os.listdir(os.path.join(root, w)) if f.lower().endswith((".jpeg", ".jpg", ".png")))
             for w in wn}
    assert all(len(v) == 50 for v in files.values()), "expected 50 validation images per class"
    held = [os.path.join(root, w, files[w][19]) for w in wn]
    for name, train in (("imagenet224_20k.npy", [os.path.join(root, w, f) for w in wn for f in files[w][:19]]),
                        ("imagenet224_50k.npy", [os.path.join(root, w, f) for w in wn
                                                 for k, f in enumerate(files[w]) if k != 19])):
        paths = train + held
        arr = np.lib.format.open_memmap(os.path.join(out, name), mode="w+", dtype=np.uint8, shape=(len(paths), 224, 224, 3))
        for i, p in enumerate(paths):
            arr[i] = load(p)
            if i % 5000 == 0:
                print(name, i, "/", len(paths), flush=True)
        arr.flush()
        del arr
        json.dump({"n_train": len(train), "n_heldout": len(held), "heldout_files": [os.path.relpath(p, root) for p in held],
                   "rule": "per class sorted files; held out = the 20th file of every class"},
                  open(os.path.join(out, name.replace(".npy", ".json")), "w"), indent=0)
        print("wrote", name, len(paths), flush=True)


if __name__ == "__main__":
    main()
