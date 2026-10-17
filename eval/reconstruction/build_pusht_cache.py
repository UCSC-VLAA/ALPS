# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""PushT frames for the reconstruction decoders that render world-model predictions (Tab. S14, S15, Fig. 6).

    python build_pusht_cache.py <DINO-WM data>/pusht_noise pusht224_cache.npy
    python train_decoder.py --enc alps --weights ... --cache pusht224_cache.npy --out runs/recon_pusht      # 20,000 steps, batch 8

Same format as the ImageNet caches: (20,000, 224, 224, 3) uint8, of which train_decoder.py trains on the first 19,000 and
holds out the last 1,000. Frames are read with decord like DINO-WM's PushT loader.
  19,000 training frames: uniform without replacement over all frames of the training episodes (seed 20261002), sorted by
  episode; 1,000 held-out frames: evenly spread over the frames of the validation episodes.
Writes the cache and <cache>_index.json (split, episode, frame of every row).
"""
import json
import os
import pickle
import sys
import time

import numpy as np


def main():
    D, OUT = sys.argv[1], sys.argv[2]
    from decord import VideoReader
    assert not os.path.exists(OUT), OUT + " exists"
    sl = [int(x) for x in pickle.load(open(f"{D}/train/seq_lengths.pkl", "rb"))]
    sv = [int(x) for x in pickle.load(open(f"{D}/val/seq_lengths.pkl", "rb"))]
    rng = np.random.default_rng(20261002)
    ct = np.cumsum([0] + sl)
    pick = np.sort(rng.choice(ct[-1], 19000, replace=False))
    ept = np.searchsorted(ct, pick, side="right") - 1
    frt = pick - ct[ept]
    cv = np.cumsum([0] + sv)
    pv = np.unique(np.round(np.linspace(0, cv[-1] - 1, 1000)).astype(int))
    assert len(pv) == 1000
    epv = np.searchsorted(cv, pv, side="right") - 1
    frv = pv - cv[epv]
    arr = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.uint8, shape=(20000, 224, 224, 3))
    rows, t0 = [], time.time()

    def fill(split, eps, frs, i):
        for n, e in enumerate(np.unique(eps)):
            sel = [int(f) for f in frs[eps == e]]
            b = VideoReader(f"{D}/{split}/obses/episode_{e:03d}.mp4", num_threads=1).get_batch(sel).asnumpy()
            assert b.shape == (len(sel), 224, 224, 3) and b.dtype == np.uint8, (split, e, b.shape)
            arr[i:i + len(sel)] = b
            rows.extend([(split, int(e), f) for f in sel])
            i += len(sel)
            if n % 2000 == 0:
                print(f"{split} episode {n} rows {i} ({time.time() - t0:.0f} s)", flush=True)
        return i

    i = fill("train", ept, frt, 0)
    assert i == 19000
    i = fill("val", epv, frv, i)
    assert i == 20000
    arr.flush()
    del arr
    json.dump({"rows": rows, "seed": 20261002, "note": "last 1000 rows = held-out val frames"},
              open(OUT[:-4] + "_index.json", "w"))
    print(f"done: {OUT}, 20000 frames ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
