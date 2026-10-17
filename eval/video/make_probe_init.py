# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Initializations of the SSv2 probe (ssv2_probe.py --init_from).

The runs of the paper started every encoder from one shared initialization: the probe saved as probe_init.pt by the
first run (ALPS (Stage 1)), drawn from that run's random state (not released). This script draws a shared
initialization under torch seed 0 instead: the same protocol, but not bitwise the paper's starting point.
V-JEPA 2-L has 1,024-d tokens and 8 tubelets per clip; its initialization is the
shared one with the input adapter re-drawn (trunc-normal, std 0.02, under torch seed 0; bias 0) and the temporal
embedding reduced to 8 positions (zeros, as initialized); the attentive classifier is unchanged.

    python eval/video/make_probe_init.py --out probe_init.pt                       # needs VJEPA2_ROOT
    python eval/video/make_probe_init.py --base probe_init.pt --in_dim 1024 --t_tokens 8 --out probe_init_vjepa2.pt
"""
import argparse
import os
import sys

import torch


def fresh(in_dim=768, t_tokens=16, num_classes=174, d=768, heads=12, depth=4, seed=0):
    """The initialization that ssv2_probe.py draws (and saves as probe_init.pt) with these settings."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ssv2_probe import TemporalProbe
    torch.manual_seed(seed)
    return TemporalProbe(in_dim, t_tokens, num_classes, d, heads, depth).state_dict()


def adapt(base, in_dim, t_tokens, seed=0):
    """`base` for tokens of width `in_dim` and `t_tokens` temporal positions: adapter weight re-drawn, bias and temporal
    embedding zero, every other tensor unchanged."""
    sd = dict(base)
    d = base["adapter.weight"].shape[0]
    torch.manual_seed(seed)
    w = torch.empty(d, in_dim)
    torch.nn.init.trunc_normal_(w, std=0.02)
    sd["adapter.weight"] = w
    sd["adapter.bias"] = torch.zeros(d)
    sd["tpos"] = torch.zeros(1, t_tokens, 1, d)
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="shared initialization to adapt (default: draw it)")
    ap.add_argument("--in_dim", type=int, default=768, help="token width of the encoder")
    ap.add_argument("--t_tokens", type=int, default=16, help="temporal tokens per clip (16 frames: 16, or 8 for tubelets of 2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.base:
        base = torch.load(a.base, map_location="cpu", weights_only=True)
    else:
        base = fresh(seed=a.seed)
    sd = base if (a.in_dim, a.t_tokens) == (768, 16) else adapt(base, a.in_dim, a.t_tokens, a.seed)
    torch.save(sd, a.out)
    print(a.out, {k: tuple(v.shape) for k, v in sd.items() if not k.startswith("clf.")})


if __name__ == "__main__":
    main()
