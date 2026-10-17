#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write the trainable weights of a LIBERO policy checkpoint as safetensors (the format of the released policies).

    python export_policy_weights.py <run dir>/multitask_model_ep50.pth --out <dir>/multitask_model_ep50.safetensors

The input is a torch checkpoint of LIBERO ({"state_dict", "cfg", "previous_masks"}) written by train_policy.sh or by the
paper's runs. Checkpoints of the paper's runs also hold the frozen network of each camera (encoders.<i>.vit.*) and its
normalization buffers (encoders.<i>.im_mean, encoders.<i>.im_std); they are dropped, as the frozen-encoder module drops them
on loading (the frozen encoder is the encoder's own weights file). The buffers must agree between the cameras; their values
go into the metadata of the output. Every other tensor is written unchanged, and the output is read back and compared bit for
bit. LIBERO's policy registers one module under two names (extra_encoder.proprio_mlp is extra_encoder.encoders.1); torch's
state dict lists both names and safetensors stores no shared tensors, so each name is written with its own copy. The configuration stays in the run's config.json; evaluate_policy.py reads multitask_model_ep<N>.safetensors next to it.
Only convert checkpoints you trust: torch checkpoints are pickles.
"""
import argparse
import hashlib
import json
import os
import re

FROZEN = re.compile(r"^encoders\.\d+\.(vit\.|im_mean$|im_std$)")


def trainable_state_dict(sd):
    """(kept tensors, dropped keys, normalization buffers) of a policy state dict."""
    import torch
    keep = {k: v.detach().cpu().contiguous().clone() for k, v in sd.items() if not FROZEN.match(k)}   # clone: no shared storage
    dropped = sorted(k for k in sd if FROZEN.match(k))
    norm = {}
    for name in ("im_mean", "im_std"):
        vals = [sd[k].detach().float().cpu().reshape(-1) for k in sorted(sd) if k.endswith("." + name) and FROZEN.match(k)]
        if vals:
            if not all(torch.equal(v, vals[0]) for v in vals):
                raise ValueError(f"{name} differs between the cameras")
            norm[name] = [float(x) for x in vals[0]]
    return keep, dropped, norm


def export(src, out):
    import torch
    from safetensors.torch import load_file, save_file
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    keep, dropped, norm = trainable_state_dict(sd)
    if not keep:
        raise ValueError(f"{src}: no trainable tensors")
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {"format": "alps-libero-policy", "source": os.path.basename(src), "source_sha256": digest,
            "dropped_frozen_tensors": str(len(dropped)), "normalization": json.dumps(norm)}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_file(keep, out, metadata=meta)
    back = load_file(out)
    if set(back) != set(keep) or not all(torch.equal(back[k], keep[k]) and back[k].dtype == keep[k].dtype for k in keep):
        raise RuntimeError(f"{out}: read-back differs from the checkpoint")
    return keep, dropped, norm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep, dropped, norm = export(a.checkpoint, a.out)
    n = sum(v.numel() for v in keep.values())
    print(f"{a.out}: {len(keep)} tensors, {n / 1e6:.3f}M values; dropped {len(dropped)} frozen tensors; normalization {norm}")


if __name__ == "__main__":
    main()
