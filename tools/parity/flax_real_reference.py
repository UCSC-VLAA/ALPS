# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Float32 Flax forward with REAL ALPS weights, for the torch-vs-Flax check on real weights (JAX env).

    JAX_PLATFORMS=cpu python tools/parity/flax_real_reference.py --code train_jax \
        --stage1 stage1b_params.npz --encoder stage2_encoder.npz --out real_ref.npz
"""
import argparse
import os
import sys

import numpy as np

TEXTS = ["a photo of a dog.", "a bad photo of the tench", "pick up the black bowl between the plate and the ramekin and place it on the plate",
         "open the top drawer of the cabinet", "a robot arm pushes the t-shaped block to the target", "two cats sleeping on a red sofa",
         "a close-up photo of a vintage camera on a wooden table", "an aerial view of a city at night with bright lights",
         "a person riding a bicycle down a snowy street", "a bowl of ramen with an egg and green onions"]


def nest(flat):
    out = {}
    for k, v in flat.items():
        d = out
        parts = k.split("/")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code))
    import importlib.util
    spec = importlib.util.spec_from_file_location(        # the tokenizer alone, without importing torch
        "alps_tokenizer", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "alps", "tokenizer.py"))
    tokmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokmod)
    import jax.numpy as jnp
    from models import vit
    from models.static_visual_state_pretrainer import ProjectionHead, TextEncoder

    s1 = dict(np.load(a.stage1))
    sub = lambda d, p: {k[len(p):]: v for k, v in d.items() if k.startswith(p)}
    enc2 = np.load(a.encoder)
    p2 = {k[len("param/"):]: enc2[k] for k in enc2.files if k.startswith("param/")}
    x = enc2["ref_input"].astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    E = vit.Model(num_classes=0, variant="B/16", posemb="sincos2d", pool_type="gap", emb_head_bias=False,
                  head_zeroinit=False, use_dense_general=False, dtype=jnp.float32, param_dtype=jnp.float32)
    zg2, z2 = E.apply({"params": nest(p2)}, x)
    zg1, z1 = E.apply({"params": nest(sub(s1, "visual_encoder/"))}, x)
    T = TextEncoder(vocab_size=32000, width=768, depth=12, num_heads=12, mlp_dim=3072, max_len=32, pooling="cls_pool",
                    dtype=jnp.float32, param_dtype=jnp.float32)
    tok = tokmod.ALPSTokenizer()
    ids = np.array([tok.encode(t) for t in TEXTS], np.int32)
    pooled = T.apply({"params": nest(sub(s1, "text_encoder/"))}, ids)
    H = ProjectionHead(768)
    txt = H.apply({"params": nest(sub(s1, "text_projection_head/"))}, pooled)
    img2 = H.apply({"params": nest(sub(s1, "image_projection_head/"))}, jnp.mean(z2, axis=1))
    img1 = H.apply({"params": nest(sub(s1, "image_projection_head/"))}, jnp.mean(z1, axis=1))
    np.savez(a.out, x=x, z_stage2=np.asarray(z2), zg_stage2=np.asarray(zg2), z_stage1=np.asarray(z1), zg_stage1=np.asarray(zg1),
             ids=ids, txt_pooled=np.asarray(pooled), txt_emb=np.asarray(txt), img_emb_stage2=np.asarray(img2),
             img_emb_stage1=np.asarray(img1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
