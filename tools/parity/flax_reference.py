# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Write reference outputs of the Flax ALPS modules for the PyTorch parity tests.

Run in a JAX environment with the Flax model code importable (`--code` points at the folder that
contains `models/`), on CPU:

    JAX_PLATFORMS=cpu python tools/parity/flax_reference.py --code train_jax --out /tmp/alps_flax_ref.npz

All parameters are perturbed away from their initial values (LayerNorm scales and biases too), so
a mapping error in any tensor changes the outputs. Depth and vocabulary are reduced to keep the
file small; the conversion code is depth-generic and the widths are the real ones.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="folder containing the Flax `models/` package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code))
    import jax
    import jax.numpy as jnp
    from models import vit
    from models.static_visual_state_pretrainer import ProjectionHead, TextEncoder

    def flat(tree, prefix=""):
        out = {}
        for k, v in tree.items():
            key = f"{prefix}/{k}" if prefix else k
            out.update(flat(v, key)) if hasattr(v, "items") else out.__setitem__(key, np.asarray(v))
        return out

    def perturb(params, key):
        leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        keys = jax.random.split(key, len(leaves))
        new = []
        for (path, x), k in zip(leaves, keys):
            name = "/".join(str(getattr(p, "key", p)) for p in path)
            noise = jax.random.normal(k, x.shape, x.dtype)
            if name.endswith("scale"):
                new.append(x + 0.1 * noise)                  # LayerNorm scale around 1
            else:
                new.append(x + 0.02 * noise)                 # kernels, biases, cls, embeddings
        return jax.tree_util.tree_unflatten(treedef, new)

    rng = jax.random.PRNGKey(0)
    r = np.random.RandomState(0)

    # -- visual encoder (the ALPS configuration, reduced depth) -----------------------------------
    E = vit.Model(num_classes=0, variant="B/16", depth=a.depth, posemb="sincos2d", pool_type="gap",
                  emb_head_bias=False, head_zeroinit=False, use_dense_general=False,
                  dtype=jnp.float32, param_dtype=jnp.float32)
    x224 = r.randn(2, 224, 224, 3).astype(np.float32)
    x_odd = r.randn(1, 160, 256, 3).astype(np.float32)      # 10 x 16 grid: recomputed sin-cos table
    pv = perturb(E.init(rng, x224)["params"], jax.random.PRNGKey(1))
    zg, z = E.apply({"params": pv}, x224)
    zg_odd, z_odd = E.apply({"params": pv}, x_odd)

    # -- text encoder + projection heads ------------------------------------------------------------
    T = TextEncoder(vocab_size=a.vocab, width=768, depth=a.depth, num_heads=12, mlp_dim=3072, max_len=32,
                    pooling="cls_pool", dtype=jnp.float32, param_dtype=jnp.float32)
    ids = np.zeros((3, 32), np.int32)
    for i, n in enumerate((5, 17, 40)):                     # short, medium, truncated
        body = list(r.randint(3, a.vocab, size=min(n, 29)))
        seq = [1] + body + [2]
        seq = (seq + [0] * 31)[:31]
        if n >= 29:
            seq[-1] = 2
        ids[i] = seq + [101]
    pt = perturb(T.init(rng, ids)["params"], jax.random.PRNGKey(2))
    t_pooled = T.apply({"params": pt}, ids)
    Hi, Ht = ProjectionHead(768), ProjectionHead(768)
    pi = perturb(Hi.init(rng, jnp.zeros((1, 768)))["params"], jax.random.PRNGKey(3))
    ph = perturb(Ht.init(rng, jnp.zeros((1, 768)))["params"], jax.random.PRNGKey(4))
    img_emb = Hi.apply({"params": pi}, jnp.mean(z, axis=1))
    txt_emb = Ht.apply({"params": ph}, t_pooled)

    out = {f"vit/{k}": v for k, v in flat(pv).items()}
    out.update({f"text/{k}": v for k, v in flat(pt).items()})
    out.update({f"img_proj/{k}": v for k, v in flat(pi).items()})
    out.update({f"txt_proj/{k}": v for k, v in flat(ph).items()})
    out.update(x224=x224, x_odd=x_odd, z=np.asarray(z), zg=np.asarray(zg), z_odd=np.asarray(z_odd),
               zg_odd=np.asarray(zg_odd), ids=ids, t_pooled=np.asarray(t_pooled),
               img_emb=np.asarray(img_emb), txt_emb=np.asarray(txt_emb),
               depth=np.int32(a.depth), vocab=np.int32(a.vocab))
    np.savez(a.out, **out)
    print("wrote", a.out, "keys", len(out))


if __name__ == "__main__":
    main()
