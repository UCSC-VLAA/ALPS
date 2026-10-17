# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Recompute assets/flan_t5_base_empty_instruction.npz, the constant text token of the Stage-2 predictor.

The predictor attends to one constant key/value token, txt_proj(u0) (alps_jax/models/predictor.py). u0 is the last
hidden state of the frozen google/flan-t5-base encoder (Apache-2.0) at position 0 for the empty instruction: token ids
[EOS=1, PAD=0, ...] (32 slots) with the attention mask on the EOS slot only. ALPS computed it inside its training step
with bfloat16 activations and float32 weights; `u0_bf16` is that computation on CPU, `u0_f32` the float32 value.

    python tools/compute_t5_empty_token.py --out assets/flan_t5_base_empty_instruction.npz
    python tools/compute_t5_empty_token.py --weights <path>/model.safetensors --out ...

The weights come from the model.safetensors file of google/flan-t5-base (fetched with huggingface_hub unless --weights
is given; about 1 GB; only the encoder tensors are read). --weights also accepts an .npz with the parameter names of
the module below ("blk0/q/kernel", ...). The encoder is the T5 v1.1 architecture: shared token embedding without
scaling; per block RMSNorm (eps 1e-6) -> self-attention without 1/sqrt(d) scaling and with one relative position bias
(32 bidirectional buckets, max distance 128) computed once and shared by all blocks -> residual, RMSNorm -> gated GELU
(tanh) feed-forward -> residual; final RMSNorm. In float32 it matches transformers' T5EncoderModel.
"""
import argparse
import math

import numpy as np

REPO = "google/flan-t5-base"
NOTE = ("Output of the flan-t5-base encoder (google/flan-t5-base, Apache-2.0) at position 0 for the empty instruction: "
        "token ids [EOS=1, PAD...], attention mask on the EOS slot only. u0_bf16 = computed with bfloat16 activations "
        "and float32 weights, as during ALPS Stage-2 training (use this to reproduce ALPS); u0_f32 = float32 "
        "activations. Recompute with tools/compute_t5_empty_token.py.")


def params_from_hf(path, depth=12):
    """Encoder tensors of a Hugging Face T5 model.safetensors -> flat {name: float32 array} in this module's layout."""
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        keys = set(f.keys())
        get = lambda k: np.asarray(f.get_tensor(k), np.float32)
        emb = "shared.weight" if "shared.weight" in keys else "encoder.embed_tokens.weight"
        p = {"embed": get(emb), "rel_bias": get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
             "final_ln/weight": get("encoder.final_layer_norm.weight")}
        for i in range(depth):
            b = f"encoder.block.{i}.layer."
            p[f"blk{i}/ln0/weight"] = get(b + "0.layer_norm.weight")
            p[f"blk{i}/ln1/weight"] = get(b + "1.layer_norm.weight")
            for n in ("q", "k", "v", "o"):
                p[f"blk{i}/{n}/kernel"] = get(b + f"0.SelfAttention.{n}.weight").T.copy()   # torch [out, in] -> [in, out]
            for n in ("wi_0", "wi_1", "wo"):
                p[f"blk{i}/{n}/kernel"] = get(b + f"1.DenseReluDense.{n}.weight").T.copy()
    return p


def unflatten(flat):
    tree = {}
    for k, v in flat.items():
        d = tree
        *parents, leaf = k.split("/")
        for q in parents:
            d = d.setdefault(q, {})
        d[leaf] = v
    return tree


def build_encoder(dtype, vocab=32128, d_model=768, heads=12, d_kv=64, d_ff=2048, depth=12):
    """The flan-T5 encoder as a Flax module (imported lazily so that params_from_hf works without JAX); the defaults
    are the sizes of flan-t5-base."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    def rel_bucket(rel, num_buckets, max_distance):
        nb = num_buckets // 2
        ret = (rel > 0).astype(jnp.int32) * nb
        n = jnp.abs(rel)
        max_exact = nb // 2
        large = max_exact + (jnp.log(n.astype(jnp.float32) / max_exact + 1e-6) / math.log(max_distance / max_exact)
                             * (nb - max_exact)).astype(jnp.int32)
        return ret + jnp.where(n < max_exact, n, jnp.minimum(large, nb - 1))

    class RMSNorm(nn.Module):
        dtype: object

        @nn.compact
        def __call__(self, x):
            w = self.param("weight", nn.initializers.ones, (x.shape[-1],), jnp.float32)
            x32 = x.astype(jnp.float32)
            return (w * (x32 * jax.lax.rsqrt((x32 * x32).mean(-1, keepdims=True) + 1e-6))).astype(self.dtype)

    class Block(nn.Module):
        dtype: object

        @nn.compact
        def __call__(self, x, bias):
            dense = lambda name, n: nn.Dense(n, use_bias=False, dtype=self.dtype, param_dtype=jnp.float32, name=name)
            h = RMSNorm(self.dtype, name="ln0")(x)
            B, L, _ = h.shape
            q, k, v = (dense(n, heads * d_kv)(h).reshape(B, L, heads, d_kv) for n in ("q", "k", "v"))
            att = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) + bias
            att = jax.nn.softmax(att, axis=-1).astype(self.dtype)
            x = x + dense("o", d_model)(jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, L, -1))
            h = RMSNorm(self.dtype, name="ln1")(x)
            g = jax.nn.gelu(dense("wi_0", d_ff)(h), approximate=True) * dense("wi_1", d_ff)(h)
            return x + dense("wo", d_model)(g)

    class T5Encoder(nn.Module):
        dtype: object

        @nn.compact
        def __call__(self, ids, mask):
            emb = self.param("embed", nn.initializers.normal(1.0), (vocab, d_model), jnp.float32)
            x = jnp.take(emb, ids, axis=0).astype(self.dtype)
            pos = jnp.arange(ids.shape[1])
            bucket = rel_bucket(pos[None, :] - pos[:, None], 32, 128)
            rel = self.param("rel_bias", nn.initializers.normal(1.0), (32, heads), jnp.float32)
            bias = (jnp.take(rel, bucket, axis=0).transpose(2, 0, 1)[None].astype(jnp.float32)
                    + jnp.where(mask[:, None, None, :], 0.0, jnp.finfo(jnp.float32).min))
            for i in range(depth):
                x = Block(self.dtype, name=f"blk{i}")(x, bias)
            return RMSNorm(self.dtype, name="final_ln")(x).astype(jnp.float32)

    return T5Encoder(dtype=dtype)


def empty_instruction_token(flat, dtype, length=32, **sizes):
    """u0 = encoder output at position 0 for ids [1, 0, ..., 0] with the mask on slot 0."""
    import jax.numpy as jnp
    ids = jnp.zeros((1, length), jnp.int32).at[:, 0].set(1)
    params = {k: jnp.asarray(v, jnp.float32) for k, v in flat.items()}
    out = build_encoder(dtype, **sizes).apply({"params": unflatten(params)}, ids, ids != 0)
    return np.asarray(out, np.float32)[0, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="", help="model.safetensors of google/flan-t5-base (default: fetch it) "
                                                  "or an .npz in this module's parameter layout")
    ap.add_argument("--revision", default="main", help="Hugging Face revision used when fetching")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = a.weights
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, "model.safetensors", revision=a.revision)
    if path.endswith(".npz"):
        with np.load(path) as z:
            flat = {k: np.asarray(z[k], np.float32) for k in z.files}
    else:
        flat = params_from_hf(path)
    import jax.numpy as jnp
    u_bf16 = empty_instruction_token(flat, jnp.bfloat16)
    u_f32 = empty_instruction_token(flat, jnp.float32)
    cos = float(u_bf16 @ u_f32 / (np.linalg.norm(u_bf16) * np.linalg.norm(u_f32)))
    print(f"|u0_bf16| {np.linalg.norm(u_bf16):.6f}  |u0_f32| {np.linalg.norm(u_f32):.6f}  cosine {cos:.7f}")
    np.savez(a.out, u0_bf16=u_bf16, u0_f32=u_f32, note=np.array(NOTE))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
