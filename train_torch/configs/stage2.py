# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of ALPS: cross-view self-distillation on single images (5 million draws). Mirrors
train_jax/configs/stage2.py."""
from alps_torch.config import Config


def get_config():
    c = Config()
    c.seed = 0
    c.res = 224
    c.global_batch = 1_024
    c.accum_steps = 4               # the original run averaged the gradients of 4 fetches of 256 images per update
    c.total_steps = 4_883
    c.warmup_steps = 244
    c.base_lr = 1e-4                # predictor and head
    c.encoder_lr_ratio = 0.1        # encoder: 1e-5
    c.weight_decay = 0.05
    c.b1, c.b2 = 0.9, 0.99
    c.clip_norm = 1.0
    c.stage1_ckpt = "REQUIRED: <stage1b workdir>"                # student init and frozen teacher
    c.token_stats = "assets/alps_stage1b_token_stats.npz"        # per-channel mean / std of the teacher tokens
    c.empty_token = "assets/flan_t5_base_empty_instruction.npz"  # u0, the constant key/value token
    c.empty_token_key = "u0_bf16"                                # the value used by ALPS (bfloat16 T5 activations)
    c.encoder = Config(width=768, depth=12, mlp_dim=3072, num_heads=12, patch_size=16)
    c.predictor = Config(width=512, depth=4, num_heads=8, mlp_ratio=4, text_dim=768)
    c.crossview = Config(b_scale=(0.4, 1.0), a_local=(0.1, 0.4), a_global=(0.4, 1.0), p_local=0.5, min_valid=16,
                         tries=10, photometric=True)
    c.precision = "float32"         # or "bfloat16" (the ALPS run), see configs/stage1a.py
    c.attn_impl = "explicit"
    c.allow_tf32 = False
    c.remat = False
    # data (not released): [(source, weight), ...]; ALPS used image-editing records (0.95) and a small pool of
    # physical-state edits (0.05), one image of each pair drawn per example.
    c.data = Config(sources=(("REQUIRED", 1.0),), shuffle_buffer=2_000, num_workers=8)
    c.log_steps = 25
    c.ckpt_steps = 2_500
    c.ckpt_keep = 3
    c.ckpt_final = True              # also save after the last update
    return c
