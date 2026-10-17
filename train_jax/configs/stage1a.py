# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1a of ALPS: 128 pixels, from scratch (12.8 billion samples)."""
import ml_collections


def get_config():
    c = ml_collections.ConfigDict()
    c.seed = 0
    c.res = 128
    c.global_batch = 32_768
    c.total_steps = 390_981
    c.warmup_steps = 3_909
    c.peak_lr = 1.024e-3            # 8e-6 per 256 images x 128 (the original config: base_lr * 64 * batch_factor)
    c.weight_decay = 0.05
    c.b1, c.b2 = 0.9, 0.95
    # loss weights and the dense objective
    c.cap_w, c.rec_w, c.dense_w = 1.0, 2.0, 0.25
    c.dense_radius = 0              # 8 x 8 grid: only the positive's own cell is excluded ...
    c.dense_n_same = 0              # ... and all 63 other locations are negatives (0 = all)
    c.dense_n_cross = 64
    c.dense_split = (0.75, 0.25)
    c.init_from = ""                # Stage 1b: the Stage-1a checkpoint directory
    # model
    c.model = ml_collections.ConfigDict(dict(width=768, depth=12, mlp_dim=3072, num_heads=12, patch_size=16, remat=True,
                                             vocab_size=32_000, text_width=768, text_depth=12, text_heads=12,
                                             text_mlp_dim=3072, itc_len=32, caption_len=80, caption=True,
                                             cap_depth=12, cap_heads=12, cap_mlp_dim=3072))
    # data (not released; see alps_jax/data/pipelines.py for the expected format)
    c.data = ml_collections.ConfigDict(dict(files="", tfds_dir="", tfds_split="full", image_key="image",
                                            web_key="caption", synthetic_key="synthetic_caption",
                                            shuffle_buffer=250_000))
    c.log_steps = 100
    c.ckpt_steps = 2_000
    c.ckpt_keep = 3
    return c
