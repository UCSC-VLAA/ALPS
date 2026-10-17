# ALPS training code (JAX / Flax)

This is the code ALPS was trained with, reduced to the paths the released model used. Module and parameter names,
initializers, numerics (casts, float32 islands, softmax precision), losses, optimizers and data transforms are those
of the original runs. Before the release it was checked against the original code base (identical parameter trees
and initial values, identical forward passes, losses and gradients to float rounding, identical optimizer updates and
data transforms for fixed seeds; see `../docs/VALIDATION.md`).

## Recipe

| | Stage 1a | Stage 1b | Stage 2 |
|---|---|---|---|
| config | `configs/stage1a.py` | `configs/stage1b.py` | `configs/stage2.py` |
| resolution | 128 | 224 | 224 |
| global batch | 32,768 | 16,384 | 1,024 (4 x 256, gradients averaged) |
| updates | 390,981 | 62,556 | 4,883 |
| warm-up, then cosine to 0 | 3,909 | 3,127 | 244 |
| peak learning rate | 1.024e-3 | 2.56e-5 | 1e-4 (predictor), 1e-5 (encoder) |
| Adam (b1, b2), first moment stored in bf16 (optax `mu_dtype`) | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.99 |
| weight decay (kernels only) | 0.05 | 0.05 | 0.05 |
| gradient clipping | none | none | global norm 1.0 |
| initialization | from scratch | all Stage-1a parameters, fresh optimizer | encoder from Stage 1b; frozen Stage-1b teacher |

Stage 1 loss: `L_itc + L_cap + w_rec L_rec + 0.25 L_dense` with `w_rec` = 2 (1a) and 1 (1b): multi-positive
image-text contrastive loss (web caption and one sentence of the synthetic caption, learned scale bounded to
[1, 100]), prefix-LM caption prediction of the synthetic caption, pixel reconstruction from the tokens and from noisy
tokens (l1 + 0.2 (1 - SSIM) + 0.1 Sobel), and patch InfoNCE between a view and a crop + flip of it (two softmaxes,
weights 0.75 / 0.25, temperature 0.1). Stage 2 loss: squared error between the predictor's output and the
standardized frozen Stage-1 tokens of the clean crop, averaged over the patches of B inside A, then over images.

## Data

The training data are not released. `alps_jax/data/pipelines.py` documents the expected inputs: TFRecords (or a TFDS
directory) of images with a web caption and a synthetic caption for Stage 1; TFRecords of single images (or image
pairs, one image drawn per example) for Stage 2. ALPS used DataComp-1B with the LLaVA-LLaMA-3 captions of
Recap-DataComp-1B for Stage 1, and 5 million draws from a curated collection of public image-editing datasets for
Stage 2 (images only).

## Running

```bash
pip install -r requirements.txt
python train_stage1.py --config=configs/stage1a.py --workdir=runs/stage1a --config.data.files='<pattern>'
python train_stage1.py --config=configs/stage1b.py --workdir=runs/stage1b --config.data.files='<pattern>' \
    --config.init_from=runs/stage1a/checkpoints
python train_stage2.py --config=configs/stage2.py --workdir=runs/stage2 --config.stage1_ckpt=runs/stage1b/checkpoints \
    --config.data.sources="(('<pattern>', 1.0),)"
```

Training is data-parallel over all devices (on a multi-host TPU slice, run the same command on every host). The global
batch must be divisible by the device count; scale the learning rate linearly when you change the batch.
`assets/` holds the two Stage-2 constants of ALPS: the per-channel statistics of the Stage-1b tokens
(`alps_stage1b_token_stats.npz`; recompute for another Stage-1 model with `compute_token_stats.py`) and the
flan-T5-base encoding of the empty instruction that serves as the predictor's constant key/value token
(`flan_t5_base_empty_instruction.npz`; `tools/compute_t5_empty_token.py` recomputes it bit for bit from the
google/flan-t5-base checkpoint, which needs `safetensors` and `huggingface_hub`).

## From checkpoints to the PyTorch model

```bash
python ../tools/export_flax_params.py --ckpt runs/stage1b/checkpoints --out stage1b_params.npz
python export_encoder.py --ckpt runs/stage2/checkpoints --subtree encoder --out stage2_encoder.npz
python ../tools/convert_checkpoint.py --stage1 stage1b_params.npz --encoder stage2_encoder.npz --out alps_vitb16.safetensors
```

## Tests

```bash
pytest tests/test_smoke_train.py         # Stage 1 -> Stage 2 -> export on CPU with synthetic data
pytest tests/test_t5_empty_token.py      # the constant token (ALPS_T5_WEIGHTS=<flan-t5-base model.safetensors>: rebuild the asset)
```
