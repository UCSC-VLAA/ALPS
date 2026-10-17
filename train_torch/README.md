# ALPS training code (PyTorch)

A PyTorch re-implementation of the JAX/Flax training code in `../train_jax/`: Stage 1 (image-text pretraining,
128 px then 224 px) and Stage 2 (cross-view self-distillation). Model parameters, initializers, numerics, losses,
the optimizer and the data transforms follow the JAX code; `tests/` checks this against `train_jax` on CPU
(forward passes, every loss term and all gradients to ~1e-5 relative in float32, optimizer updates against optax,
the image transforms against TensorFlow for identical random draws). The encoder and text tower are the modules of
the inference package `alps`, so trained weights export directly to the `alps` format.

## Recipe

The configs in `configs/` mirror `train_jax/configs/` (same keys, same values; see the table in
`train_jax/README.md`): Stage 1a 32,768 x 390,981 updates at 128 px, Stage 1b 16,384 x 62,556 at 224 px from all
Stage-1a parameters with a fresh optimizer, Stage 2 1,024 (4 x 256 accumulated) x 4,883.

## Data

The training data are not released. Inputs (`alps_torch/data/datasets.py`):

| stage | source | record |
|---|---|---|
| 1 | `*.jsonl` | `{"image": <path>, "caption": <web caption>, "synthetic_caption": <str or list>}` (keys configurable; relative paths resolve against the JSONL directory) |
| 1 | tar shards (`shards/{000000..004095}.tar`) | WebDataset layout: `<key>.jpg` + `<key>.json` with the same caption keys (or the web caption in `<key>.txt`) |
| 2 | `*.jsonl` | `{"image": <path>}`, or `{"source": <path>, "target": <path>}` (one of the two drawn with p 0.5 per example) |
| 2 | `*.txt` / a directory / a glob / tar shards | image paths, image files, or tar records with one or more images |

Stage 2 mixes several sources per example with weights: `--config.data.sources="(('a.jsonl', 0.95), ('b.jsonl', 0.05))"`.
ALPS used DataComp-1B with the Recap-DataComp-1B LLaVA-LLaMA-3 captions for Stage 1 and 5 million draws from a
collection of public image-editing datasets (images only) for Stage 2.

## Running

```bash
pip install -r requirements.txt      # and the inference package: pip install -e ..
torchrun --nproc_per_node=8 train_stage1.py --config=configs/stage1a.py --workdir=runs/stage1a \
    --config.data.source='/data/s1/{000000..004095}.tar'
torchrun --nproc_per_node=8 train_stage1.py --config=configs/stage1b.py --workdir=runs/stage1b \
    --config.data.source='/data/s1/{000000..004095}.tar' --config.init_from=runs/stage1a
torchrun --nproc_per_node=8 train_stage2.py --config=configs/stage2.py --workdir=runs/stage2 \
    --config.stage1_ckpt=runs/stage1b --config.data.sources="(('/data/s2/images.jsonl', 1.0),)"
python export.py --stage1 runs/stage1b --stage2 runs/stage2 --out alps_vitb16.safetensors   # also written by the trainers
```

```python
import alps
model = alps.load("alps_vitb16.safetensors")       # Stage-2 encoder + Stage-1 language interface
```

* **Global batch.** Losses have the semantics of one global batch under DDP: the contrastive softmax runs over the
  whole batch (embeddings gathered with gradient), the caption loss is divided by the global number of target
  tokens, and the 64 cross-image negatives of the dense loss are one set of global patch indices per step,
  gathered from the processes that own them with gradient. The global batch must be divisible by the number of
  processes; scale the learning rate linearly with the batch.
* **Numerics.** `precision="float32"` (default) is the mode the parity tests check. `precision="bfloat16"` uses the
  compute dtype of the ALPS runs with the cast points of the Flax code (bf16 Dense / Conv / attention, float32
  patch embedding, LayerNorm statistics, softmax of the Stage-2 predictor, losses, parameters and optimizer state;
  in Stage 1 the residual branches are added in float32, in Stage 2 in bf16, as in the original). TF32 is off
  unless `allow_tf32=True`. `attn_impl="sdpa"` swaps the explicit attention for PyTorch's fused kernel (same math in
  float32; in bfloat16 its softmax runs in float32). bfloat16 runs of the two frameworks agree only to bfloat16 rounding
  noise (XLA fuses ops with excess precision, accumulation orders differ), so exact parity is a float32 property.
* **Optimizer.** Adam with the first moment stored in bfloat16 exactly as optax does it (including the bfloat16
  product of the decay with the stored moment), decoupled weight decay on Linear / Conv weights only, warm-up +
  cosine with an exact-zero first step, Stage-2 global-norm clipping without epsilon (`alps_torch/optim.py`).
* **Resume.** A run continues from the newest `<workdir>/checkpoints/ckpt_*.pt` (parameters, optimizer, step); as in
  the JAX code the data stream restarts.
* **Stage-2 constants.** The token statistics and the constant text token of ALPS are read from `../train_jax/assets/`
  (relative paths in `--config.token_stats` / `--config.empty_token` are looked up in the working directory,
  `train_torch/` and then `train_jax/`); `compute_token_stats.py` recomputes the statistics for another Stage-1 model.

## Differences from the JAX pipeline

Exact random streams are not reproduced (the transforms take their uniform draws as arguments and match the TF
functions for identical draws; `alps_torch.data.tf_ops.TFPhiloxRNG` even reproduces TF's stateless Inception-crop
sampler seed for seed). Images are decoded by PIL instead of `tf.io.decode_image`. Shuffling uses per-pass
permutations (JSONL / lists) or a 2,000-record buffer (tar shards) instead of 250,000 / 50,000-record buffers.
Inside the TF resize kernels some builds fuse multiply-adds; `tf_ops.FMA = True` emulates that bit for bit, the
default differs by float rounding.

## Tests

```bash
pytest tests/                                     # CPU; the parity tests need jax + flax + optax next to torch
pytest tests/ -m "not slow"                       # without the full-size ViT-B/16 smoke run
python tests/tf_reference/make_reference.py --out /tmp/tf_ref.npz      # in an environment with TensorFlow
ALPS_TF_REF=/tmp/tf_ref.npz pytest tests/test_transforms_tf.py
```

| test | checks |
|---|---|
| `test_parity_models.py` | every module (ViT, text encoder, caption decoder, pixel decoder, dense projector, predictor, head) vs Flax: outputs, parameter and input gradients |
| `test_parity_losses.py` | the Stage-1 model + loss (both dense-loss forms) and the Stage-2 loss vs Flax with the JAX random draws: outputs, each loss term, all gradients; bfloat16 mode |
| `test_parity_optim.py`, `test_parity_train_step.py` | the optimizer, schedule, clipping and logit-scale projection vs optax; complete training steps of both stages |
| `test_init.py` | initializer distributions vs the Flax initializers; parameter counts |
| `test_ddp.py` | 2 and 4 CPU processes (gloo) vs one process on the whole batch |
| `test_data.py`, `test_transforms_tf.py` | transforms, caption fields, input pipelines; the transforms vs TensorFlow |
| `test_smoke_train.py` | Stage 1a -> resume -> 1b -> Stage 2 -> export on synthetic data; `alps.load` of a full-size export |
| `test_checkpoint.py` | an interrupted and resumed run equals the uninterrupted one bit for bit |

License: Apache-2.0 (see `../LICENSE`).
