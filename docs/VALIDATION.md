# Validation of the release

What was checked before the release, with the released code and weights. Paper values are those of the ALPS paper's
tables; "paper run" values are the unrounded outputs of the runs behind them.

## Weights

| Check | Result |
|---|---|
| `alps_vitb16.safetensors` = Stage-2 encoder of ALPS + the Stage-1 language interface; `alps_stage1_vitb16.safetensors` = Stage 1 | converted from the training checkpoints (`tools/export_flax_params.py`, `train_jax/export_encoder.py`, `tools/convert_checkpoint.py`) |
| PyTorch (`alps`) vs the float32 Flax forward of the training code, released weights | patch tokens: max relative error 1.7e-5, cosine 1.0000000; text embeddings: 1e-6 |
| The PyTorch ports used for the paper's evaluations vs the released files (`tools/check_legacy_port.py --release`) | the same 148 tensors, bit for bit, for both models |
| Arithmetic of those ports (LayerNorm epsilon 1e-5; training and release: 1e-6) | float32 patch tokens within a cosine of 0.99999 of the release encoder's on COCO images |

## Inference package (`alps/`)

* **Tokenizer.** The pure-Python `ALPSTokenizer` gives the ids of the TensorFlow Text tokenizer used in training on all
  191,010 texts of the language evaluation (COCO and Flickr30k test captions, ImageNet prompts of both template sets).
* **Position tables.** `pos_mode="interpolate"` evaluates the sine-cosine table at scaled positions (the protocol of the
  video benchmarks) and equals the training table on the 14 x 14 grid (tests).
* **Packaging.** The wheel built by `pip install .` carries the package data (vocabulary, zero-shot class names and
  templates); installed outside the source tree, `alps.load`, the tokenizer and the zero-shot classifier work.

## Evaluation (`eval/`)

| Benchmark | Release code | Paper |
|---|---|---|
| ImageNet-1k zero-shot top-1 / top-5, ALPS | 73.436 / 93.878 | 73.436 / 93.878 |
| Flickr30k 1k i2t / t2i R@1, ALPS | 93.20 / 78.92 | 93.20 / 78.92 |
| COCO 5k i2t / t2i R@1, ALPS | 70.20 / 51.892 | 70.20 / 51.892 |
| ImageNet-1k zero-shot top-1 / top-5, ALPS (Stage 1) | 74.452 / 94.354 | 74.454 / 94.352 (one image of 50,000 each) |
| Flickr30k, COCO, ALPS (Stage 1) | identical | |
| Reconstruction, ImageNet split, ALPS decoder of the paper: l1 / LPIPS | .1202 / .3723 | paper run .12015 / .37232 |
| LIBERO state probe, ALPS (Stage 1), Object, read-out position error | 1.0362 cm | paper run 1.0315 cm (the probe training is not bit-deterministic on GPU) |
| DAVIS-2017 val J&F, ALPS, float32 | .5345 | .535 |
| DAVIS-2017 val J&F, ALPS, bfloat16 | .5359 | .533 |

* Language: same weights, prompts and protocol as the paper (`eval/language/`), float32, one GPU.
* LIBERO probe: the token extractor reproduces the cached tokens of the paper's run at a cosine of at least 0.999998
  (with `--tf32 conv`, the convolution setting of that run); the probe was trained on the same tokens with the same seed.
* DAVIS in bfloat16: the release encoder computes its patch embedding in float32 with LayerNorm epsilon 1e-6, the port of
  the paper in bfloat16 with 1e-5; bfloat16 DAVIS moves at the 1e-3 level with such changes (the paper's code gives .533
  with torch 2.3.0 and .540 with torch 2.5.1). In float32 the two agree.
* SSv2, world models: not re-trained. The SSv2 probe (initialization, forward, learning-rate schedule) and the arguments
  it passes to the V-JEPA 2 data pipeline equal those of the original script on test inputs; the world-model
  configurations equal the 30 configurations of the paper's runs (6 encoders x 5 tasks) key by key apart from paths and
  run names, and the DINO-WM and LeWorldModel patches apply to the pinned commits.
* LIBERO policies: `scripts/analyze_policies.py --check` recomputes all 560 policy numbers printed in the paper from the
  per-rollout records in `eval/libero_policy/paper_records/` (also in the Python 3.8 environment of the policies). A real
  installation (LIBERO 8f1084e3 patched by `setup_libero.sh`, that environment, one RTX A5000) trained an ALPS policy on
  LIBERO-Object for one epoch with `train_policy.sh` (8.8M trainable parameters, a 35 MB checkpoint without the frozen
  encoder) and ran `evaluate_policy.py` on it with 2 rollouts per task. The eight C-RADIOv3-B policies of
  Table 2 were trained and evaluated with this release code (frozen network in eval mode); their records replace those of the
  earlier runs.
* Released LIBERO policies (the eight of Table 2, `libero_policies/` on Hugging Face): `export_policy_weights.py` wrote each
  from its training checkpoint (81 tensors, 8,823,979 values; 398 frozen tensors dropped; read back bit for bit). Built
  from their `config.json` exactly as `evaluate_policy.py` builds them, each loads with `strict=True`, runs a forward pass,
  and its frozen encoder, `alps_vitb16.safetensors` with the settings of the paper's runs (`alps_port_numerics`), equals
  the frozen network stored in the checkpoint bit for bit: weights, position table, and normalization constants.

## Training code

* **JAX (`train_jax/`).** Against the code that trained ALPS (CPU, small models; that code is not public): identical
  parameter trees and initial values, encoder outputs in float32 and bfloat16, the loss and gradients of Stage 1 (bfloat16
  compute, as trained) and of Stage 2 (float32 and bfloat16), optimizer updates, and data transforms for fixed seeds.
  End-to-end run of Stage 1a -> 1b -> Stage 2 -> export on synthetic data. The Stage-2 constant token
  (`assets/flan_t5_base_empty_instruction.npz`) is rebuilt bit for bit from google/flan-t5-base by
  `tools/compute_t5_empty_token.py`.
* **PyTorch (`train_torch/`).** Against `train_jax` on CPU: every module, loss term and gradient to about 1e-5 relative in
  float32, optimizer updates against optax, the image transforms against TensorFlow for identical random draws,
  multi-process training (2 and 4 processes) against one process, interrupted and resumed runs bit for bit. On two RTX
  A5000 GPUs (NCCL, bfloat16, full ViT-B/16): Stage 1 -> Stage 2 -> export; the exported files load with `alps.load`.

## Test suites

| Suite | Tests | Environment |
|---|---|---|
| `tests/` | 15 | CPU, torch 2.5.1 (`test_tokenizer.py` against TensorFlow Text needs `tensorflow-text`) |
| `eval/tests`, `eval/world_models/tests`, `eval/libero_policy/tests`, `eval/video/tests` | 3, 34, 38, 17 | CPU; the configuration test needs `omegaconf`, the V-JEPA 2 tests transformers >= 4.53 |
| `train_jax/tests` | 3 | CPU, jax 0.10, flax 0.12, TensorFlow 2.21 |
| `train_torch/tests` | 65 | CPU, torch 2.5.1 with jax / flax / optax for the parity tests; a TensorFlow reference file for the transform tests |

Every suite passes with the optional references present (Flax reference forward, TensorFlow transforms, the flan-T5
weights). The tests that need `tensorflow-text`, `omegaconf` or transformers >= 4.53 ran in a Linux environment that has
them (Python 3.9, torch 2.3.0). The comparisons with the original research code (training code and video scripts) were
run before the release and are not part of the test suites, since that code is not public.
