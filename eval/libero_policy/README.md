# LIBERO policy learning on frozen encoder tokens

Code of the policy experiments of the ALPS paper: LIBERO's transformer policy (BC-Transformer) trained on the frozen
patch tokens of an encoder, on LIBERO-Long (LIBERO-10), -Goal, -Spatial and -Object.

| Paper | What | Here |
|---|---|---|
| Tab. 2 | success of 7 encoders with BERT instructions and of 4 encoders with their own text tower | `train_policy.sh`, `evaluate_policy.sh`, `analyze_policies.py` |
| Tab. S8, S9 | 95% intervals, worst task, tasks >= 80; success of every task | `analyze_policies.py` |
| Tab. S10 | settings of the policies | this README, `frozen_encoder_pool.yaml` (in the patch) |
| Sec. S3 | SigLIP 2-B/16 refined by Stage 2 (Long, Goal, Spatial) | `train_policy.sh siglip2_b16` + `ENCODER_WEIGHTS` |
| Sec. S2 | 3 trainings per suite of ALPS with BERT and with its own text tower | `paper_policies.tsv`, `analyze_policies.py` |
| Fig. (rollouts) | closed-loop rollouts of the ALPS policies | `record_rollouts.py`, `make_rollout_figure.py` |

```
eval/libero_policy/
  frozen_encoder_bridge.py        FrozenEncoderPool: frozen encoder (through eval/encoders.py) + attention pool, LIBERO image module
  patches/libero_8f1084e3.patch   changes to LIBERO 8f1084e3 (+ the yaml and the import module of the bridge)
  patches/robomimic-0.2.0_dataset.patch
  paper_policies.tsv              the 63 policies of the paper: encoder, suite, seed, epochs, instructions, value in the paper
  paper_records/                  the records of those policies: configs, the outcome of every evaluation rollout, embeddings
  paper_numbers.json              every policy number printed in the paper (Tab. 2, S8, S9, Sec. S2, S3), for --check
  scripts/
    setup_libero.sh               clone LIBERO at the paper's commit, apply the patches, write LIBERO's path config
    download_libero_data.sh       the demonstrations of the four suites (official hdf5, Hugging Face)
    train_policy.sh               train one policy
    evaluate_policy.py / .sh      the three evaluations of a trained policy (50 rollouts per task each)
    export_policy_weights.py      a policy checkpoint -> safetensors of its trainable weights (the released policies)
    run_paper_policies.sh         train + evaluate the rows of paper_policies.tsv that match a pattern
    dump_libero_instructions.py   the 40 instructions and their BERT embeddings (LIBERO environment)
    embed_instructions.py         own-text-tower instruction embeddings: ALPS, CLIP, SigLIP 2, C-RADIOv3
    analyze_policies.py           tables from the per-rollout records
    record_rollouts.py, make_rollout_figure.py
  tests/                          CPU tests (no LIBERO needed): pytest eval/libero_policy/tests
```

## Setup

**Environment of the paper's runs** (one conda environment for training and evaluation): Python 3.8.20, torch 2.4.1 and
torchvision 0.19.1 (CUDA build), timm 1.0.29, transformers 4.21.1, robosuite 1.4.0, robomimic 0.2.0 (patched, below),
mujoco 3.2.3, bddl 1.0.1, gym 0.25.2, hydra-core 1.2.0, numpy 1.22.4, h5py 3.11.0, einops 0.4.1, easydict 1.9, thop. One
NVIDIA RTX A5000 (Ampere) per run; rendering with EGL. The Python 3.8 environment also needs `pip install safetensors`: the
released ALPS weights are safetensors files (the runs read earlier PyTorch ports of the same weights).

```bash
export LIBERO_ROOT=$PWD/LIBERO LIBERO_DATA=$PWD/libero_datasets ALPS_REPO=<this repository>
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_ROOT
git -C $LIBERO_ROOT checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
conda create -n libero_env python=3.8 -y && conda activate libero_env
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r $LIBERO_ROOT/requirements.txt && pip install -e $LIBERO_ROOT     # LIBERO's pins (robosuite 1.4.0, robomimic 0.2.0, ...)
pip install timm==1.0.29 safetensors                     # safetensors: the released ALPS weights
bash $ALPS_REPO/eval/libero_policy/scripts/setup_libero.sh          # patches LIBERO and robomimic, writes ~/.libero/config.yaml
bash $ALPS_REPO/eval/libero_policy/scripts/download_libero_data.sh  # LIBERO-10 / Goal / Spatial / Object hdf5 into $LIBERO_DATA
```

`setup_libero.sh` applies `patches/libero_8f1084e3.patch` with `git apply`, writes LIBERO's path config (`$LIBERO_CONFIG_PATH`,
default `~/.libero/config.yaml`, so that LIBERO does not prompt) and applies `patches/robomimic-0.2.0_dataset.patch` to the
installed robomimic. The demonstrations come from the Hugging Face dataset `yifengzhu-hf/LIBERO-datasets` (the official
`benchmark_scripts/download_libero_datasets.py --use-huggingface`; LIBERO-10 with LIBERO's own Hugging Face helper, because
the official script only offers `libero_100`); the fixed initial states of the evaluation (`init_files/*.pruned_init`) are part
of the LIBERO repository. LIBERO downloads `bert-base-cased` on first use (into `$LIBERO_ROOT/bert`).

**Encoders.** The bridge builds every encoder through `eval/encoders.py`:

| Paper row | `encoder` | weights | the paper's run configs (`encoder_ckpt`) |
|---|---|---|---|
| ALPS | `alps` | `alps_vitb16.safetensors` (`ALPS_WEIGHTS`) | torch port `.pt` of the same weights |
| ALPS (Stage 1) | `alps_stage1` | `alps_stage1_vitb16.safetensors` (`ALPS_STAGE1_WEIGHTS`) | torch port `.pt` |
| CLIP-B/16 | `clip_b16_qg` | timm `vit_base_patch16_clip_quickgelu_224.openai` | `timm:` the same |
| SigLIP 2-B/16 | `siglip2_b16` | timm `vit_base_patch16_siglip_224.v2_webli` | `timm:` the same |
| SigLIP 2-B/16 + Stage 2 (Sec. S3) | `siglip2_b16` | a full timm state dict of the refined tower (`SIGLIP2_STAGE2_WEIGHTS`; not released) | state-dict path |
| DINOv2-B/14 | `dinov2_b14_timm` | timm `vit_base_patch14_dinov2.lvd142m` (not the torch-hub `dinov2_b14`) | `timm:` the same |
| DINOv3-B/16 | `dinov3_b16` | timm `vit_base_patch16_dinov3.lvd1689m` (DINOv3 license; Hugging Face may ask to accept it) | `timm:` the same |
| C-RADIOv3-B | `cradio_v3_b` | `c-radio_v3-b_half.pth.tar` of `nvidia/C-RADIOv3-B` (`RADIO_CKPT`); code: a clone of github.com/NVlabs/RADIO (`RADIO_REPO`; the runs used a copy of the main branch of 2026-09-27, recorded as commit c0f37017) | `radio:<checkpoint>` |

## The frozen-encoder module (`frozen_encoder_bridge.py`)

`FrozenEncoderPool` replaces LIBERO's `ResnetEncoder` inside `BCTransformerPolicy` (one instance per camera; LIBERO creates it
by the name in `libero/configs/policy/image_encoder/frozen_encoder_pool.yaml`). Per camera and frame:

1. LIBERO's float image in [0, 1], 128 x 128 (agent view or eye in hand), checked on the first batch;
2. `flip_ud`: `torch.flip(x, dims=[-2])`. LIBERO stores and renders its frames upside down; every policy of the paper shows
   the encoder upright frames (in training and in the rollouts);
3. bilinear resize to 224 x 224 (`align_corners=False`, no antialiasing);
4. the encoder's normalization (`FrozenEncoder.normalize`; C-RADIOv3 normalizes inside, its input is clamped to [0, 1]);
5. the frozen forward under `torch.no_grad()` and bfloat16 autocast on CUDA (float32 weights; no autocast on the CPU); the
   native last-layer patch tokens of `eval/encoders.py`: 196 per frame (DINOv2-B/14 at 224 px: 256; DINOv3: CLS and 4 register
   tokens dropped; C-RADIOv3: backbone spatial features, no adaptor), cast to float32;
6. the trainable attention pool: query = `Linear(L, D)` of the task embedding (L = 768 for BERT), `nn.MultiheadAttention(D,
   4 heads)` over the tokens (keys = values, no position encoding added), then `Linear(D, 64)`: one 64-d token per camera,
   the size of LIBERO's own image token.

Configuration (`policy.image_encoder.network_kwargs.*`; the yaml holds the defaults):

| key | default | paper's runs | meaning |
|---|---|---|---|
| `encoder`, `weights` | | see the table above | the frozen encoder |
| `flip_ud` | `true` | `true` | show the encoder upright frames |
| `resize` | `224` | `224` | input side of the encoder |
| `autocast_dtype` | `bfloat16` | `bfloat16` | autocast of the frozen forward on CUDA (`none`: float32) |
| `num_queries`, `pool_heads` | `1`, `4` | `1`, `4` | attention pool |
| `encoder_train_mode` | `eval` | `follow` (C-RADIOv3-B: `eval`) | `eval`: the frozen network stays in eval mode; `follow`: it follows the policy's train / eval mode (below) |
| `alps_port_numerics` | `false` | `true` (ALPS rows) | `true`: ALPS encoders compute like the port of the paper's runs (below); no effect on other encoders |
| `language_fusion` | `film` | `film` | unused, kept for the config surface of LIBERO's image encoders |

The defaults are a truly frozen encoder and the released model's arithmetic, as in every other benchmark of this repository;
`scripts/run_paper_policies.sh` passes the settings of the paper's runs explicitly
(`policy.image_encoder.network_kwargs.encoder_train_mode=follow` for every policy except C-RADIOv3-B, whose policies were
trained with `eval`; `...alps_port_numerics=true` for ALPS).

The two settings of the paper's runs:

* **Train mode of the frozen network** (`encoder_train_mode: follow`; C-RADIOv3-B: `eval`). In the runs the frozen network
  was a registered submodule of the policy, so LIBERO's `policy.train()` put it in training mode for the training epochs, and
  the evaluation ran it in eval mode. For ALPS and the timm encoders (CLIP, SigLIP 2, DINOv2, DINOv3) the mode changes
  nothing: they have no training-mode randomness. C-RADIOv3-B computes differently in training mode: its cropped position
  embedding (CPE) takes a random crop of the position-embedding grid for every image and drops the position embedding for 10%
  of the images. The paper's eight C-RADIOv3 policies were therefore trained with the release code and the frozen network in
  eval mode (`encoder_train_mode=eval`), so that training and evaluation see the same tokens.
* **ALPS arithmetic** (`alps_port_numerics: true`). ALPS and ALPS (Stage 1) ran through an earlier PyTorch port of the
  encoder, with the released weights but LayerNorm epsilon 1e-5 (the training value is 1e-6), the patch embedding inside the
  bfloat16 autocast (the release ViT keeps it in float32), a float64 sine-cosine table and normalization constants computed as
  float32(255 c) / 255. The switch applies this arithmetic to the release encoder; on the CPU it reproduces the port's tokens
  bit for bit, in float32 and under bfloat16 autocast (tests). With `false` the release encoder runs as is: its float32 tokens
  are within a cosine of 0.99999 of the port's, while under the policy's bfloat16 autocast they differ by about as much as
  bfloat16 rounding itself.

Behaviour of the runs that the module always keeps:

* **TF32.** Importing the module (LIBERO's `bc_transformer_policy.py` does, in every process) sets
  `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True`, so the float32 policy also
  ran its matrix products in TF32.
* **Construction order.** The frozen network is built first, then the pool (`query_proj`, `pool`, `out_proj`), so that the
  trainable initial weights are the same function of the training seed; the release ALPS ViT draws exactly as many random
  numbers during its (overwritten) initialization as the port did (tested).

One difference: the frozen network is not a registered submodule. Policy checkpoints hold the 8.8M trainable parameters only
(about 35 MB instead of 380 MB), the frozen network moves to the GPU at the first forward, and LIBERO's printed
"GFLOPs / MParams" count the trainable part only (its counter also skips the two `nn.MultiheadAttention` pools: it prints
4.1 MParams for the 8.8M trainable parameters). Checkpoints of the research code (which contain the frozen weights as
`*.vit.*` and the normalization as `*.im_mean`, `*.im_std`) load as well; the frozen weights in them are ignored.

## Settings (Tab. S10)

LIBERO's defaults, except where marked; every encoder and instruction embedding of a suite uses the same settings and seed.

| | |
|---|---|
| observation | agent view and eye in hand at 128 x 128 (upright for the encoder); joint positions (7) and gripper state (2) |
| language | BERT-base (cased) pooled output, frozen, max 25 tokens; or an encoder's own text tower (below) |
| policy | `BCTransformerPolicy`: 4 layers, 6 heads, width 64, MLP 256, dropout 0.1, 10-step context, GMM head with 5 modes; 8.8M trainable parameters for every encoder |
| image encoder | **frozen encoder + attention pool** instead of LIBERO's ResNet with FiLM trained end to end; the frozen network in training mode during the training epochs, like the policy (`encoder_train_mode=follow`); for C-RADIOv3-B in eval mode (`encoder_train_mode=eval`) |
| optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 1e-4; cosine schedule per epoch to 1e-5, no warm-up; gradient clipping 100 |
| batch, data | 32 windows of 10 steps; 50 demonstrations x 10 tasks, all used for training; 4 data-loader workers |
| augmentation | color jitter (p 0.9) and a random shift of up to 4 px, the same draw for both cameras and the 10 frames |
| epochs | **25 on LIBERO-Long** (LIBERO: 50), 50 on Goal, Spatial, Object; no zero-shot loss pass at epoch 0 (`LIBERO_SKIP_EP0=1`); no evaluation during training |
| training seed | **LIBERO-Long seed 1, the other suites seed 0** (Tab. 2, Sec. S3); ALPS retrainings of Sec. S2: seeds 0, 1, 2 |
| evaluation | **the last checkpoint, evaluated three times** (LIBERO: the best of several checkpoints): 50 rollouts per task, one per fixed initial state, at most 600 steps |

The training seed sets the Python, NumPy and PyTorch generators and `cudnn.deterministic` (LIBERO's `control_seed`).
Training is still not bit-reproducible across machines and library versions, and repeated trainings of one policy differ by
several points (ALPS, BERT, LIBERO-Long, seeds 0 / 1 / 2: 63.5 / 69.7 / 75.3).

## Training

```bash
export LIBERO_ROOT ALPS_WEIGHTS=<alps_vitb16.safetensors> GPU=0
bash scripts/train_policy.sh alps 10 1            # LIBERO-Long, seed 1, 25 epochs
bash scripts/train_policy.sh alps goal 0          # LIBERO-Goal, seed 0, 50 epochs (spatial, object likewise)
bash scripts/train_policy.sh dinov3_b16 spatial 0
ENCODER_WEIGHTS=$RADIO_CKPT bash scripts/train_policy.sh cradio_v3_b object 0
bash scripts/train_policy.sh alps 10 1 task_embeddings/alps_task_embs.npz      # instructions from ALPS's own text tower
```

This runs `libero/lifelong/main.py` with `benchmark_name=<suite> policy=bc_transformer_policy lifelong=multitask
policy/image_encoder=frozen_encoder_pool`, the encoder overrides, `train.n_epochs=<25|50> eval.eval_every=<same>
lifelong.eval_in_train=false eval.eval=false train.num_workers=4 eval.num_workers=0`, and `LIBERO_EXP_SUFFIX=<run tag>
LIBERO_SKIP_EP0=1`, with the module's defaults (a frozen encoder in eval mode, the released arithmetic); add
`EXTRA_OVERRIDES='policy.image_encoder.network_kwargs.encoder_train_mode=follow policy.image_encoder.network_kwargs.alps_port_numerics=true'`
for the settings of the paper's runs (C-RADIOv3-B: the defaults). A policy took 16-27 hours on one RTX A5000.
`run_paper_policies.sh '<pattern>' <gpu>` trains (with the paper's settings) and evaluates the rows of `paper_policies.tsv`
that match, e.g. `'Table 2 \(BERT\)'` (28 policies) or `'.'` (all 63).

## Evaluation

```bash
bash scripts/evaluate_policy.sh $LIBERO_ROOT/experiments/LIBERO_10/Multitask/BCTransformerPolicy_seed1_alps_10_e25_s1/run_001 0
```

runs `evaluate_policy.py` three times, as for every policy of the paper: `eval50` and `repA` with no seed set, `repB` with
`--eval-seed 1` (python, numpy and torch seeded after the policy is loaded). Each evaluation loads the last checkpoint
(`multitask_model_ep25.pth` on Long, `ep50` otherwise) and runs LIBERO's `evaluate_one_task_success`: 50 rollouts per task,
one from each fixed initial state (reset, initial state, 5 zero-action steps, at most 600 steps, success when the goal
predicates hold), actions sampled from the policy's Gaussian mixture, 10 parallel simulators on LIBERO-10 and 20 otherwise.
The simulator is not seeded, so the three evaluations differ only in the sampled actions (e.g. ALPS (Stage 1), Long:
69.4 / 67.2 / 70.0). Output: `eval50.json`, `eval50_repA.json`, `eval50_repB.json` in the run directory, each with
`per_task` and `per_rollout[task]` (50 outcomes, index = initial state). An evaluation takes roughly an hour on one GPU.
Run directories from another machine: `--datasets/--bddl/--init-states` (default: LIBERO's config when the stored folders
do not exist), `--encoder-weights`, `--task-embeddings`. The configurations of the paper's runs (`paper_records/raw/*/config.json`,
which name the encoder in `encoder_ckpt`) evaluate as well, with the settings of those runs: the encoder is
recognized, its weights come from `--encoder-weights` or `ALPS_WEIGHTS` / `ALPS_STAGE1_WEIGHTS` / `RADIO_CKPT`, the instruction
embeddings from `--task-embeddings` or `TASK_EMBEDDINGS_DIR` (e.g. `paper_records/raw/task_embeddings`).

## Released policies (Hugging Face)

The eight ALPS policies of Tab. 2 (LIBERO's BERT instructions and ALPS's own text tower, four suites each) are released
in the folder `libero_policies/` of the Hugging Face repository `UCSC-VLAA/ALPS` that holds the ALPS weights (published
with the paper). Each policy is one directory, named by its suite and instruction embedding, with the run's `config.json` (from
`paper_records/raw`; only `experiment_dir` and `experiment_name` follow the new directory name) and `multitask_model_ep<N>.safetensors`, its 8.8M trainable parameters (35 MB); the frozen encoder is
`alps_vitb16.safetensors` of the same repository, the encoder of the runs.

| `libero_policies/...` | instructions | Tab. 2 |
|---|---|---|
| `LIBERO_10/bert` | BERT | 69.7 |
| `LIBERO_GOAL/bert` | BERT | 85.1 |
| `LIBERO_SPATIAL/bert` | BERT | 86.5 |
| `LIBERO_OBJECT/bert` | BERT | 92.2 |
| `LIBERO_10/alps_text` | ALPS text tower | 78.5 |
| `LIBERO_GOAL/alps_text` | ALPS text tower | 87.4 |
| `LIBERO_SPATIAL/alps_text` | ALPS text tower | 82.6 |
| `LIBERO_OBJECT/alps_text` | ALPS text tower | 93.7 |

```bash
huggingface-cli download UCSC-VLAA/ALPS alps_vitb16.safetensors --local-dir alps_hf
huggingface-cli download UCSC-VLAA/ALPS --include "libero_policies/*" --local-dir alps_hf
export LIBERO_ROOT ALPS_WEIGHTS=$PWD/alps_hf/alps_vitb16.safetensors \
       TASK_EMBEDDINGS_DIR=$ALPS_REPO/eval/libero_policy/paper_records/raw/task_embeddings
bash scripts/evaluate_policy.sh alps_hf/libero_policies/LIBERO_OBJECT/bert 0
```

`evaluate_policy.py` reads `multitask_model_ep<N>.safetensors` of a run directory (else LIBERO's `.pth`) and evaluates these
configurations with the settings of the paper's runs (above). The rollouts sample their actions from the policy and the
simulator is not seeded, so a new evaluation differs from each of the paper's three evaluations of a policy about as much as
they differ from one another; Tab. 2 gives their mean. The files were written from the checkpoints of the runs by
`scripts/export_policy_weights.py` (frozen network dropped, every other tensor unchanged).

## Instructions from an encoder's own text tower

The lower block of Tab. 2 replaces the BERT embedding by the encoder's projected text embedding (before L2 normalization),
times one global factor so that its mean norm over the 40 instructions equals that of the BERT embeddings (21.533).

```bash
# 1. LIBERO environment: the 40 instructions and LIBERO's BERT embeddings of them
cd $LIBERO_ROOT && PYTHONPATH=. python $ALPS_REPO/eval/libero_policy/scripts/dump_libero_instructions.py --out-dir task_embeddings
# 2. a recent environment (paper: Python 3.11, torch 2.5.1, transformers 5.14, open_clip 3.3): one file per encoder
python scripts/embed_instructions.py --model alps --alps-weights alps_vitb16.safetensors \
    --instructions task_embeddings/libero_instructions.json --bert task_embeddings/bert_pooler_libero.npz --out task_embeddings/alps_task_embs.npz
python scripts/embed_instructions.py --model clip    ... --out task_embeddings/clip_task_embs.npz
python scripts/embed_instructions.py --model siglip2 ... --out task_embeddings/siglip2_task_embs.npz
RADIO_REPO=... RADIO_CKPT=... python scripts/embed_instructions.py --model cradio ... --out task_embeddings/cradio_task_embs.npz
```

| `--model` | text tower | dim | scale |
|---|---|---|---|
| `alps` | ALPS text tower + text projection (Stage 1; Stage 2 leaves them unchanged), `ALPSTokenizer` | 768 | 0.51683 |
| `clip` | open_clip `hf-hub:timm/vit_base_patch16_clip_224.openai`, `encode_text` | 512 | 2.31931 |
| `siglip2` | transformers `google/siglip2-base-patch16-224`, `get_text_features`, lower-cased, padded to 64 | 768 | 1.17740 |
| `cradio` | the text tower of C-RADIOv3's `clip` adaptor (DFN CLIP ViT-H/14-378, its language interface in Tab. 1) | 1024 | 30.44498 |

The embeddings of the paper are in `paper_records/raw/task_embeddings/*_task_embs.npz`; `--reference <file>` compares a new
file with them (ALPS with the release package: max relative difference 1.3e-6, token ids
identical). Train with the file as the fourth argument of `train_policy.sh` (`task_embedding_format=file:<npz>`); the
query projection of the pool then reads L = 512 / 768 / 1024 inputs.

## Tables

```bash
python scripts/analyze_policies.py runs $LIBERO_ROOT/experiments                      # your runs
python scripts/analyze_policies.py paper --check                                       # the paper, from paper_records/
```

`runs` prints, for every run directory with evaluation files, the success per evaluation, the mean, the 95% interval, the
worst task and the tasks >= 80. `paper` recomputes, from the per-rollout records of the paper's 63 policies (189
evaluations, `paper_records/`, described in its README), Tab. 2, S8, S9, the refined SigLIP 2 policies of Sec. S3
and the retraining summary of Sec. S2; `--check` compares all 560 printed values with `paper_numbers.json` (all reproduced). Statistics: success = mean over three
evaluations x 10 tasks x 50 initial states; intervals from 4,000 bootstrap resamples of the tasks, then of the initial states
within each task, with the three evaluations of an initial state averaged first, shared by the policies of a column (paired);
the retraining interval from 10,000 resamples of the three trainings, the tasks and the rollouts. Generator seeds are in the
docstring of the script.

Values of the paper (success in %, mean of three evaluations; every policy with its seed and run in `paper_policies.tsv`):

| Tab. 2 | Long | Goal | Spatial | Object | mean |
|---|---|---|---|---|---|
| *BERT* | | | | | |
| CLIP-B/16 | 56.3 | 74.1 | 76.5 | 93.2 | 75.0 |
| SigLIP 2-B/16 | 72.3 | 78.8 | 84.1 | 90.3 | 81.4 |
| DINOv2-B/14 | 66.5 | 78.6 | 80.5 | 88.1 | 78.4 |
| DINOv3-B/16 | 68.4 | 72.7 | 80.7 | 94.4 | 79.0 |
| C-RADIOv3-B | 66.0 | 79.9 | 74.9 | 92.3 | 78.2 |
| ALPS (Stage 1) | 68.9 | 84.3 | 76.8 | 94.3 | 81.1 |
| ALPS | 69.7 | 85.1 | 86.5 | 92.2 | 83.4 |
| *own text tower* | | | | | |
| CLIP-B/16 | 61.5 | 84.3 | 79.1 | 91.9 | 79.2 |
| SigLIP 2-B/16 | 73.3 | 90.2 | 84.1 | 91.7 | 84.8 |
| C-RADIOv3-B | 73.4 | 89.5 | 82.9 | 81.4 | 81.8 |
| ALPS | 78.5 | 87.4 | 82.6 | 93.7 | 85.5 |

Sec. S3, SigLIP 2-B/16 + Stage 2 with BERT: Long 74.4, Goal 79.0, Spatial 82.4. Sec. S2, ALPS over three trainings per suite:
BERT 82.1%, own text tower 85.2%, paired gain +3.1 points, 95% interval -0.9 to +7.1.

## Rollout figure

```bash
cd $LIBERO_ROOT && PYTHONPATH=. MUJOCO_GL=egl python $ALPS_REPO/eval/libero_policy/scripts/record_rollouts.py \
    --run-dir <ALPS, BERT, LIBERO-Spatial run> --tasks 0 1 2 3 4 5 6 7 8 9 --init 0 --stride 5 --out rollouts_spatial.npz
# the same for LIBERO-Goal, then
python scripts/make_rollout_figure.py --suite LIBERO-Spatial rollouts_spatial.npz --suite LIBERO-Goal rollouts_goal.npz --out policy_rollouts.pdf
```

The figure shows the first two tasks per suite whose recorded rollout succeeded (15 of the paper's 20 recorded rollouts did;
a recorded rollout samples its own actions and can end differently from the benchmark rollout of the same initial state).

## The patches

`patches/libero_8f1084e3.patch` (apply to github.com/Lifelong-Robot-Learning/LIBERO at 8f1084e3132a39270c3a13ebe37270a43ece2a01):

| file | change |
|---|---|
| `libero/libero/envs/venv.py` | simulator workers of `SubprocVectorEnv` with the spawn context: a forked worker inherits the parent's EGL / CUDA context and dies at its first render (every dead worker counted as a failed rollout) |
| `libero/lifelong/algos/multitask.py` | training data-loader workers with the fork context (LIBERO's `main.py` sets spawn globally, which loses robomimic's observation specs in the workers) and `h5_worker_init`; `persistent_workers` only with workers; `LIBERO_SKIP_EP0` |
| `libero/lifelong/datasets.py` | `h5_worker_init`: every worker reopens its hdf5 files |
| `libero/lifelong/metric.py` | `LAST_SUCCESSES`: the outcome of every rollout of the last `evaluate_one_task_success` call |
| `libero/lifelong/models/bc_transformer_policy.py` | imports `FrozenEncoderPool` |
| `libero/lifelong/utils.py` | `LIBERO_EXP_SUFFIX` (one experiment directory per policy); `task_embedding_format=file:<npz>` |
| new `libero/lifelong/models/modules/frozen_encoder_pool.py` | imports the module of this folder from `$ALPS_REPO` (a placeholder that raises when it is missing) |
| new `libero/configs/policy/image_encoder/frozen_encoder_pool.yaml` | the image-encoder config of the paper's policies |

Nothing else of the research fork is needed for the paper's policies (all `lifelong=multitask`); the patch leaves out the
data-loader changes of the other lifelong algorithms (`base.py`, `er.py`, `ewc.py`, `packnet.py`), a `filter_key` pass-through
of `main.py` (unset in every run, so the default of LIBERO) and the goal-image conditioning of another study (off in every run).
The kept changes are the code of the runs (comments rewritten; two function-local imports replaced by the module's own `os`
/ `np`); the yaml names the encoder by `encoder` / `weights` (one key, `encoder_ckpt`, in the paper's run configs) and lists the
module's switches.

`patches/robomimic-0.2.0_dataset.patch` adds `SequenceDataset.__getstate__` (drop the open hdf5 handle when the dataset is
pickled), as installed in the environment of the runs. With the fork-context workers of the LIBERO patch the dataset is never
pickled, so the patch has no effect on these experiments; it matters only if the workers are started with spawn.

LIBERO and robomimic are MIT-licensed; the patched files keep their licenses. The files of this folder are Apache-2.0.

## Tests

```bash
pytest eval/libero_policy/tests          # CPU, about a minute (most of it the check of the paper's numbers); no LIBERO, no downloads
```

The tests check the preprocessing of the module against an explicit flip / resize / normalize, the attention pool, the
parameter and state-dict content, the construction order (initial weights for a seed), the train / eval mode handling,
the loading of research checkpoints, the safetensors export and loading of policy weights, the defaults, the timm
encoders of the paper (random weights), the ALPS port arithmetic against a port-structured ViT, the random-number
consumption of the ALPS ViT against the port, the LIBERO-side import module of the patch, the records of
`paper_records/` (manifest, content) and every printed policy number of the paper.
