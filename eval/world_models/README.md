# World-model evaluation

Frozen visual encoders as the state of world models: planning and prediction accuracy with
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) on PushT, Wall, PointMaze, Granular and Rope, and with
[LeWorldModel](https://github.com/lucas-maes/le-wm) on Reacher. This folder reproduces Table 3 (planning), Table 4
(prediction 5 world-model steps ahead), Tables S11 to S17 and Fig. 6 (rendered predictions) of the paper.

| Folder | Contents |
|---|---|
| `dino_wm/dino_wm.patch` | changes to DINO-WM at commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c` (10 files) |
| `dino_wm/overlay/` | new files copied into the DINO-WM checkout: frozen-encoder adapter, predictor with projections, configurations, trainer, token statistics, probes, prediction evaluation, rendering export |
| `dino_wm/scripts/` | launchers (environment variables, no machine-specific paths) |
| `reacher/` | LeWorldModel integration: backbone adapter, statistics, trainer, probes, prediction evaluation, held-out episodes, `le_wm.patch` (episode sharding of `eval.py`) |
| `tools/` | merging of planning shards, decoding of predicted tokens, tables and statistics, Fig. 6 |
| `tests/` | CPU tests (`pytest eval/world_models/tests`) |

Nothing here contains data, checkpoints, token statistics or probes: every step below produces its own.

## Encoders

Six frozen encoders, each read through the unified registry `eval/encoders.py` (environment variable `ALPS_ROOT` =
the root of this repository). Every encoder gives 196 tokens: the native last-layer patch tokens, prefix tokens dropped.

| Paper row | Name | Tokens and input |
|---|---|---|
| ALPS | `alps` | raw tokens of the last block (no final LayerNorm); ImageNet mean / std; 224 px; weights `ALPS_WEIGHTS` |
| ALPS (Stage 1) | `alps_stage1` | the same, weights `ALPS_STAGE1_WEIGHTS` |
| DINOv2-B/14 | `dinov2_b14` | torch hub `facebookresearch/dinov2` `dinov2_vitb14`, `x_norm_patchtokens`; mean = std = 0.5 (the DINO-WM transform); **196 px** (14 x 14 tokens, the DINO-WM rule). Reacher: `dinov2_b14_196`, the Hugging Face model `facebook/dinov2-base` (`last_hidden_state`), ImageNet normalisation, 196 px |
| DINOv3-B/16 | `dinov3_b16` | timm `vit_base_patch16_dinov3.lvd1689m` (`img_size=224`), `forward_features`, its own mean / std |
| SigLIP 2-B/16 | `siglip2_b16` | timm `vit_base_patch16_siglip_224.v2_webli`, `forward_features`, its own mean / std |
| C-RADIOv3-B | `cradio_v3_b` | NVlabs RADIO hubconf `radio_model` with `c-radio_v3-b_half.pth.tar` (HF `nvidia/C-RADIOv3-B`), backbone spatial features, no adaptor; RGB in [0, 1]; env `RADIO_REPO`, `RADIO_CKPT` |

DINO-WM tasks (`models/encoder/frozen_encoder.py`): the frame in [-1, 1] of the DINO-WM transform is mapped to [0, 1],
resized to the encoder input size if needed (torchvision bilinear, antialiased), normalised with the encoder's constants
(C-RADIOv3: clamped to [0, 1]), encoded in float32 with TF32 allowed inside the encoder forward only, and the tokens are
standardised per channel with the statistics of the task (below). The world model works on the 768 standardised channels;
the predictor projects them to 384 and back (`models/vit_proj.py`). Reacher (`reacher/frozen_backbone.py`): the
ImageNet-normalised 224-px pixels of the PreJEPA pipeline, the encoder in float32, every token normalised to unit length
and then standardised per channel.

## Data

DINO-WM tasks: the data release of DINO-WM on OSF (project `bmw48`, view-only link
`https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28`; direct downloads
`https://osf.io/download/<id>/?view_only=a56a296ce3b24cceaf408383a175ce28`):

| File | id | Size | Folder in `$DATASET_DIR` |
|---|---|---|---|
| `datasets/pusht_noise.zip` | `k2d8w` | 2.8 GB | `pusht_noise/` (18,685 train / 21 validation trajectories) |
| `datasets/wall_single.zip` | `49rnx` | 1.7 GB (59 GB unzipped) | `wall_single/` (1,920 episodes) |
| `datasets/point_maze.zip` | `vr5gy` | 0.7 GB | `point_maze/` (2,000 episodes) |
| `datasets/deformable.zip` + `.z01` `.z02` `.z03` | `w9qry`, `vfj7m`, `sdzyx`, `xs426` | 15.8 GB split zip | `deformable/rope/`, `deformable/granular/` (1,000 episodes each) |
| `checkpoints/outputs.zip` | `xvzs4` | 1.0 GB | released DINO-WM models; only for the pipeline check |

The split zip unpacks with `7z x deformable.zip` (or `zip -s- deformable.zip -O full.zip && unzip full.zip`). The loaders
read `states.pth`, `actions.pth` and the `obses` files; the `*.h5` files of the deformable episodes are not used.

Reacher: the LeWorldModel data on Hugging Face, dataset `quentinll/lewm-reacher`, file `reacher.tar.zst` (sha256 begins
with `4ff2385e49712caa`); `tar --zstd -xvf reacher.tar.zst` and place `reacher.h5` under `$STABLEWM_HOME` (dataset name
`reacher`). The prediction evaluation needs 100 held-out episodes, collected once with the dataset's own procedure and a
new seed (`reacher/reacher_collect_heldout.py`: `swm/ReacherDMControl-v0`, random policy, 200 steps, 224 x 224, seed
20261001); `reacher/scripts/predeval.sh` runs it.

## Environments

**DINO-WM** (PushT, Wall, PointMaze, Granular, Rope). The conda environment of DINO-WM (`conda env create -f
environment.yaml`: Python 3.9.19, torch 2.3.0 + CUDA 12.1, torchvision 0.18.0, hydra-core 1.2.0, accelerate 0.26.1,
numpy 1.26.4, decord 0.6.0, gym 0.23.1, mujoco-py 2.1.2.14, d4rl 1.1, pymunk 6.8.0, wandb 0.13.1), plus
`pip install timm==1.0.28 "transformers>=4.53,<5" "huggingface_hub>=0.30" safetensors` (the paper used transformers
4.57.6, huggingface_hub 0.36.2). Then:

```bash
bash dino_wm/scripts/setup_dino_wm.sh /path/to/dino_wm      # clone at 0a9492fa, apply dino_wm.patch, copy overlay/
cp dino_wm/scripts/env.sh.example env.sh                     # edit the paths, then: source env.sh
```

* DINOv2 through torch hub: the current `main` of `facebookresearch/dinov2` does not import on Python 3.9; put the last
  2024 commit into the hub cache before the first use:
  `git clone https://github.com/facebookresearch/dinov2 $TORCH_HOME/hub/facebookresearch_dinov2_main && git -C $TORCH_HOME/hub/facebookresearch_dinov2_main checkout 85a24602099d397264d5b30461ad7f3bfd726ca1`.
  The weights (`dinov2_vitb14_pretrain.pth`, and `dinov2_vits14_pretrain.pth` for the pipeline check) are downloaded to
  `$TORCH_HOME/hub/checkpoints/` on first use.
* PointMaze needs MuJoCo 2.1.0 (`mujoco210-linux-x86_64.tar.gz`, `MUJOCO_PY_MUJOCO_PATH`) and mujoco-py, which compiles
  against the GL / GLEW / EGL headers; without root, a separate conda environment with `glew mesalib libglvnd-devel
  libegl-devel patchelf` (conda-forge) provides them (`CPATH` and `LIBRARY_PATH` for the first import, its `lib/` and
  `/usr/lib/nvidia` on `LD_LIBRARY_PATH` at run time). With the patch, the other tasks run without mujoco-py.
* Granular and Rope need PyFleX: the sources of [AdaptiGraph](https://github.com/Boey-li/AdaptiGraph) (commit
  `a7c75357793422e9108c36f58c6928f3f8a123ad`), compiled in the docker image `xingyu/softgym` as the DINO-WM README
  describes (`PYFLEXROOT`, `PYTHONPATH`, `LD_LIBRARY_PATH` as in `env.sh.example`). PyFleX puts the rendering context of
  every process on the first GPU, and several processes that initialise the simulator at the same moment crash: keep the
  first GPU free of planning shards and start shards one after the other (`plan_all.sh` does). If GPU indices shift while
  the first GPU is full, pass GPU UUIDs.
* C-RADIOv3: a copy of [NVlabs/RADIO](https://github.com/NVlabs/RADIO) (`RADIO_REPO`) and the checkpoint
  `c-radio_v3-b_half.pth.tar` of `nvidia/C-RADIOv3-B` (`RADIO_CKPT`).

**LeWorldModel** (Reacher). A Python 3.10 environment (the paper used uv, CPython 3.10.21): `pip install torch torchvision`,
`pip install "stable-worldmodel[train]==0.1.1" dm_control "mujoco<3.12" h5py hdf5plugin opencv-python-headless hydra-core
zstandard timm transformers safetensors` (the paper: torch 2.14.0 + CUDA 12.6, stable-worldmodel 0.1.1,
stable-pretraining 0.1.8, lightning 2.6.6, hydra-core 1.3.7, timm 1.0.30, transformers 5.17.0, mujoco 3.11.0,
dm_control 1.0.44, gymnasium 1.3.0, h5py 3.16.0, numpy 2.2.6). Then:

```bash
bash reacher/scripts/setup_lewm.sh /path/to/le-wm /path/to/stable-worldmodel   # le-wm 8edfeb33 + le_wm.patch, stable-worldmodel 63988116
cp reacher/scripts/env.sh.example reacher_env.sh                               # edit, then: source reacher_env.sh
```

The trainer imports `scripts/train/prejepa.py` from the stable-worldmodel clone (`SWM_ROOT`); `reacher/` must be on
`PYTHONPATH` (the world-model configs instantiate `frozen_backbone.FrozenTokens`). Rendering uses MuJoCo EGL.

## The world-model recipe

One recipe for every task and encoder (`wm_train.py`, `reacher_wm_train.py`):

| | DINO-WM tasks | Reacher |
|---|---|---|
| model | ViT predictor of DINO-WM, 6 layers, 16 heads, MLP 2048, dropout 0.1, block-causal; learned projections 768 -> 384 -> 768 on the visual channels; proprioception and action embeddings of 10 channels each (`concat_dim` 1); no decoder | PreJEPA of stable-worldmodel: causal predictor, 6 layers, 16 heads, MLP 2048, width 768 + 10; action embedder 10 -> 10; no proprioception |
| loss | one-step teacher-forced MSE on the visual and proprioceptive channels against the detached targets | `dinowm_forward` of `prejepa.py` |
| optimizer | AdamW (betas .9 / .999, eps 1e-8, weight decay .01), no clipping, float32 | AdamW without weight decay; fp16 autocast with a GradScaler; gradient norm clipped at 1.0 after the loss scale is removed |
| batch | 8 processes x 32 = 256 (DDP); needs 8 GPUs to reproduce the data order | the same |
| learning rate | 5e-4 x sqrt(256 / 32) = 1.4142e-3 for every optimizer; 200 linear warm-up updates; cosine to 1% of the peak at update C; 1% to the end T | the same |
| (T, C) | PushT (31,000, 29,500); Wall and PointMaze (15,500, 14,700); Granular and Rope (7,750, 7,350) | (26,000, 24,700) |
| history / frame skip | PushT and PointMaze 3 / 5, Wall 1 / 5, Granular and Rope 1 / 1 | 3 / 5 |
| seed | training seed 1: initialisation (identical for every encoder), data order, dropout | the same; train / validation split of the clips with seed 42 |
| data order | the official training slices (DINO-WM's permutation of the slices of the train split in the order of `subsets/<task>_100.json`), one continuous stream over epochs | per epoch e: `RandomState(seed * 100003 + e)` permutation of the training clips |

Every model is evaluated at the fixed endpoint T (no checkpoint selection). The trainers save full resumable states
(`model_s<update>.pth`); the Reacher trainer also exports `weights_s<update>.pt` in LeWorldModel's format.

**Token statistics.** DINO-WM tasks (`fit_stats.py`): 20,000 frames sampled with `RandomState(0)` from 4,000 (or all, if
fewer) training trajectories, through the training image path (the adapter in raw mode, TF32 inside the encoder); over all
196 tokens of every frame in float64: per-channel mean and sd = sqrt(var + 1e-8). Reacher (`reacher/make_stats.py`):
4,096 frames of `reacher.h5` (`RandomState(0)`), unit-length tokens in float32, mean and sd = sqrt(max(E[t^2] - mean^2,
1e-12)). The trajectory order files (`make_subsets.py`, all training trajectories permuted with `RandomState(0)`) are
byte-identical to the paper's files when the data are complete (md5 sums in `make_subsets.py`).

## Planning

DINO-WM's `plan.py` with `planner.max_iter=10 seed=99` and the fixed endpoint, in shards (`+eval_start=A +eval_end=B`; a
shard keeps the global goal set and the per-instance seeds `99 n + 1`; the paper's numbers come from these shards):

| Task | Config | Instances (shards) | CEM | Overrides |
|---|---|---|---|---|
| PushT | `plan_pusht` (goals 25 simulator steps ahead in validation trajectories) | 50 (5 x 10) | 300 samples, top 30, horizon 5, 30 iterations | `+planner.sub_planner.rollout_chunk=100 +tf32=true` |
| Wall | `plan_wall` (random states) | 50 (5 x 10) | 10 iterations | the same |
| PointMaze | `plan_point_maze` (random states, objective without proprioception) | 50 (5 x 10) | 10 iterations | the same |
| Granular | `plan_granular` (= `plan_wall`, 10 instances) | 10 (5 x 2) | 10 iterations | the same + `planner.sub_planner.eval_every=1000` |
| Rope | `plan_rope` (= `plan_wall`) with `n_evals=50` | 50 (5 x 10) | 10 iterations | the same + `planner.sub_planner.eval_every=1000` |

MPC executes the 5 planned steps and plans again, for at most 10 rounds; PushT, Wall and PointMaze stop early when every
goal is reached (success: PushT agent and block within 20 units in total and the block angle within 20 degrees; Wall
within 4.5 units; PointMaze within 0.5 units); Granular and Rope report the final Chamfer distance and always run 10 rounds.
`rollout_chunk` only bounds the memory of the CEM rollouts; `+tf32=true` runs the world-model rollouts in TF32.

Reacher: LeWorldModel's `eval.py` with `reacher.yaml` (`eval.num_eval=50 eval.dataset_name=reacher`): the same 50
episodes (seed 42), goals 25 steps ahead, budget 50 environment steps, success when every joint is within 0.05 rad; CEM
300 samples, 30 iterations, top 30, horizon 5; planner seeds `solver.seed=42, 43, 44`, each in 5 shards of 10
(`+eval.shard=i/5`, from `le_wm.patch`); rendering pinned to the shard's GPU (`eval_pinned.py`).

## Prediction accuracy

A state probe per encoder and task reads the state from the tokens of one frame (2 cross-attention blocks with 8 heads,
8 learned queries, width 256, AdamW 3e-4, weight decay 0.05, cosine to zero over 6,000 updates, batch 128, seed 0; 20,000
real training frames, 2,000 other training frames for model selection every 500 updates). Targets: PushT agent and block
position and block angle (sin, cos), Wall agent position (MSE on standardised targets); Granular and Rope 256 points
trained with the Chamfer distance to the particles (Granular: a fixed subset of 2,048; Rope: all 1,965); Reacher sin / cos
of the two joint angles and the fingertip. The probe depends only on the frozen encoder and its statistics.

The world model then rolls out open loop from real history frames with the recorded actions, 1 to 5 world-model steps
ahead, on held-out data: PushT all 1,695 validation windows, Wall 2,000 windows, Granular and Rope all 1,500 windows,
Reacher 20 windows of each of the 100 held-out episodes (2,000). Table 4 is the error 5 steps ahead (PushT block position,
Wall agent position, Chamfer distance, Reacher joint angles in degrees); Table S16 adds every step and two references 5
steps ahead: the probe on the true tokens of that frame and on the tokens of the last observed frame.

## Rendered predictions (PushT)

`render_tokens.py` (CPU, as in the paper) exports, for 200 validation windows (`default_rng(0)`), the true frames, the
real tokens and the tokens of the open-loop rollout (and two diagnostic conditions: one-step predictions from the true
history, and the rollout with the recorded proprioception), converted back to each encoder's raw token space
(z sd + mu). `tools/render_decode.py` renders them with the encoder's evaluation decoder of the reconstruction benchmark,
trained on the raw tokens of 19,000 PushT training frames (its PushT option), and the frozen SDXL VAE decoder
(`stabilityai/sdxl-vae`); l1 on images in [-1, 1] and LPIPS-VGG against the true frames. DINOv2 is not rendered (its world
model sees 196-px frames, which its decoder does not match). `tools/render_stats.py` gives Tables S15 and S17 and the
windows of the motion rule (834, 138, 991); `tools/make_render_figure.py` assembles Fig. 6.

## Commands

```bash
source env.sh
# per DINO-WM task: trajectory order file and token statistics of the six encoders (one GPU)
CUDA_VISIBLE_DEVICES=0 bash dino_wm/scripts/prepare.sh pusht            # wall point_maze granular rope
# per task and encoder: training (8 GPUs), planning (shards on the listed GPUs), prediction accuracy (one GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash dino_wm/scripts/train.sh pusht alps
GPUS="0 1 2 3 4" bash dino_wm/scripts/plan_all.sh pusht alps           # Granular / Rope: GPUS="1 2 3 4 5", never the first GPU
CUDA_VISIBLE_DEVICES=0 bash dino_wm/scripts/predeval.sh pusht alps      # pusht wall granular rope
# Reacher
source reacher_env.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash reacher/scripts/train.sh alps          # also writes the statistics
GPUS="0 1 2 3 4" bash reacher/scripts/plan_all.sh alps                         # planner seeds 42, 43, 44
CUDA_VISIBLE_DEVICES=0 bash reacher/scripts/predeval.sh alps
# tables
python tools/summarize.py --dino_wm $WM_ROOT/results --reacher $REACHER_OUT/results
```

Encoder names: `alps alps_stage1 dinov2_b14 dinov3_b16 siglip2_b16 cradio_v3_b` (Reacher: `dinov2_b14_196` for DINOv2).
`summarize.py` prints every reproduced cell next to the paper's value.

**Table 3** (planning; success in %, PushT / Wall final state distance, Granular / Rope final Chamfer distance; Reacher:
mean of the three planner seeds):

| Encoder | PushT success | dist. | Wall success | dist. | PointMaze | Reacher | Granular | Rope |
|---|---|---|---|---|---|---|---|---|
| DINOv2-B/14 | 88 | 25.0 | 92 | 3.62 | 100 | 76.7 | .211 | .701 |
| DINOv3-B/16 | 88 | 24.2 | 90 | 3.69 | 100 | 86.0 | .275 | .681 |
| C-RADIOv3-B | 90 | 20.2 | 90 | 3.31 | 100 | 82.0 | .180 | .558 |
| SigLIP 2-B/16 | 84 | 29.0 | 88 | 3.73 | 100 | 74.0 | .230 | .848 |
| ALPS (Stage 1) | 88 | 24.0 | 90 | 3.43 | 100 | 76.0 | .324 | .943 |
| ALPS | 90 | 23.4 | 92 | 3.21 | 100 | 78.7 | .176 | .750 |

**Table S11** (Rope, 50 instances, mean / median Chamfer): DINOv2 .701 / .446, DINOv3 .681 / .406, C-RADIOv3 .558 / .289,
SigLIP 2 .848 / .511, ALPS (Stage 1) .943 / .514, ALPS .750 / .448; DINOv2-S/14 with the DINO-WM recipe .547 / .309
(.567 on the first 10 instances). **Table S12** (Reacher success per planner seed 42 / 43 / 44): DINOv2 72 / 80 / 78,
DINOv3 90 / 90 / 78, C-RADIOv3 82 / 86 / 78, SigLIP 2 68 / 78 / 76, ALPS (Stage 1) 70 / 78 / 80, ALPS 82 / 80 / 74.

**Table 4** (error 5 world-model steps ahead; Table S16 adds k = 1..4 and the references, all in `summarize.py`):

| Encoder | PushT block | Wall agent | Granular Chamfer | Rope Chamfer | Reacher joints (deg) |
|---|---|---|---|---|---|
| DINOv2-B/14 | 7.16 | 9.69 | .246 | .410 | 1.21 |
| DINOv3-B/16 | 8.70 | 9.78 | .222 | .407 | 1.03 |
| C-RADIOv3-B | 5.64 | 9.75 | .218 | .372 | 1.06 |
| SigLIP 2-B/16 | 11.96 | 9.91 | .268 | .469 | 2.02 |
| ALPS (Stage 1) | 9.32 | 9.81 | .272 | .424 | 1.42 |
| ALPS | 6.73 | 9.69 | .236 | .377 | 1.18 |

**Paired comparisons** (world models; the paper states their readings, not the table): `summarize.py` recomputes the
paired comparisons, ALPS minus each encoder (prediction:
bootstrap over trajectories, 4,000 resamples, seed 1, and the 10% rule; planning: exact McNemar over goals, exact
sign-flip over Reacher episodes with the successes summed over the planner seeds, bootstrap over Granular / Rope
instances, 4,000 resamples, seed 0), and prints our reference values next to them.

**Tables S15, S17 and Fig. 6** (PushT, the five rendered encoders):

```bash
for e in alps alps_stage1 cradio_v3_b dinov3_b16 siglip2_b16; do
  bash dino_wm/scripts/render.sh $e                                                    # DINO-WM environment, CPU
  python tools/render_decode.py --enc $e --tokens $WM_ROOT/results/render/${e}_tokens.npz \
    --decoder <PushT evaluation decoder of $e> --out $WM_ROOT/results/render --check_tokens   # reconstruction environment, one GPU
done
python tools/render_stats.py --diag_dir $WM_ROOT/results/render --render_dir $WM_ROOT/results/render
python tools/make_render_figure.py --png_dir $WM_ROOT/results/render --out render_main.pdf
```

Table S15 (token cosine, decoded real tokens k = 5 LPIPS, rendered prediction k = 1 LPIPS, k = 5 l1, k = 5 LPIPS):
SigLIP 2 1.000 .0081 .0107 .0150 .0308; C-RADIOv3 1.000 .0070 .0088 .0103 .0176; DINOv3 1.000 .0074 .0096 .0128 .0242;
ALPS (Stage 1) 1.000 .0066 .0093 .0126 .0248; ALPS 1.000 .0068 .0090 .0114 .0203. Paired k = 5 LPIPS (not printed in the paper), ALPS minus C-RADIOv3 +.0028
[+.0013, +.0042], DINOv3 -.0038 [-.0068, -.0014], ALPS (Stage 1) -.0044 [-.0061, -.0027], SigLIP 2 -.0104 [-.0135, -.0079].
Table S17 (block error 5 steps ahead: true future tokens, one-step prediction, five-step rollout, rollout with recorded
proprioception, last observation): ALPS 3.1 3.6 6.6 6.7 42.0; ALPS (Stage 1) 3.8 4.3 9.9 9.2 42.3; C-RADIOv3 2.4 2.8 5.8
5.1 41.9; DINOv3 3.0 3.2 9.1 8.4 41.7; SigLIP 2 3.7 5.2 14.1 12.1 42.0.

**Planning figures** (Figs. S2, S3, not in a table): `plan_viz.py` is `plan.py` plus a record of the executed rollouts;
run it with the PushT settings above and `+eval_start=0 +eval_end=4` for the ALPS and DINOv2 world models.

**Pipeline check** (Sec. S2, last row of Table S11): DINOv2-S/14 with the encoder, predictor and schedule of DINO-WM,
trained by the official `train.py` (decoder off, fixed number of updates, `conf/train_dinowm_recipe.yaml`):

```bash
bash dino_wm/scripts/train_dinowm_recipe.sh pusht 0,1,2,3 training.max_steps=123858            # the budget of the released PushT model
bash dino_wm/scripts/train_dinowm_recipe.sh wall 0,1 training.batch_size=128 training.max_steps=36010
bash dino_wm/scripts/train_dinowm_recipe.sh rope 0,1,2,3 training.max_steps=53450               # 100 epochs
# plan with the settings of the task; Rope in shards of 3 instances (the last of 2), started one after the other
for a in 0 10 20 30 40; do MODEL=pusht_dinov2_vits14_dinowm_recipe EPOCH=final CUDA_VISIBLE_DEVICES=1 \
  bash dino_wm/scripts/plan.sh pusht dinov2_vits14_dinowm_recipe $a $((a + 10)); done
for a in $(seq 0 3 48); do MODEL=rope_dinov2_vits14_dinowm_recipe EPOCH=final CUDA_VISIBLE_DEVICES=1 \
  bash dino_wm/scripts/plan.sh rope dinov2_vits14_dinowm_recipe $a $(( a + 3 < 50 ? a + 3 : 50 )); done
python tools/merge_plan_shards.py $WM_ROOT/plan_logs/rope_dinov2_vits14_dinowm_recipe_*.log > $WM_ROOT/results/plan/rope_dinov2_vits14_dinowm_recipe.json
```

The paper: PushT 94% and Wall 92% over 50 goals (DINO-WM reports 90% and 96%); Rope .547 over the 50 instances and .567
over the first 10 (.41 reported by DINO-WM). The released DINO-WM models (`outputs.zip`; `plan.py ckpt_base_path=<folder
that contains outputs/> model_name=pusht|wall_single model_epoch=latest` with the planning settings above) reach 92% and
94% in this planner.

## Reproducibility notes

* The code path, configurations, seeds, shards and evaluation sets are those of the paper's runs; the trajectory order
  files are byte-identical. The world models, token statistics and probes are retrained, so the numbers reproduce up to
  GPU non-determinism (cuDNN / NCCL reductions, TF32) and the following implementation detail.
* The ALPS encoders are loaded through the inference package of this repository (`alps/`), which reproduces the float32
  training forward (LayerNorm epsilon 1e-6). The paper's world-model runs used an earlier PyTorch port of the same weights
  with LayerNorm epsilon 1e-5; on the real weights the tokens of the two agree to a cosine of at least 0.9999999 (largest
  relative difference 1.5e-4).
* `+tf32=true` and the TF32 setting of the frozen encoders are part of the protocol; planning without them is the official
  float32 path of DINO-WM and gives slightly different CEM samples.
* The first world-model run of PushT for ALPS and C-RADIOv3 in the paper used a trainer that differs from `wm_train.py`
  only in a configuration check (same mathematics, data order and initialisation).

## Changes to DINO-WM (`dino_wm.patch`)

* `datasets/*`: optional `traj_subset` (the train split restricted to and ordered by a json file); a slice loader that
  decodes only the frames of a slice (identical tensors, `DINO_WM_FAST_SLICE=0` restores the official path); a guard in
  `TrajSubset.__getattr__` against infinite recursion when unpickling.
* `env/__init__.py`: PointMaze is registered only when mujoco-py imports.
* `plan.py`: shards `+eval_start / +eval_end` that keep the global goal set and seeds; optional `+tf32`.
* `planning/cem.py`: optional `rollout_chunk` (memory); `eval_every > opt_steps` switches the per-step simulator evaluation
  off completely (also at step 0).
* `planning/evaluator.py`: the plotting decode of imagined rollouts without autograd and in chunks (decoder models only).
* `train.py` (pipeline check only): a fixed number of updates, step checkpoints and `model_final.pth`, the inner torch
  optimizer in checkpoints, validation without autograd.
