# Video benchmarks: DAVIS-2017 label propagation and the SSv2 attentive probe

ALPS is pretrained on still images only; these two benchmarks test how far its frozen tokens transfer to video
(supplementary, "Temporal transfer"): video object segmentation on DAVIS-2017 by label propagation between frames,
which tests correspondence across frames, and action recognition on Something-Something v2 with an attentive probe,
which tests temporal action information. Every encoder is frozen and evaluated with one fixed protocol.

| File | Purpose |
|---|---|
| `davis.py` | DAVIS-2017 label propagation (no training) and J&F |
| `ssv2_probe.py` | SSv2 attentive probe (training and evaluation, `torchrun` data-parallel) |
| `run_ssv2.sh` | launcher of `ssv2_probe.py` over a list of GPUs |
| `make_ssv2_csv.py` | SSv2 split files (train minus a 2,000-video holdout, holdout, validation) |
| `make_probe_init.py` | the shared probe initialization, and its adaptation to V-JEPA 2 |
| `video_encoders.py` | the encoders of both benchmarks on top of `eval/encoders.py`, incl. the V-JEPA 2 RoPE handling |
| `summarize_davis.py`, `summarize_ssv2.py` | tables over runs; paired comparison of DAVIS runs over videos |
| `ssv2_pair.py` | paired comparison of two SSv2 runs over validation videos |
| `tests/` | CPU tests (label propagation, probe, splits, V-JEPA 2 positions) |

## Protocols

### DAVIS-2017 (`davis.py`)

The label propagation of DINO ([facebookresearch/dino](https://github.com/facebookresearch/dino),
`eval_video_segmentation.py`) with its default hyper-parameters, on the 30 validation videos, without training:

* **Frames.** The 480p frames are resized (cv2, bilinear) to a short side of 480 px, floored to a multiple of the
  patch size (476 px for patch 14); the long side is scaled and floored to a multiple of 64 px when the patch size
  divides 64 (patch 16), else to a multiple of the patch size. An 854 x 480 frame becomes 832 x 480 (52 x 30 tokens)
  for patch 16 and 840 x 476 (60 x 34 tokens) for patch 14.
* **Features.** The native last-layer patch tokens of each encoder (as in every table of the paper), stored in
  float16 and L2-normalized. Image encoders encode every frame alone. V-JEPA 2 receives the 16 frames up to the
  current one, padded at the start with the first frame, and the tokens of its last tubelet are the features of the
  current frame. The prediction of frame t only uses frames up to t.
* **Propagation.** Frame t is predicted from the first frame (its ground-truth mask) and the 7 previous frames (their
  predicted soft masks): affinities exp(cos / 0.1) between every token of frame t and the tokens of these frames within
  a radius of 12 token cells; per token, the 5 largest affinities over all context tokens are kept and normalized, and
  the soft masks are propagated with these weights. The soft mask is upsampled bilinearly by the patch size, normalized
  per object, and its argmax is resized (nearest) to the frame size.
* **Scores.** J and F from [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)
  (semi-supervised task); J&F = (mean J + mean F) / 2 over objects.
* **Precision.** Encoder inference under bfloat16 autocast (`--precision bf16`, the default and the setting of the
  tables); `--precision fp32` gives the 32-bit numbers of the Precision paragraph. Propagation runs in float32.
* **Positions at the frame grid**, per encoder (`video_encoders.py`):

| Encoder | `--enc` | Positions at grids other than the pretraining grid |
|---|---|---|
| ALPS, ALPS (Stage 1) | `alps`, `alps_stage1` | fixed sine-cosine table evaluated at cell indices rescaled into the 14 x 14 training range (`pos_mode="interpolate"`) |
| TIPSv2-B/14, DINOv3-B/16, CLIP-B/16 (QuickGELU), SigLIP 2-B/16, PE-Core-B/16 | `tipsv2_b14`, `dinov3_b16`, `clip_b16_qg`, `siglip2_b16`, `pe_core_b16` | timm models created with `img_size=224, dynamic_img_size=True`: position tables resampled to 224 px at creation and interpolated to the frame grid at run time; rotary positions computed by timm for the frame grid |
| DINOv2-B/14 | `dinov2_b14_timm` | as above, timm `vit_base_patch14_dinov2.lvd142m` (`dinov2_b14` is the torch-hub DINOv2 of the world-model experiments, with a different position interpolation; it is not used here) |
| C-RADIOv3-B | `cradio_v3_b` | the model's own handling of arbitrary resolutions |
| V-JEPA 2-L/16 | `vjepa2_vitl` | 3-D RoPE ids recomputed for the actual token grid; height and width positions scaled to the 16 x 16 pretraining grid, (i * 16 / gh, j * 16 / gw) (`--vjepa2_rope interp`, the default); `--vjepa2_rope recompute` keeps integer positions far beyond the pretraining range |

Frames are normalized with the ImageNet mean and standard deviation; `VideoEncoder.from_imagenet` converts them to
each encoder's own input convention (identity for ImageNet statistics, an affine map for other statistics, back to
[0, 1] for C-RADIOv3).

### Something-Something v2 (`ssv2_probe.py`)

The frozen-encoder video classification evaluation of V-JEPA 2
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) at commit
`204698b45b3712590f06245fbfba32d3be539812`; its `VideoDataset`, `make_transforms` and `AttentiveClassifier` are
imported from a checkout given by `VJEPA2_ROOT`), with one configuration for every encoder:

* **Clips.** 16 frames with a frame step of 4, at 224 px, one clip per video and epoch. Training augmentation:
  RandAugment, RandomResizedCrop (scale 0.08 to 1, aspect ratio 3/4 to 4/3), random erasing (probability 0.25), no
  horizontal flip. Evaluation: 1 temporal segment x 3 spatial views at a short side of 224 px, softmax-averaged.
* **Probe.** A non-affine LayerNorm and a linear adapter from the token width to 768, a learnable temporal embedding
  (initialized to zeros), and the AttentiveClassifier (depth 4: 3 self-attention blocks and the cross-attention query
  block, 12 heads, width 768) over all tokens of the clip, 174 classes. Image encoders encode the 16 frames one by one
  (16 temporal positions); V-JEPA 2 encodes the clip (8 tubelets).
* **Training.** AdamW, learning rate 1e-3, weight decay 0.1, cosine schedule to 0 without warm-up over the optimizer
  steps, global batch 256 (GPUs x per-GPU batch x `--accum`), 20 epochs, bfloat16 autocast without a gradient scaler,
  encoder frozen (bfloat16 inference), torch seed 0.
* **Splits.** Training on the official training set minus a fixed holdout of 2,000 videos (`make_ssv2_csv.py`), which
  is evaluated after every epoch for monitoring only; the reported number is top-1 on the official validation set
  (24,777 videos) after epoch 20 (`val_top1_final` in `results.json`). No decision uses the validation set.
* **Probe initialization.** Every encoder starts from one shared initialization of the probe (`--init_from`). In the
  paper's runs it was the probe saved by the first run (ALPS (Stage 1)), drawn from that run's random state; that file
  is not released. `make_probe_init.py` draws a shared initialization under torch seed 0 instead: the same protocol,
  but runs from it are not bitwise those of the paper. V-JEPA 2-L (1,024-d tokens, 8 tubelets) starts from the shared
  initialization with the input adapter re-drawn (trunc-normal, std 0.02, seed 0), bias 0 and 8 temporal positions
  (zeros); the attentive classifier is unchanged (`make_probe_init.py --base ... --in_dim 1024 --t_tokens 8`).
* **Differences from the official V-JEPA 2 configuration** (also written to `meta.json`): resolution 224 for every
  encoder (official: 256); the probe runs at width 768 after the adapter (official: the encoder's own width) with 12
  heads and a temporal embedding; one fixed learning rate and weight decay (official: a sweep on the validation set);
  the 2,000-video holdout; 1 temporal segment (official: 2, identical clips for SSv2 videos of at most 64 frames);
  bfloat16 autocast without a gradient scaler (official: float16 with GradScaler); deterministic clip sampling at
  evaluation; activation checkpointing off (`--act_ckpt 1` gives the same numbers with less memory).

## Data and code

* **DAVIS-2017**: `DAVIS-2017-trainval-480p.zip` from the [DAVIS challenge](https://davischallenge.org/davis2017/code.html),
  unpacked so that `--davis` holds `JPEGImages/480p/`, `Annotations/480p/` and `ImageSets/2017/val.txt`.
  Evaluation code:
  ```bash
  git clone https://github.com/davisvideochallenge/davis2017-evaluation
  git -C davis2017-evaluation checkout ac7c43fca936f9722837b7fbd337d284ba37004b
  ```
* **Something-Something v2**: the videos (`<id>.webm`, about 220k) and the label files (`labels.json`, `train.json`,
  `validation.json`) from [Qualcomm](https://www.qualcomm.com/developer/software/something-something-v-2-dataset).
  ```bash
  python eval/video/make_ssv2_csv.py --videos <dir of the .webm files> --labels <dir of the label files> --out ssv2_csv
  git clone https://github.com/facebookresearch/vjepa2
  git -C vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
  ```
  The csv files hold absolute video paths (no spaces). `splits.json` records the split sizes and, in
  `basename_sha256_16`, a hash of the split that does not depend on where the videos are stored.
* **Weights**: `ALPS_WEIGHTS` / `ALPS_STAGE1_WEIGHTS` (or `--weights`) for ALPS; the public encoders as in
  `eval/encoders.py` (timm and Hugging Face downloads; C-RADIOv3 needs `RADIO_REPO` and `RADIO_CKPT`).

## Environments

The numbers of the paper were produced with these package versions:

| | DAVIS | SSv2 |
|---|---|---|
| Python | 3.9 | 3.11 |
| torch / torchvision | 2.3.0 / 0.18.0 | 2.5.1 / 0.20.1 |
| timm | 1.0.28 | 1.0.29 |
| transformers (V-JEPA 2) | 4.57.6 | 4.57.6 |
| numpy | 1.26.4 | 1.26.4 |
| other | opencv-python 4.6.0.66, Pillow; scikit-image, pandas, scipy for davis2017-evaluation | decord 0.6.0, av 18.1.0, Pillow 9.4.0, einops 0.8.2, the requirements of the vjepa2 repository |

The SSv2 run of V-JEPA 2 used the DAVIS environment. V-JEPA 2 needs transformers >= 4.53.

DAVIS involves no training, but its bfloat16 numbers depend on the numerical kernels: the same ALPS run gives .533 with
the DAVIS environment above and .540 with the SSv2 environment (torch 2.5.1). The release ALPS module (`alps/vit.py`)
also computes differently from the PyTorch port that ran the paper's evaluations (LayerNorm epsilon 1e-6 as in
training instead of 1e-5, fused attention, patch embedding in float32); the tokens agree to a cosine above 0.99999 in
float32, so expect bfloat16 J&F within a few thousandths of the table and closer agreement with `--precision fp32`.
Measured with the released weights in the DAVIS environment above (torch 2.3): ALPS J&F .536 with bfloat16 (table .533)
and .5345 with `--precision fp32` (the Precision paragraph: .535).
SSv2 training is not bitwise reproducible (augmentation in data-loader workers whose streams depend on the numbers of
GPUs and workers, cuDNN autotuning).

## Commands

DAVIS, one GPU (a few minutes per image encoder, about 15 minutes for V-JEPA 2):

```bash
export ALPS_WEIGHTS=alps_vitb16.safetensors ALPS_STAGE1_WEIGHTS=alps_stage1_vitb16.safetensors
for e in tipsv2_b14 dinov2_b14_timm dinov3_b16 cradio_v3_b clip_b16_qg siglip2_b16 pe_core_b16 vjepa2_vitl alps_stage1 alps; do
  python eval/video/davis.py --enc $e --davis DAVIS --eval_repo davis2017-evaluation --out runs/davis/$e
done
python eval/video/davis.py --enc alps --precision fp32 --davis DAVIS --eval_repo davis2017-evaluation --out runs/davis_fp32/alps
python eval/video/davis.py --enc vjepa2_vitl --vjepa2_rope recompute --davis DAVIS --eval_repo davis2017-evaluation \
    --out runs/davis/vjepa2_vitl_recompute
python eval/video/summarize_davis.py runs/davis/* --paired runs/davis/alps runs/davis/alps_stage1
```

Any checkpoint of the ALPS architecture (e.g. a Stage-1 ablation variant or a Stage-2 variant, converted with
`tools/convert_checkpoint.py`) runs as `--enc alps --weights <file>`; a timm encoder with other weights (SigLIP 2 or
CLIP refined with Stage 2, Sec. 5.2) as `--enc siglip2_b16 --state_dict <file>` (loaded strictly into the timm model).
`--videos a,b` predicts the masks of a subset without scoring them. To check a DAVIS setup against a published
number, use `--enc dino_b16` (DINO ViT-B/16): J&F .622 in our runs of this protocol, .623 in the DINO paper.

SSv2, 30 to 40 GPU-hours per encoder (24 GB per GPU suffices for the ViT-B encoders at per-GPU batch 32):

```bash
export SSV2_DATA=$PWD/ssv2_csv VJEPA2_ROOT=$PWD/vjepa2
mkdir -p runs/ssv2 && INIT=runs/ssv2/probe_init.pt
python eval/video/make_probe_init.py --out $INIT        # the shared initialization (torch seed 0)
for e in tipsv2_b14 dinov2_b14_timm dinov3_b16 cradio_v3_b clip_b16_qg siglip2_b16 pe_core_b16 alps_stage1 alps; do
  bash eval/video/run_ssv2.sh $e 0,1,2,3,4,5,6,7 1 --init_from $INIT    # or 4 GPUs with accum 2
done
# V-JEPA 2: the same initialization with its input adapter re-drawn for 1,024-d tokens and 8 tubelets
python eval/video/make_probe_init.py --base $INIT --in_dim 1024 --t_tokens 8 --out runs/ssv2/probe_init_vjepa2.pt
bash eval/video/run_ssv2.sh vjepa2_vitl 0,1,2,3 2 --init_from runs/ssv2/probe_init_vjepa2.pt
python eval/video/summarize_ssv2.py runs/ssv2/*
python eval/video/ssv2_pair.py runs/ssv2/alps runs/ssv2/alps_stage1
```

A run resumes from `<out>/latest.pt` at the last finished epoch (`--no_resume` starts over); `--eval_only <probe.pt>`
evaluates a saved probe on the validation set; `--limit_train / --limit_eval / --max_steps` are for smoke tests.

## Numbers of the paper

Table S19 (temporal transfer), DAVIS-2017 J&F and SSv2 top-1 (%):

| Encoder | `--enc` | DAVIS | SSv2 |
|---|---|---|---|
| TIPSv2-B/14 | `tipsv2_b14` | .652 | 57.8 |
| DINOv2-B/14 | `dinov2_b14_timm` | .661 | 63.7 |
| DINOv3-B/16 | `dinov3_b16` | .699 | 61.8 |
| C-RADIOv3-B | `cradio_v3_b` | .659 | 62.1 |
| CLIP-B/16 | `clip_b16_qg` | .574 | 43.3 |
| SigLIP 2-B/16 | `siglip2_b16` | .503 | 49.4 |
| PE-Core-B/16 | `pe_core_b16` | .392 | 42.2 |
| V-JEPA 2-L/16 (reference; SSv2 is in its pretraining data) | `vjepa2_vitl` | .526 | 70.3 |
| ALPS (Stage 1) | `alps_stage1` | .456 | 56.4 |
| ALPS | `alps` | .533 | 55.6 |

Other video numbers of the paper, all with the protocols above:

| Statement | Run | Number |
|---|---|---|
| V-JEPA 2 without RoPE interpolation (Temporal transfer) | `davis.py --enc vjepa2_vitl --vjepa2_rope recompute` | DAVIS .347 |
| 32-bit inference (Precision) | `davis.py --enc alps_stage1 / alps --precision fp32` | DAVIS .462 / .535 |
| SigLIP 2 + Stage 2 (Sec. 5.2) | `--enc siglip2_b16 --state_dict <refined>` | DAVIS .552 (SSv2 51.2, not printed in the paper) |
| CLIP + Stage 2 (Sec. 5.2) | `--enc clip_b16_qg --state_dict <refined>` | DAVIS .585 |
| SigLIP 2 + Stage 2 with encoder learning rate 3e-6 | `--enc siglip2_b16 --state_dict <refined>` | DAVIS .553 |
| Stage-2 variants of the image source and the augmentation (supplementary): default images, web images (DataComp), no photometric augmentation, no local crops, neither | `davis.py --enc alps --weights <variant>` | DAVIS .533, .523, .525, .539, .533 |
| Stage-1 objective ablation (dense-correspondence test), after Stage 1 / after Stage 2: all four objectives, no reconstruction, no dense correspondence, no captioning, contrastive only | `davis.py --enc alps --weights <variant>` | DAVIS .589 / .582, .554 / .584, .573 / .580, .581 / .584, .566 / .572 |

The refined SigLIP 2 / CLIP and the variant checkpoints are not part of this release.

## Tests

```bash
pytest eval/video/tests/                                   # CPU; the V-JEPA 2 tests need transformers >= 4.53
```

`test_label_propagation.py` checks the propagation against a direct per-token implementation of DINO's rule, the
ALPS position table at DAVIS grids, and the whole DAVIS pipeline on a synthetic clip; `test_video_encoders.py` the input
conversion of every encoder type and the timm creation of the DINOv2 row; `test_vjepa2_rope.py` the RoPE handling on a
small random V-JEPA 2; `test_ssv2_probe.py` and `test_ssv2_splits.py` the probe, its initialization, the schedule, the
data pipeline and the splits.
