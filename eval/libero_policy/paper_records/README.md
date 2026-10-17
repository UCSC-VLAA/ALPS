# Records of the LIBERO policies of the ALPS paper

This folder lists every LIBERO policy that the paper reports, with its training seed, training budget, evaluated checkpoint and
evaluation settings, and holds the configuration and the evaluation records of each run: the outcome of every one of the
94,500 evaluation rollouts. `python ../scripts/analyze_policies.py paper --check` recomputes from these records every policy
number printed in the paper (560 values) and compares them with `../paper_numbers.json`.

The records were copied from the run directories of the two training machines, here called `machine_a` and `machine_b`, and
prepared for publication: machine names replaced by these labels, local paths replaced by placeholders, run directories named
like the runs of `../scripts/train_policy.sh`, and LIBERO's `torch.save` training log converted to JSON. Nothing else was
changed: the evaluation records, the CSV columns and the instruction embeddings are the original values.

## Files

| file | content |
|---|---|
| `policies.csv` | One row per policy, 63 in total: suite, encoder and weights, instruction embedding, training seed, epochs, batch size, optimizer, schedule, evaluated checkpoint (file name, md5, bytes), the three evaluations, mean success, the value printed in the paper, the machine, the run directory (relative to this folder) and the md5 of the run's configuration. |
| `evaluations.csv` | One row per evaluation, 189 in total (3 per policy): the evaluation and its seed, 500 rollouts, the success in %, and the success of each of the 10 tasks. |
| `retrainings.csv` | The three independent trainings of ALPS per suite and instruction embedding: per training seed the success per suite and the mean, the mean of the three, and the own-text-tower minus BERT difference per seed. |
| `retrainings_summary.json` | The summary that the paper reports: BERT 82.1%, own text tower 85.2%, paired gain +3.1 points, 95% interval -0.9 to +7.1 points. |
| `raw/<machine>/<LIBERO benchmark>/Multitask/<run>/run_<n>/` | Per run: `config.json` (LIBERO's resolved training configuration), `cell_id.txt` (the run tag), `multitask_auc.json` (LIBERO's training log: the mean training loss of the last epoch, and a success of 0 because no evaluation ran during training), and the evaluation files `eval50.json`, `eval50_repA.json`, `eval50_repB.json` (per task: success and the outcome of each of the 50 rollouts, index = initial state) with their per-task partial files `eval_tasks*.json` and `eval_rollouts*.json`. |
| `raw/task_embeddings/` | The instruction embeddings of the own text towers for the 40 LIBERO instructions, used by the "own text tower" policies: the projected text embedding, times one global factor so that its mean norm equals that of the BERT embeddings. Unchanged files (md5 below). |
| `MANIFEST.sha256` | sha256 of every file of this folder. |

The columns `checkpoint_md5`, `checkpoint_bytes` and `config_md5` of `policies.csv` describe the ORIGINAL files of the runs: the
checkpoints (about 380 MB each with the frozen network; 35 to 37 MB for the eight C-RADIOv3-B policies, which were trained with
the release code and hold only the trainable weights; not included) and the `config.json` files before the replacements
below; they identify the runs but cannot be checked against the files here. `MANIFEST.sha256` covers the files as
distributed.

Replacements in the files of the runs:

| where | original | here |
|---|---|---|
| directory names, `policies.csv` `host` | the names of the two machines | `machine_a`, `machine_b` |
| `policies.csv` `run_dir`; `rundir`, `ckpt` of the evaluation files | absolute paths on the machines | paths relative to this folder |
| run directory names, `cell_id.txt`, `experiment_dir`, `experiment_name` | the research code's run names | `BCTransformerPolicy_seed<seed>_<encoder>_<suite>_e<epochs>_s<seed>[_<embeddings>]` (`train_policy.sh` tags) |
| `policies.csv` `encoder_weights` | local files | `alps_vitb16.safetensors`, `alps_stage1_vitb16.safetensors`, `timm:<model>`, `radio:c-radio_v3-b_half.pth.tar`, `timm:vit_base_patch16_siglip_224.v2_webli+siglip2_b16_posttrained.pt` (SigLIP 2-B/16 refined by Stage 2, a timm state dict) |
| `policies.csv` `task_embedding_source` | local files | `bert`, or `file:raw/task_embeddings/<model>_task_embs.npz` |
| `config.json` `folder`, `bddl_folder`, `init_states_folder` | local paths | `<LIBERO_DATA>`, `<LIBERO_ROOT>/libero/libero/bddl_files`, `<LIBERO_ROOT>/libero/libero/init_files` |
| `config.json` `task_embedding_format` | `file:<local path>` | `file:<TASK_EMBEDDINGS_DIR>/<model>_task_embs.npz` |
| `config.json` `network` and the encoder key of `network_kwargs` | the research code's module and key names, local weight files | `FrozenEncoderPool`, `encoder_ckpt`; weights `<alps_vitb16.safetensors>`, `<alps_stage1_vitb16.safetensors>`, `<siglip2_b16_posttrained.pt>`; `timm:<model>` unchanged |
| `multitask_auc.log` (`torch.save` of two arrays) | binary | `multitask_auc.json`, the same float64 values |
| the eight C-RADIOv3-B runs: run tag, `network_kwargs.weights` | `..._evalmode` (tag of the re-training), the local RADIO checkpoint | the tag without the suffix; empty (read from `RADIO_CKPT`) |

The configurations are those of the research code, which named the encoder in one key (`encoder_ckpt` here); the release
module takes `encoder` and `weights` instead. The eight C-RADIOv3-B policies of Table 2 were re-trained in October 2026 with the
release code (`../scripts/train_policy.sh`, the frozen network in eval mode, otherwise the settings below), so their
configurations already use the release keys. `../scripts/evaluate_policy.py` reads these configurations directly (placeholders
resolved through its options and `TASK_EMBEDDINGS_DIR`), for example to evaluate a checkpoint of the paper if one is published.

## Where the policies appear in the paper

- **Table 2:**
  - 28 policies with BERT instructions: CLIP-B/16, SigLIP 2-B/16, DINOv2-B/14, DINOv3-B/16, C-RADIOv3-B, ALPS (Stage 1) and ALPS, each on four suites.
  - 16 policies with each encoder's own text tower: CLIP, SigLIP 2, C-RADIOv3 and ALPS.
  - Every cell of the table is the mean of the three evaluations of one trained policy. Tables S8 and S9 use the same policies.
- **Supplementary Sec. S3, the refined "SigLIP 2-B/16 + Stage 2":** 3 policies, on Long, Goal and Spatial.
- **Supplementary Sec. S2, the paragraph on retraining ALPS:** three independent trainings per suite with each instruction
  embedding, 24 policies. Eight of them are the ALPS policies of Table 2.

All 47 suite values that the paper prints for these policies (44 in Table 2, 3 in Sec. S3) are the column `value_in_paper`.

## Settings

**Matched within a suite.** Every encoder and both instruction embeddings use the same settings within a suite, including the
training seed. The suites differ in epochs and seed:

| suite | epochs | training seed, Table 2 and Sec. S3 | training seeds, ALPS retrainings |
|---|---|---|---|
| LIBERO-Long (`LIBERO_10`) | 25 | 1 | 0, 1, 2 |
| LIBERO-Goal | 50 | 0 | 0, 1, 2 |
| LIBERO-Spatial | 50 | 0 | 0, 1, 2 |
| LIBERO-Object | 50 | 0 | 0, 1, 2 |

The training seed sets the Python, NumPy and PyTorch generators and cuDNN determinism, through LIBERO's `control_seed`.

**Settings shared by every policy** (they are also in `config.json` of every run):

- **Observation:** agent view and eye-in-hand cameras, 128 x 128 pixels, flipped upright; each frame resized bilinearly to
  224 x 224 and passed through the frozen encoder under bfloat16 autocast, tokens cast to 32 bit; per camera an attention pool
  with one query from the task embedding, 4 heads, output size 64; proprioception: joint positions (7) and gripper state (2).
- **Policy:** the transformer policy of LIBERO (`BCTransformerPolicy`): 4 layers, 6 heads, width 64, MLP 256, dropout 0.1;
  context of 10 steps; Gaussian-mixture action head with 5 modes; 8.8M trainable parameters for every encoder.
- **Training:** 50 demonstrations per task, 10 tasks per suite; AdamW, learning rate 1e-4, betas (0.9, 0.999), weight decay
  1e-4; cosine annealing per epoch down to 1e-5, no warm-up; batch of 32 windows of 10 steps; gradient clipping at norm 100;
  color jitter with p = 0.9 and a random shift of up to 4 pixels; no evaluation during training. The frozen network was in
  training mode during the training epochs (it was a submodule of the policy), which changes nothing for these encoders,
  except for C-RADIOv3-B (its cropped position embedding is randomized and dropped for 10% of the images): its policies were
  trained with the frozen network in eval mode (`encoder_train_mode=eval`).
- **Evaluation:** the last epoch's checkpoint (`multitask_model_ep25.pth` on Long, `multitask_model_ep50.pth` otherwise),
  evaluated three times: `eval50` and `eval50_repA` with no seed set, `eval50_repB` with `EVAL_SEED=1`. Each evaluation runs 50
  rollouts per task, one for each fixed LIBERO initial state; episodes last at most 600 steps and succeed when the goal
  predicates hold. The simulator is never seeded, so the three evaluations differ only in the actions sampled from the policy.

## Code

- **LIBERO:** commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` with 11 modified files, identical on both machines.
  `../patches/libero_8f1084e3.patch` holds the changes that these policies use.
- **Frozen-encoder module:** the versions on the two machines differed only in code paths that the other machine's runs did
  not use (the C-RADIO branch on one, an unused multi-GPU path on the other). `../frozen_encoder_bridge.py` is the release
  version, which also trained and evaluated the eight C-RADIOv3-B policies (on `machine_a`, LIBERO at the same commit with
  `../patches/libero_8f1084e3.patch`).
- **Evaluation:** the same two scripts on both machines (one for `eval50`, one for the repetitions); `../scripts/evaluate_policy.py`
  is the release version.
- **Instruction embeddings** (md5, identical on both machines and here): `alps_task_embs.npz` `e75b0e731a463ffeef974f105be1e456`,
  `clip_task_embs.npz` `966661cf864d5d664334d36e383d1353`, `cradio_task_embs.npz` `edfee8b71d8c36c2e44b8fa8e6493e98`,
  `siglip2_task_embs.npz` `b1d6ba5e8e18794acc5b491a8d807cd3`.
- **Encoder weights:** the column `encoder_weights` of `policies.csv`. ALPS and ALPS (Stage 1) ran through an earlier PyTorch
  port of the released weights (`../frozen_encoder_bridge.py`, `alps_port_numerics`).

## Retrainings of ALPS

- **What was trained:** every suite three times with each instruction embedding, with training seeds 0, 1 and 2
  (`retrainings.csv`). Each policy is the mean of its three evaluations.
- **Result:** averaged over the four suites and the three trainings, BERT reaches 82.1% and the own text tower 85.2%, a paired
  gain of 3.1 points.
- **Interval:** a bootstrap with 10,000 resamples gives a 95% interval of -0.9 to +7.1 points. It resamples the three trainings
  (one draw for both embeddings and all suites), the ten tasks of every suite and the rollouts (`../scripts/analyze_policies.py`,
  bootstrap seed 20261006).
- **Reading:** the interval includes zero, so the paper does not claim a stable or significant improvement.
- **Scope of the other intervals:** the intervals of Table S8 instead resample tasks and initial states of fixed
  trained policies. They do not include the randomness of training.

## Provenance

The list was built from a read-only listing of every run on the two machines (2026-10-11). For every policy it was checked that
the seed and epochs in `config.json` match the run, that the evaluated checkpoint is the last epoch, that there are three
evaluations of 500 rollouts with the seeds listed above, and that the mean matches the printed value. The records of the eight
C-RADIOv3-B policies were replaced on 2026-10-20 by those of their re-training, after the same checks.
