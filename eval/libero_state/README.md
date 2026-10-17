# LIBERO state probe

How much physical state the frozen tokens of an encoder carry: a small attentive probe (2 cross-attention layers, width
256, 8 queries; only the probe is trained) reads object positions, symmetry-aware orientations and LIBERO predicates
from the tokens of the current frame (readout), or from 16 frames of history plus the true actions, 1 s and 3 s ahead
(future). Labels come from the simulator: the MuJoCo state stored in each demonstration is restored frame by frame.

## Data
1. LIBERO-Spatial, -Object and -Goal demonstrations (official hdf5; `benchmark_scripts/download_libero_datasets.py
   --use-huggingface` in github.com/Lifelong-Robot-Learning/LIBERO).
2. Episodes (10 Hz frames at 128 px, upright):
   `python build_episodes.py --root <libero_datasets> --suites libero_spatial libero_object libero_goal --out <episodes> --stride 2 --res 128`
3. Labels (needs a LIBERO installation with MuJoCo and EGL):
   `MUJOCO_GL=egl python label_states.py --libero_root <LIBERO> --data_root <libero_datasets> --episodes_json <episodes>/episodes.json --out <labels>`
   then `python label_states.py --out <labels> --manifest` (orientation symmetry classes from `mesh_symmetry.json`).
4. Splits: `splits/libero_probe_split.json`, per task 35 train / 5 val / 5 dev_test episodes (results are reported on
   dev_test); `splits/libero_probe_sealed_test.json` holds 5 further episodes per task that no result in the paper uses.

## Running one encoder (six cells: two suites x readout, +1 s, +3 s)
```bash
EPISODES=<episodes> LABELS=<labels> CACHE=<token cache> OUT=<runs> bash run_encoder.sh alps alps_vitb16.safetensors
EPISODES=<episodes> LABELS=<labels> CACHE=<token cache> OUT=<runs> bash run_encoder.sh dinov3_b16
python table.py <runs>            # Table 1 state columns (relative to alps_stage1, which must be in <runs>)
python summarize.py <runs>        # absolute errors, orientation, predicate F1, paired differences by episode
```
Probe settings: AdamW lr 3e-4 (weight decay 0.01), cosine, 6,000 steps, batch 128, gradient clipping 1.0, bf16
autocast in the probe, checkpoint selected on the val split; tokens standardized per channel with statistics of the
training split; seed 0. Frames are resized from 128 to 224 px with PIL bicubic before every encoder.
