# Third-party components

## Included in this repository (modified or copied)

| Component | Where | License |
|---|---|---|
| big_vision (Google) and CLIPA (UCSC-VLAA), on which the JAX training code is based | `train_jax/` | Apache-2.0 |
| BERT-base uncased WordPiece vocabulary, slots 1-2 renamed [bos], [eos] | `alps/assets/`, `train_jax/assets/` | Apache-2.0 |
| OpenAI CLIP ImageNet class names and prompt templates, as distributed with CLIP_benchmark (LAION-AI) | `alps/assets/imagenet_zeroshot.json` | MIT |
| Patch against DINO-WM (github.com/gaoyuezhou/dino_wm, commit 0a9492fa) | `eval/world_models/` | MIT |
| Patch against LIBERO (github.com/Lifelong-Robot-Learning/LIBERO, commit 8f1084e3) and robomimic 0.2.0 | `eval/libero_policy/` | MIT |
| Integration with LeWorldModel (github.com/lucas-maes/le-wm) and stable-worldmodel | `eval/world_models/reacher/` | MIT |
| DAVIS label propagation, re-implemented after the evaluation of DINO (github.com/facebookresearch/dino) | `eval/video/davis.py` | Apache-2.0 |
| flan-t5-base encoding of the empty instruction (one 768-d vector), google/flan-t5-base | `train_jax/assets/flan_t5_base_empty_instruction.npz` | Apache-2.0 |

## Used at run time, not distributed (install them yourself)

| Component | Used for | License |
|---|---|---|
| PyTorch, timm, diffusers, transformers, safetensors, huggingface_hub | inference, evaluation | BSD-3 / Apache-2.0 |
| JAX, Flax, optax, orbax, TensorFlow, TensorFlow Text | JAX training | Apache-2.0 |
| SDXL autoencoder (stabilityai/sdxl-vae) | evaluation decoder | see the model card |
| LPIPS (richzhang/PerceptualSimilarity) | reconstruction metric | BSD-2-Clause |
| V-JEPA 2 (facebookresearch/vjepa2) | SSv2 probe recipe, V-JEPA 2 baseline | MIT |
| DAVIS-2017 evaluation (davisvideochallenge/davis2017-evaluation) | DAVIS J&F | BSD-3-Clause |
| DINOv2 / dino.txt (facebookresearch/dinov2) | baselines | Apache-2.0 |
| C-RADIOv3 (NVlabs/RADIO) | baseline | see the repository (NVIDIA license) |
| OpenVision (UCSC-VLAA/OpenVision) | baselines | Apache-2.0 |
| open_clip (mlfoundations/open_clip) | CLIP / PE-Core baselines | see the repository |
| VC-1 (facebookresearch/eai-vc) | baseline | see the repository |
| robosuite, MuJoCo, dm_control, pymunk, PyFleX (AdaptiGraph) | simulators | see the repositories |

Pretrained weights of the public baselines keep their own licenses.
