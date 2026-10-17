#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a trained LIBERO policy with the protocol of the ALPS paper, recording the outcome of every rollout.

    python evaluate_policy.py --run-dir <run dir> --tag eval50                # eval50.json       no seed set
    python evaluate_policy.py --run-dir <run dir> --tag repA                  # eval50_repA.json  no seed set
    python evaluate_policy.py --run-dir <run dir> --tag repB --eval-seed 1    # eval50_repB.json  EVAL_SEED=1

Every policy of the paper was evaluated these three times (evaluate_policy.sh runs them). Protocol: the last checkpoint
(multitask_model_ep<n_epochs>.safetensors of the released policies, else LIBERO's .pth), 50 rollouts per task, one from each fixed LIBERO initial state; LIBERO's own loop
(libero.lifelong.metric.evaluate_one_task_success: reset, set the initial state, 5 zero-action steps, at most 600 steps,
success when the goal predicates hold at any step, actions sampled from the policy's Gaussian mixture). Parallel simulators:
10 on LIBERO-10 (5 loops of 10 = the same 50 initial states; 20 simulators of LIBERO-10 scenes did not fit next to a training
job), otherwise eval.num_procs of the run config (20). The simulator is never seeded, so evaluations differ only in the
sampled actions; with --eval-seed the python / numpy / torch generators are seeded after the policy is loaded, right before
the rollouts. Output (in the run directory): eval50[_<tag>].json with per_task, suite_mean and per_rollout[task] (50 booleans,
index = initial state), plus partial files written after every task; tag eval50 resumes from them, the other tags restart.

Run from the root of the patched LIBERO checkout with PYTHONPATH=<LIBERO root>, MUJOCO_GL=egl and ALPS_REPO=<this repository>
(evaluate_policy.sh sets them). The script must stay a real file with a __main__ guard: the simulators are started with
spawn, which re-imports it.
"""
import argparse
import json
import os
import random

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained LIBERO policy (ALPS protocol).")
    ap.add_argument("--run-dir", required=True, help="run directory with config.json and multitask_model_ep<N>.safetensors / .pth")
    ap.add_argument("--tag", default="eval50", help="eval50 (output eval<N>.json, resumable) or a tag such as repA / repB")
    ap.add_argument("--eval-seed", type=int, default=None, help="seed the generators before the rollouts (repB: 1)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-rollouts", type=int, default=50, help="rollouts per task (paper: 50)")
    ap.add_argument("--num-procs", type=int, default=None, help="parallel simulators (default: 10 on LIBERO-10, else the config's)")
    ap.add_argument("--checkpoint", default=None, help="default: multitask_model_ep<train.n_epochs>.safetensors, else .pth, of the run")
    ap.add_argument("--datasets", default=None, help="LIBERO datasets folder (default: the config's, else get_libero_path)")
    ap.add_argument("--bddl", default=None, help="bddl folder (default: the config's, else get_libero_path)")
    ap.add_argument("--init-states", default=None, help="init_files folder (default: the config's, else get_libero_path)")
    ap.add_argument("--encoder", default=None, help="override network_kwargs.encoder (run configs of the paper: "
                    "needed when their encoder_ckpt names a weights file, e.g. alps, alps_stage1, siglip2_b16)")
    ap.add_argument("--encoder-weights", default=None, help="override network_kwargs.weights (e.g. on another machine)")
    ap.add_argument("--task-embeddings", default=None, help="override the .npz of task_embedding_format=file:<npz> "
                    "(paper_records configs: or set TASK_EMBEDDINGS_DIR)")
    ap.add_argument("--force", action="store_true", help="evaluate even if the output file exists")
    a = ap.parse_args()
    if not a.tag.replace("_", "").isalnum():
        ap.error(f"--tag must be alphanumeric, got {a.tag!r}")
    return a


# placeholders of paper_records/raw/*/config.json for the weight files of the research configs -> encoder of the release
_RESEARCH_PLACEHOLDERS = {"<alps_vitb16.safetensors>": "alps", "<alps_stage1_vitb16.safetensors>": "alps_stage1",
                          "<siglip2_b16_posttrained.pt>": "siglip2_b16"}


def adapt_network_kwargs(nk, encoder=None, weights=None):
    """network_kwargs of a run config -> the keys of FrozenEncoderPool.

    The research run configs of the paper (paper_records/raw) name the encoder in encoder_ckpt: "timm:<model>",
    "radio:<checkpoint>", a weight file, or a placeholder of paper_records. They get the encoder and weights of the release
    and the settings of the paper's runs (encoder_train_mode "follow", alps_port_numerics True); their checkpoints load
    into the release module. Configs written by the release code (those of the C-RADIOv3-B policies in paper_records) keep
    their keys; encoder and weights replace theirs when given."""
    from eval.libero_policy.frozen_encoder_bridge import PAPER_ENCODERS
    if "encoder_ckpt" in nk:
        spec = str(nk.pop("encoder_ckpt"))
        nk.pop("port_encoder_path", None)
        timm_names = {v[1]: k for k, v in PAPER_ENCODERS.items() if v[1].startswith("timm:")}
        if encoder is None and spec in timm_names:
            encoder = timm_names[spec]
        elif encoder is None and spec.startswith("radio:"):
            given = spec[len("radio:"):]
            encoder, weights = "cradio_v3_b", weights or ("" if given.startswith("<") else given)
        elif encoder is None and spec in _RESEARCH_PLACEHOLDERS:
            encoder = _RESEARCH_PLACEHOLDERS[spec]
            if encoder == "siglip2_b16" and not weights:
                raise SystemExit("SigLIP 2-B/16 + Stage 2: pass --encoder-weights <state dict of the refined SigLIP 2>")
        elif encoder is None:
            raise SystemExit(f"this run config of the paper names the weights file {os.path.basename(spec)}: "
                             "pass --encoder (alps, alps_stage1 or siglip2_b16) and --encoder-weights")
        nk.encoder = encoder
        nk.weights = os.path.abspath(weights) if weights else ""
        if "encoder_train_mode" not in nk:
            nk.encoder_train_mode = "follow"
        if "alps_port_numerics" not in nk:
            nk.alps_port_numerics = True
        print("run config of the paper: encoder_ckpt", spec, "-> encoder", nk.encoder, flush=True)
    else:
        if encoder:
            nk.encoder = encoder
        if weights:
            nk.weights = os.path.abspath(weights)
    return nk


def resolve_task_embeddings(cfg, given=None):
    """task_embedding_format=file:<npz>: --task-embeddings, or <TASK_EMBEDDINGS_DIR> (paper_records) from the environment."""
    tef = str(cfg.task_embedding_format)
    if not tef.startswith("file:"):
        if given:
            raise SystemExit("--task-embeddings: this policy was trained with task_embedding_format=%s" % tef)
        return
    path = os.path.abspath(given) if given else tef[len("file:"):]
    if "<TASK_EMBEDDINGS_DIR>" in path and os.environ.get("TASK_EMBEDDINGS_DIR"):
        path = path.replace("<TASK_EMBEDDINGS_DIR>", os.environ["TASK_EMBEDDINGS_DIR"])
    if not os.path.exists(path):
        raise SystemExit(f"no instruction embeddings {path}: pass --task-embeddings <npz> (or set TASK_EMBEDDINGS_DIR)")
    cfg.task_embedding_format = "file:" + path


def default_checkpoint(rundir, n_epochs):
    """The last checkpoint of a run: multitask_model_ep<N>.safetensors (the released policies, export_policy_weights.py)
    when present, else LIBERO's multitask_model_ep<N>.pth."""
    for ext in (".safetensors", ".pth"):
        path = os.path.join(rundir, f"multitask_model_ep{n_epochs}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(rundir, f"multitask_model_ep{n_epochs}.pth")


def load_policy_weights(path, device="cpu"):
    """State dict of a policy checkpoint: a safetensors file of the trainable parameters, or a torch checkpoint of LIBERO
    ({"state_dict", "cfg", "previous_masks"}; checkpoints of the paper's runs also hold the frozen network, which the
    frozen-encoder module drops on loading)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    from libero.lifelong.utils import torch_load_model
    sd, _, _ = torch_load_model(path, map_location=device)
    return sd


def _local_path(cfg, key, given, libero_key, get_libero_path):
    if given:
        cfg[key] = os.path.abspath(given)
    elif not cfg.get(key) or not os.path.isdir(str(cfg[key])):
        cfg[key] = get_libero_path(libero_key)
        print(f"{key}: the folder of the run config does not exist here, using {cfg[key]}", flush=True)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    rundir = os.path.abspath(args.run_dir)
    nroll = args.n_rollouts
    suffix = "" if args.tag == "eval50" else "_" + args.tag
    out = os.path.join(rundir, f"eval{nroll}{suffix}.json")
    if os.path.exists(out) and not args.force:
        print("already done:", out)
        print("EVAL_CELL_DONE", out)
        return
    cfg = OmegaConf.create(json.load(open(os.path.join(rundir, "config.json"))))
    cfg.device = args.device
    cfg.eval.n_eval = nroll
    cfg.eval.num_workers = 0
    cfg.eval.num_procs = int(args.num_procs or (10 if str(cfg.benchmark_name).upper() == "LIBERO_10" else int(cfg.eval.num_procs)))
    print("parallel simulators =", cfg.eval.num_procs, "| benchmark", cfg.benchmark_name, "| tag", args.tag, flush=True)

    from libero.libero import benchmark as bmklib
    from libero.libero import get_libero_path
    from libero.lifelong import metric
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.utils import get_task_embs, safe_device

    _local_path(cfg, "folder", args.datasets, "datasets", get_libero_path)
    _local_path(cfg, "bddl_folder", args.bddl, "bddl_files", get_libero_path)
    _local_path(cfg, "init_states_folder", args.init_states, "init_states", get_libero_path)
    import libero.lifelong.models  # noqa: F401  (imports the frozen-encoder module, which puts ALPS_REPO on the path)
    nk = adapt_network_kwargs(cfg.policy.image_encoder.network_kwargs, args.encoder, args.encoder_weights)
    resolve_task_embeddings(cfg, args.task_embeddings)
    if nk.get("weights") and not os.path.exists(str(nk.weights)):
        raise SystemExit(f"no encoder weights {nk.weights}: pass --encoder-weights <file>")

    bmk = bmklib.get_benchmark_dict()[str(cfg.benchmark_name).lower()]()
    n = bmk.n_tasks
    get_dataset(dataset_path=os.path.join(cfg.folder, bmk.get_task_demonstration(0)), obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True, seq_len=cfg.data.seq_len)              # initializes robomimic's observation specs
    descriptions = [bmk.get_task(i).language for i in range(n)]
    task_embs = get_task_embs(cfg, descriptions)
    bmk.set_task_embs(task_embs)
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n, cfg), args.device)
    ckpt = os.path.join(rundir, args.checkpoint) if args.checkpoint else default_checkpoint(rundir, cfg.train.n_epochs)
    algo.policy.load_state_dict(load_policy_weights(ckpt, args.device))
    algo.policy.eval()
    print("flip_ud =", nk.get("flip_ud", None), "| encoder", nk.get("encoder", nk.get("encoder_ckpt", None)),
          "| instructions", cfg.task_embedding_format, flush=True)

    seed = args.eval_seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print("EVAL_SEED =", seed if seed is not None else "not set", "| torch initial seed", torch.initial_seed(), flush=True)

    part = os.path.join(rundir, f"eval_tasks{suffix}.json")
    rpart = os.path.join(rundir, f"eval_rollouts{suffix}.json")
    done, rolls = {}, {}
    if args.tag == "eval50" and os.path.exists(part) and os.path.exists(rpart):
        try:
            done = {int(k): float(v) for k, v in json.load(open(part)).items()}
            rolls = {int(k): v for k, v in json.load(open(rpart)).items()}
            done = {k: v for k, v in done.items() if k in rolls}
            print(f"RESUME with {len(done)} tasks", flush=True)
        except Exception as e:                                   # a partial file cut by an interruption
            print(f"RESUME_CORRUPT {e}: starting fresh", flush=True)
            done, rolls = {}, {}
    for i in range(n):
        if i in done:
            print(f"TASK{i}_SUCCESS = {done[i]} (cached)", flush=True)
            continue
        metric.LAST_SUCCESSES.clear()
        s = metric.evaluate_one_task_success(cfg, algo, bmk.get_task(i), task_embs[i], i, sim_states=None, task_str=f"t{i}")
        done[i] = float(s)
        rolls[i] = [bool(metric.LAST_SUCCESSES.get(j, False)) for j in range(nroll)]
        for pth, obj in ((rpart, rolls), (part, done)):
            json.dump({str(k): v for k, v in obj.items()}, open(pth + ".tmp", "w"))
            os.replace(pth + ".tmp", pth)
        print(f"TASK{i}_SUCCESS = {s} ({bmk.get_task(i).language})", flush=True)
    succ = [done[i] for i in range(n)]
    res = {"rundir": rundir, "ckpt": ckpt, "n_eval": nroll, "tag": args.tag, "eval_seed": seed, "per_task": succ,
           "suite_mean": float(np.mean(succ)), "tasks": [bmk.get_task(i).name for i in range(n)],
           "per_rollout": {str(i): rolls[i] for i in range(n)}}
    json.dump(res, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print("SUITE_MEAN =", round(res["suite_mean"], 4))
    print("EVAL_CELL_DONE", out)


if __name__ == "__main__":
    main()
