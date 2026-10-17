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
"""Record rollouts of a trained policy for the rollout figure of the paper (visualization only, no new number).

    cd <LIBERO root> && PYTHONPATH=. MUJOCO_GL=egl ALPS_REPO=<this repository> \
        python record_rollouts.py --run-dir <run dir> --tasks 0 1 2 3 4 5 6 7 8 9 --init 0 --stride 5 --out rollouts_spatial.npz

The loop of the benchmark with one simulator (libero.lifelong.metric): the fixed initial state --init, 5 zero-action steps,
then the policy acts until success or 600 steps; the agent-view frame of every --stride-th step (and of the last one) is
stored upright. The figure of the paper: the BERT policies of ALPS on LIBERO-Spatial and LIBERO-Goal, tasks 0-9, initial
state 0, stride 5; the policy samples its actions, so a recorded rollout can end differently from the benchmark rollout of
the same initial state (5 of the 20 recorded rollouts of the paper did). Output npz: frames_<task> [frames, 128, 128, 3]
uint8 and meta_<task> (JSON: task, language, success, steps, init, stride, suite, benchmark outcome).
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--init", type=int, default=0, help="index of the fixed initial state")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--encoder", default=None, help="as in evaluate_policy.py (run configs of the paper)")
    ap.add_argument("--encoder-weights", default=None, help="as in evaluate_policy.py")
    ap.add_argument("--task-embeddings", default=None, help="as in evaluate_policy.py")
    a = ap.parse_args()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from libero.libero import benchmark as bmklib
    from libero.libero import get_libero_path
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.utils import get_task_embs, safe_device
    from evaluate_policy import adapt_network_kwargs, default_checkpoint, load_policy_weights, resolve_task_embeddings

    rundir = os.path.abspath(a.run_dir)
    cfg = OmegaConf.create(json.load(open(os.path.join(rundir, "config.json"))))
    cfg.device = a.device
    cfg.eval.num_workers = 0
    adapt_network_kwargs(cfg.policy.image_encoder.network_kwargs, a.encoder, a.encoder_weights)
    resolve_task_embeddings(cfg, a.task_embeddings)
    for key, lk in (("folder", "datasets"), ("bddl_folder", "bddl_files"), ("init_states_folder", "init_states")):
        if not os.path.isdir(str(cfg.get(key) or "")):
            cfg[key] = get_libero_path(lk)
    bmk = bmklib.get_benchmark_dict()[str(cfg.benchmark_name).lower()]()
    n = bmk.n_tasks
    get_dataset(dataset_path=os.path.join(cfg.folder, bmk.get_task_demonstration(0)), obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True, seq_len=cfg.data.seq_len)
    task_embs = get_task_embs(cfg, [bmk.get_task(i).language for i in range(n)])
    bmk.set_task_embs(task_embs)
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n, cfg), a.device)
    algo.policy.load_state_dict(load_policy_weights(default_checkpoint(rundir, cfg.train.n_epochs), a.device))
    algo.eval()
    stored_path = os.path.join(rundir, "eval50.json")
    stored = json.load(open(stored_path)) if os.path.exists(stored_path) else None
    res = {}
    for ti in a.tasks:
        task = bmk.get_task(ti)
        args = {"bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
                "camera_heights": cfg.data.img_h, "camera_widths": cfg.data.img_w}
        env = DummyVectorEnv([lambda: OffScreenRenderEnv(**args)])
        init_states = torch.load(os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file))
        env.reset()
        algo.reset()
        obs = env.set_init_state(init_states[a.init:a.init + 1])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros((1, 7)))
        frames = [obs[0]["agentview_image"][::-1].copy()]           # stored upside down; shown upright
        steps, success = 0, False
        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1
                obs, reward, done, info = env.step(algo.policy.get_action(raw_obs_to_tensor_obs(obs, task_embs[ti], cfg)))
                if steps % a.stride == 0 or done[0]:
                    frames.append(obs[0]["agentview_image"][::-1].copy())
                if done[0]:
                    success = True
                    break
        env.close()
        bench = None if stored is None else bool(stored["per_rollout"][str(ti)][a.init])
        print("task %d | %s | success %s in %d steps | benchmark outcome of this initial state %s" % (
            ti, task.language, success, steps, bench), flush=True)
        res["frames_%d" % ti] = np.stack(frames)
        res["meta_%d" % ti] = json.dumps({"task": task.name, "language": task.language, "success": success, "steps": steps,
                                          "init": a.init, "stride": a.stride, "suite": str(cfg.benchmark_name),
                                          "benchmark_outcome": bench,
                                          "success_rate": None if stored is None else stored["per_task"][ti]})
    np.savez_compressed(a.out, **res)
    print("saved", a.out)


if __name__ == "__main__":
    main()
