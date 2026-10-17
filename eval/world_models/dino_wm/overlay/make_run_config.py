# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Writes the configuration of one world model of the ALPS paper: <run dir>/hydra.yaml.

  python make_run_config.py --task <pusht|wall|point_maze|granular|rope> --encoder <alps|alps_stage1|dinov2_b14|dinov3_b16|
      siglip2_b16|cradio_v3_b> --stats <npz of fit_stats.py> --subsets <dir of make_subsets.py> --out <run dir>
      [--data $DATASET_DIR] [--weights <safetensors of alps / alps_stage1>]

hydra.yaml = conf/wm/base.yaml + conf/wm/task/<task>.yaml + conf/encoder/frozen_<encoder>.yaml, with the data folder, the
trajectory order file and the token statistics filled in, resolved. wm_train.py trains from it; plan.py
(ckpt_base_path=<runs root> model_name=<run name>, i.e. <run dir> = <runs root>/outputs/<run name>) and the prediction
evaluation re-instantiate the frozen encoder from it.
"""
import argparse
import os

from omegaconf import OmegaConf, open_dict

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ("pusht", "wall", "point_maze", "granular", "rope")
ENCODERS = ("alps", "alps_stage1", "dinov2_b14", "dinov3_b16", "siglip2_b16", "cradio_v3_b")


def build_config(task, encoder, stats, subsets, data=None, weights=None, conf_dir=None):
    if task not in TASKS or encoder not in ENCODERS:
        raise ValueError(f"unknown task {task!r} or encoder {encoder!r}")
    conf_dir = conf_dir or os.path.join(HERE, "conf")
    data = data or os.environ.get("DATASET_DIR")
    if not data:
        raise ValueError("pass --data or set DATASET_DIR")
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(conf_dir, "wm", "base.yaml")),
                          OmegaConf.load(os.path.join(conf_dir, "wm", "task", f"{task}.yaml")))
    with open_dict(cfg):
        cfg.encoder = OmegaConf.load(os.path.join(conf_dir, "encoder", f"frozen_{encoder}.yaml"))
        cfg.encoder.stats = os.path.abspath(stats)
        if encoder in ("alps", "alps_stage1"):
            cfg.encoder.weights = os.path.abspath(weights) if weights else None
        cfg.env.dataset.data_path = os.path.join(os.path.abspath(data), cfg.data_subdir)
        cfg.env.dataset.traj_subset = os.path.join(os.path.abspath(subsets), f"{task}_100.json")
        del cfg["data_subdir"]
    OmegaConf.resolve(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--encoder", required=True, choices=ENCODERS)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--subsets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="DINO-WM data folder (default: env DATASET_DIR)")
    ap.add_argument("--weights", default=None, help="ALPS safetensors (default: env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS at run time)")
    a = ap.parse_args()
    for p in (a.stats, os.path.join(a.subsets, f"{a.task}_100.json")):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    cfg = build_config(a.task, a.encoder, a.stats, a.subsets, a.data, a.weights)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "hydra.yaml")
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists (one configuration per run directory)")
    with open_dict(cfg):
        cfg.run_name = os.path.basename(os.path.abspath(a.out))
        cfg.saved_folder = os.path.abspath(a.out)
    OmegaConf.save(cfg, path)
    print(f"wrote {path}: task {a.task}, encoder {a.encoder}, data {cfg.env.dataset.data_path}, stats {cfg.encoder.stats}")


if __name__ == "__main__":
    main()
