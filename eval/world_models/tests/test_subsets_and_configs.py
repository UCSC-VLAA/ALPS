# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests: the trajectory order files reproduce the paper's files byte for byte; the run configurations parse and carry
the settings of the paper; the planning settings of the scripts agree with the trainer."""
import hashlib
import os
import pickle
import re
import subprocess
import sys
import tempfile

import pytest
import torch

from _load import OVERLAY, WM, load

yaml = pytest.importorskip("yaml")
MS = load(os.path.join(OVERLAY, "make_subsets.py"), "wm_make_subsets")
TRAINER = load(os.path.join(OVERLAY, "wm_train.py"), "wm_train_under_test")
CONF = os.path.join(OVERLAY, "conf")


def test_subset_files_match_the_paper():
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        os.makedirs(os.path.join(data, "pusht_noise", "train"))
        with open(os.path.join(data, "pusht_noise", "train", "seq_lengths.pkl"), "wb") as f:
            pickle.dump([100] * 18685, f)
        for sub, n in (("wall_single", 1920), ("point_maze", 2000), ("deformable/granular", 1000), ("deformable/rope", 1000)):
            os.makedirs(os.path.join(data, sub))
            torch.save(torch.zeros(n, 1), os.path.join(data, sub, "states.pth"))
        out = os.path.join(d, "subsets")
        subprocess.run([sys.executable, os.path.join(OVERLAY, "make_subsets.py"), "--data", data, "--out", out], check=True, capture_output=True)
        for name, md5 in MS.EXPECTED_MD5.items():
            assert hashlib.md5(open(os.path.join(out, name), "rb").read()).hexdigest() == md5, name


def merged(task, enc):
    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    cfg = merge(yaml.safe_load(open(os.path.join(CONF, "wm", "base.yaml"))), yaml.safe_load(open(os.path.join(CONF, "wm", "task", f"{task}.yaml"))))
    cfg["encoder"] = yaml.safe_load(open(os.path.join(CONF, "encoder", f"frozen_{enc}.yaml")))
    return cfg


@pytest.mark.parametrize("task,hist,skip", [("pusht", 3, 5), ("wall", 1, 5), ("point_maze", 3, 5), ("granular", 1, 1), ("rope", 1, 1)])
def test_task_configs(task, hist, skip):
    for enc in ("alps", "alps_stage1", "dinov2_b14", "dinov3_b16", "siglip2_b16", "cradio_v3_b"):
        c = merged(task, enc)
        assert c["task"] == task and c["num_hist"] == hist and c["frameskip"] == skip and c["num_pred"] == 1
        assert c["model"]["num_hist"] == "${num_hist}" and c["has_decoder"] is False and c["concat_dim"] == 1
        assert c["predictor"]["_target_"] == "models.vit_proj.ProjViTPredictor" and c["predictor"]["depth"] == 6 and c["predictor"]["proj_dim"] == 384
        assert c["training"] == {"batch_size": 32, "predictor_lr": 0.0005, "action_encoder_lr": 0.0005}
        assert c["encoder"]["_target_"] == "models.encoder.frozen_encoder.FrozenTokenEncoder" and c["encoder"]["tf32"] is True
        assert c["encoder"]["res"] == (196 if enc == "dinov2_b14" else 224)
        assert c["env"]["dataset"]["split_ratio"] == 0.9 and c["env"]["dataset"]["transform"]["img_size"] == 224


def test_make_run_config_resolves():
    pytest.importorskip("omegaconf")
    MRC = load(os.path.join(OVERLAY, "make_run_config.py"), "wm_make_run_config")
    cfg = MRC.build_config("rope", "alps", stats="/x/stats.npz", subsets="/x/subsets", data="/x/data", weights="/x/alps.safetensors")
    assert cfg.env.dataset.data_path == "/x/data/deformable" and cfg.env.dataset.traj_subset == "/x/subsets/rope_100.json"
    assert cfg.model.num_hist == 1 and cfg.encoder.weights == "/x/alps.safetensors" and "data_subdir" not in cfg


def test_plan_configs():
    for t in ("granular", "rope"):
        c = yaml.safe_load(open(os.path.join(CONF, f"plan_{t}.yaml")))
        sp = c["planner"]["sub_planner"]
        assert c["n_evals"] == 10 and c["seed"] == 99 and c["goal_source"] == "random_state" and c["objective"]["alpha"] == 1
        assert (sp["horizon"], sp["topk"], sp["num_samples"], sp["opt_steps"]) == (5, 30, 300, 10)


def test_script_settings_agree_with_the_trainer():
    text = open(os.path.join(WM, "dino_wm", "scripts", "common.sh")).read()
    for task, (total, _, _) in TRAINER.SCHEDULES.items():
        m = re.search(r"^\s*%s\)\s+T=(\d+);" % task, text, re.M)
        assert m and int(m.group(1)) == total, task
