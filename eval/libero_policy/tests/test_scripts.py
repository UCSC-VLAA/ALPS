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
"""CPU tests of the analysis and embedding helpers on synthetic records.

Includes the check of every printed policy number of the paper against paper_records (about a minute)."""
import ast
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
import analyze_policies as ap  # noqa: E402
import embed_instructions as ei  # noqa: E402


def write_run(d, suite="LIBERO_GOAL", p=0.8, seed=0, n_evals=3, rng=None):
    rng = rng or np.random.default_rng(0)
    os.makedirs(d, exist_ok=True)
    cfg = {"benchmark_name": suite, "seed": seed, "train": {"n_epochs": 50}, "task_embedding_format": "bert",
           "policy": {"image_encoder": {"network_kwargs": {"encoder": "alps", "weights": "/w/alps_vitb16.safetensors"}}}}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    for (tag, fname) in ap.EVALS[:n_evals]:
        r = rng.random((10, 50)) < p
        ev = {"n_eval": 50, "per_task": r.mean(1).tolist(), "suite_mean": float(r.mean()), "tasks": ["t%d" % i for i in range(10)],
              "per_rollout": {str(i): r[i].tolist() for i in range(10)}, "tag": tag}
        json.dump(ev, open(os.path.join(d, fname), "w"))
    return d


def test_load_and_success():
    with tempfile.TemporaryDirectory() as d:
        write_run(d)
        x, tasks, tags = ap.load_evaluations(d)
        assert x.shape == (3, 10, 50) and tags == ["eval50", "repA", "repB"] and len(tasks) == 10
        assert abs(ap.success(x) - 100 * x.mean()) < 1e-12
        assert np.allclose(ap.per_task(x), 100 * x.mean(axis=(0, 2)))


def test_inconsistent_record_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        write_run(d)
        p = os.path.join(d, "eval50_repA.json")
        ev = json.load(open(p))
        ev["per_task"][0] = 1.5
        json.dump(ev, open(p, "w"))
        with pytest.raises(ValueError):
            ap.load_evaluations(d)


def test_bootstrap_is_paired_and_seeded():
    rng = np.random.default_rng(1)
    a, b = (rng.random((3, 10, 50)) < 0.7).astype(float), (rng.random((3, 10, 50)) < 0.6).astype(float)
    one = ap.bootstrap_column({"a": a, "b": b}, "policy_goal", b=200)
    two = ap.bootstrap_column({"b": b, "a": a}, "policy_goal", b=200)
    alone = ap.bootstrap_column({"a": a}, "policy_goal", b=200)
    assert np.array_equal(one["a"], two["a"]) and np.array_equal(one["a"], alone["a"])     # the rows of a column share draws
    assert not np.array_equal(one["a"], ap.bootstrap_column({"a": a}, "policy_f7_goal", b=200)["a"])
    lo, hi = np.percentile(one["a"], [2.5, 97.5])
    assert lo < ap.success(a) < hi


def test_runs_mode_cli():
    with tempfile.TemporaryDirectory() as d:
        write_run(os.path.join(d, "LIBERO_GOAL", "Multitask", "BCTransformerPolicy_seed0_alps_goal_e50_s0", "run_001"))
        write_run(os.path.join(d, "LIBERO_10", "Multitask", "x", "run_001"), suite="LIBERO_10", seed=1, n_evals=1)
        out = os.path.join(d, "summary.json")
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "analyze_policies.py"), "runs", d, "--json", out],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        res = json.load(open(out))
        assert [x["suite"] for x in res] == ["LIBERO-Long", "LIBERO-Goal"]
        assert len(res[0]["evaluations"]) == 1 and len(res[1]["evaluations"]) == 3
        assert "(1 of 3 evaluations)" in r.stdout


def test_scale_to_bert():
    raw = np.random.default_rng(0).normal(size=(40, 16)).astype(np.float32) * 3
    embs, scale = ei.scale_to_bert(raw, 21.53294)
    assert embs.dtype == np.float32
    assert abs(np.linalg.norm(embs, axis=1).mean() - 21.53294) < 1e-3
    assert np.allclose(embs / scale, raw, rtol=1e-5, atol=1e-6)


def test_scripts_are_python38_syntax():
    for f in glob.glob(os.path.join(SCRIPTS, "*.py")) + glob.glob(os.path.join(HERE, "..", "*.py")):
        ast.parse(open(f).read(), filename=f, feature_version=(3, 8))


def test_every_printed_number_of_the_paper():
    """All 560 policy numbers printed in the paper, recomputed from paper_records (about a minute: the retraining bootstrap)."""
    cmd = [sys.executable, os.path.join(SCRIPTS, "analyze_policies.py"), "paper", "--check"]
    if os.environ.get("LIBERO_POLICY_CONFIG"):
        cmd += ["--config", os.environ["LIBERO_POLICY_CONFIG"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "560 of 560 printed values reproduced" in r.stdout and "MISMATCH" not in r.stdout


RECORDS = os.path.join(HERE, "..", "paper_records")


def test_paper_records_are_complete_and_clean():
    """Every file in MANIFEST.sha256; text files ASCII; no absolute path anywhere; local paths only as placeholders."""
    import csv
    import hashlib
    import re
    listed = {}
    for line in open(os.path.join(RECORDS, "MANIFEST.sha256")):
        h, p = line.rstrip("\n").split("  ", 1)
        listed[p[2:]] = h
    files = sorted(os.path.relpath(os.path.join(dp, f), RECORDS) for dp, _, fs in os.walk(RECORDS) for f in fs if f != "MANIFEST.sha256")
    assert sorted(listed) == files
    absolute = re.compile(r"(?<![\w.<>~:/-])/(?:[\w.-]+/)+")
    for f in files:
        b = open(os.path.join(RECORDS, f), "rb").read()
        assert hashlib.sha256(b).hexdigest() == listed[f], f
        if not f.endswith(".npz"):
            assert not absolute.search(b.decode("ascii")), (f, absolute.search(b.decode("ascii")))
    assert sorted(os.listdir(os.path.join(RECORDS, "raw"))) == ["machine_a", "machine_b", "task_embeddings"]
    rows = list(csv.DictReader(open(os.path.join(RECORDS, "policies.csv"))))
    assert len(rows) == 63 and {r["host"] for r in rows} == {"machine_a", "machine_b"}
    for r in rows:
        assert r["run_dir"].startswith("raw/" + r["host"] + "/") and os.path.isdir(os.path.join(RECORDS, r["run_dir"]))
        assert "/" not in r["encoder_weights"] and r["task_embedding_source"].split("/")[0] in ("bert", "file:raw")
        cfg = json.load(open(os.path.join(RECORDS, r["run_dir"], "config.json")))
        assert all(cfg[k].startswith("<") for k in ("folder", "bddl_folder", "init_states_folder"))
        assert cfg["task_embedding_format"] == "bert" or cfg["task_embedding_format"].startswith("file:<TASK_EMBEDDINGS_DIR>/")
        nk = cfg["policy"]["image_encoder"]["network_kwargs"]
        if "encoder_ckpt" in nk:                                 # the research runs
            assert nk["encoder_ckpt"].startswith(("timm:", "<"))
        else:                                                    # the C-RADIOv3-B policies, re-trained with the release code
            assert r["encoder"] == "C-RADIOv3-B" and nk["encoder"] == "cradio_v3_b" and nk["weights"] == ""
            assert nk["encoder_train_mode"] == "eval"
        assert cfg["experiment_dir"] == "./experiments/" + r["run_dir"].split("/", 2)[2]
        for e in ("eval50.json", "eval50_repA.json", "eval50_repB.json"):
            ev = json.load(open(os.path.join(RECORDS, r["run_dir"], e)))
            assert ev["rundir"] == r["run_dir"] and ev["ckpt"].startswith(r["run_dir"] + "/")


def test_policy_list_matches_the_records():
    import csv
    recs = {r["policy"]: r for r in csv.DictReader(open(os.path.join(RECORDS, "policies.csv")))}
    lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(HERE, "..", "paper_policies.tsv")) if not l.startswith("#")]
    head, rows = lines[0], lines[1:]
    assert "research_run" not in head and len(rows) == 63
    for r in rows:
        d = dict(zip(head, r))
        rec = recs[d["policy"]]
        assert (d["seed"], d["epochs"], d["paper_value"].replace("-", "")) == (rec["training_seed"], rec["epochs"], rec["value_in_paper"])


class AttrDict(dict):                       # stands in for the OmegaConf node of a run config
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_research_run_configs_are_adapted():
    sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
    import evaluate_policy as ev
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="timm:vit_base_patch16_dinov3.lvd1689m", flip_ud=True, num_queries=1))
    assert nk["encoder"] == "dinov3_b16" and nk["weights"] == "" and "encoder_ckpt" not in nk and nk["flip_ud"] is True
    assert nk["encoder_train_mode"] == "follow" and nk["alps_port_numerics"] is True   # the settings of the paper's runs
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="timm:vit_base_patch14_dinov2.lvd142m"))
    assert nk["encoder"] == "dinov2_b14_timm"
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="radio:/somewhere/c-radio_v3-b_half.pth.tar"))
    assert nk["encoder"] == "cradio_v3_b" and nk["weights"].endswith("c-radio_v3-b_half.pth.tar")
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="radio:<c-radio_v3-b_half.pth.tar>"))     # placeholder form
    assert nk["encoder"] == "cradio_v3_b" and nk["weights"] == ""                          # -> RADIO_CKPT
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="<alps_stage1_vitb16.safetensors>"))
    assert nk["encoder"] == "alps_stage1" and nk["weights"] == ""                          # -> ALPS_STAGE1_WEIGHTS
    with pytest.raises(SystemExit):
        ev.adapt_network_kwargs(AttrDict(encoder_ckpt="<siglip2_b16_posttrained.pt>"))         # needs the refined weights
    with pytest.raises(SystemExit):
        ev.adapt_network_kwargs(AttrDict(encoder_ckpt="/somewhere/encoder_port.pt"))
    nk = ev.adapt_network_kwargs(AttrDict(encoder_ckpt="/somewhere/encoder_port.pt"), encoder="alps", weights="alps_vitb16.safetensors")
    assert nk["encoder"] == "alps" and os.path.isabs(nk["weights"])
    nk = ev.adapt_network_kwargs(AttrDict(encoder="alps", weights="/old/alps.safetensors", encoder_train_mode="eval"),
                                 weights="/new/alps.safetensors")
    assert nk["weights"] == "/new/alps.safetensors" and nk["encoder_train_mode"] == "eval"


def test_task_embeddings_of_the_records_resolve(monkeypatch):
    import evaluate_policy as ev
    cfg = AttrDict(task_embedding_format="file:<TASK_EMBEDDINGS_DIR>/alps_task_embs.npz")
    monkeypatch.setenv("TASK_EMBEDDINGS_DIR", os.path.join(RECORDS, "raw", "task_embeddings"))
    ev.resolve_task_embeddings(cfg)
    assert cfg["task_embedding_format"] == "file:" + os.path.join(RECORDS, "raw", "task_embeddings", "alps_task_embs.npz")
    monkeypatch.delenv("TASK_EMBEDDINGS_DIR")
    with pytest.raises(SystemExit):
        ev.resolve_task_embeddings(AttrDict(task_embedding_format="file:<TASK_EMBEDDINGS_DIR>/alps_task_embs.npz"))
    ev.resolve_task_embeddings(AttrDict(task_embedding_format="bert"))
