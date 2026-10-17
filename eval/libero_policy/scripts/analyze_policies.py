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
"""Success tables of the LIBERO policies from the per-rollout evaluation records.

An evaluation file (eval50.json, eval50_repA.json, eval50_repB.json in a run directory, written by evaluate_policy.py) holds
per_rollout[task][i] = success of the rollout from fixed initial state i. The success of a policy is the mean over its three
evaluations x 10 tasks x 50 initial states, in %.

  python analyze_policies.py paper [--check] [--json out.json] [--config <records folder>]
      the numbers of the paper from the records of its 63 policies (../paper_records: policies.csv + raw/): Tab. 2, Tab. S8
      (interval, worst task, tasks >= 80), Tab. S9 (every task), the refined SigLIP 2 policies of Sec. S3
      and the retrainings of ALPS (Sec. S2); --check compares every printed value with ../paper_numbers.json
  python analyze_policies.py runs <dir> [<dir> ...] [--json out.json]
      the same per-policy numbers for any run directories found below the given folders (e.g. new trainings)

Statistics (as in the paper):
  * Tab. S8 interval: 4,000 bootstrap resamples; the 10 tasks of a suite are resampled, then the 50 initial states within
    every task; the three evaluations of one initial state are averaged first. numpy default_rng(crc32(key)) per column, key
    "policy_<suite>" for SigLIP 2, DINOv2, DINOv3, C-RADIOv3, ALPS (Stage 1) and ALPS with BERT, "policy_f7_<suite>" for the
    rows added later (CLIP with BERT and the four own-text-tower rows). <suite> = 10 | goal | spatial | object.
  * Retrainings (Sec. S2): per training seed s, D_s = mean over the four suites of (own text tower - BERT); interval of the
    mean of D over s = 0, 1, 2 from 10,000 resamples of the seeds, the tasks of every suite and the rollouts of every task
    (python random.Random(20261006)).
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_RECORDS = os.path.join(HERE, "..", "paper_records")
PAPER_NUMBERS = os.path.join(HERE, "..", "paper_numbers.json")
SUITES = [("10", "LIBERO-Long", "LIBERO_10"), ("goal", "LIBERO-Goal", "LIBERO_GOAL"),
          ("spatial", "LIBERO-Spatial", "LIBERO_SPATIAL"), ("object", "LIBERO-Object", "LIBERO_OBJECT")]
SUITE_NAME = {k: n for k, n, _ in SUITES}
EVALS = (("eval50", "eval50.json"), ("repA", "eval50_repA.json"), ("repB", "eval50_repB.json"))
B_TABLE = 4000
B_RETRAIN = 10000
# rows of Tab. 2 / S8 in the order of the paper; "orig" / "f7" = the generator key of the row's interval (see the docstring)
BERT_ROWS = [("CLIP-B/16", "f7"), ("SigLIP 2-B/16", "orig"), ("DINOv2-B/14", "orig"), ("DINOv3-B/16", "orig"),
             ("C-RADIOv3-B", "orig"), ("ALPS (Stage 1)", "orig"), ("ALPS", "orig")]
OWN_ROWS = [("CLIP-B/16", "f7"), ("SigLIP 2-B/16", "f7"), ("C-RADIOv3-B", "f7"), ("ALPS", "f7")]


# ------------------------------------------------------------------------------------------------------------------ data
def load_evaluations(run_dir, require_all=True):
    """-> (array [evaluations, tasks, rollouts] of 0/1, task names, evaluation tags found)."""
    xs, tags, tasks = [], [], None
    for tag, fname in EVALS:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            if require_all:
                raise FileNotFoundError(path)
            continue
        ev = json.load(open(path))
        n = len(ev["per_task"])
        r = np.array([ev["per_rollout"][str(t)] for t in range(n)], float)
        if r.shape != (n, int(ev["n_eval"])):
            raise ValueError(f"{path}: per_rollout has shape {r.shape}, expected ({n}, {ev['n_eval']})")
        if not np.allclose(r.mean(1), np.asarray(ev["per_task"], float), atol=1e-6):
            raise ValueError(f"{path}: per_rollout does not reproduce per_task")
        if tasks is not None and ev["tasks"] != tasks:
            raise ValueError(f"{path}: task order differs between the evaluations of one policy")
        tasks = ev["tasks"]
        xs.append(r)
        tags.append(tag)
    if not xs:
        return None, None, []
    return np.stack(xs), tasks, tags


def success(x):
    return 100.0 * float(x.mean())


def per_task(x):
    return 100.0 * x.mean(axis=(0, 2))


# ------------------------------------------------------------------------------------------------------------- statistics
def bootstrap_column(cells, key, b=B_TABLE):
    """cells {row: [evaluations, 10, 50]} of one suite -> {row: [b] resampled success in %}; resamples shared by the rows."""
    g = np.random.default_rng(zlib.crc32(key.encode()))
    y = {r: x.mean(0) for r, x in cells.items()}                       # [10 tasks, 50 initial states]
    out = {r: np.empty(b) for r in cells}
    for k in range(b):
        t = g.integers(0, 10, 10)
        i = g.integers(0, 50, (10, 50))
        for r in sorted(cells):
            out[r][k] = y[r][t[:, None], i].mean() * 100
    return out


def retraining_bootstrap(cells, b=B_RETRAIN, seed=20261006):
    """cells {(suite, seed, 'bert' | 'text'): [3, 10, 50]} for seeds 0, 1, 2 -> (mean of D over the seeds, lo, hi)."""
    seeds = [0, 1, 2]
    suites = [s for s, _, _ in SUITES]
    # per task the rollouts of the three evaluations concatenated (eval50, repA, repB), as lists of floats
    lists = {k: [list(np.concatenate([x[e, t] for e in range(x.shape[0])])) for t in range(x.shape[1])] for k, x in cells.items()}
    succ = {k: 100 * sum(sum(t) / len(t) for t in c) / len(c) for k, c in lists.items()}
    d_seed = {s: sum(succ[(su, s, "text")] - succ[(su, s, "bert")] for su in suites) / 4 for s in seeds}
    rng = random.Random(seed)
    choice, randrange = rng.choice, rng.randrange                # local names only (the same draws in the same order)
    boots = []
    for _ in range(b):
        seeds_b = [choice(seeds) for _ in seeds]
        tasks_b = {su: [randrange(10) for _ in range(10)] for su in suites}
        dsum = 0.0
        for s in seeds_b:
            for su in suites:
                d = 0.0
                for ver, sign in (("text", 1), ("bert", -1)):
                    c = lists[(su, s, ver)]
                    m = 0.0
                    for t in tasks_b[su]:
                        rr = c[t]
                        n = len(rr)
                        m += sum([rr[randrange(n)] for _ in range(n)]) / n
                    d += sign * 100 * m / 10
                dsum += d / 4
        boots.append(dsum / len(seeds_b))
    boots.sort()
    lo, hi = boots[int(0.025 * b)], boots[int(0.975 * b) - 1]
    return sum(d_seed.values()) / 3, lo, hi, d_seed, succ


# ------------------------------------------------------------------------------------------------------------ paper mode
def paper(args):
    rows = list(csv.DictReader(open(os.path.join(args.config, "policies.csv"))))
    pol = []
    for r in rows:
        if os.path.isabs(r["run_dir"]):                   # an unscrubbed configuration: raw/<host>/<benchmark>/Multitask/<run>/<id>
            d = os.path.join(args.config, "raw", r["host"], r["libero_benchmark"], "Multitask", *r["run_dir"].rstrip("/").split("/")[-2:])
        else:                                            # paper_records: run_dir relative to the folder
            d = os.path.join(args.config, r["run_dir"])
        x, tasks, _ = load_evaluations(d)
        suite = [s for s, n, _ in SUITES if n == r["suite"]][0]
        pol.append(dict(row=r, x=x, tasks=tasks, suite=suite, enc=r["encoder"], instr=r["instruction_embedding"],
                        seed=int(r["training_seed"]), used=r["used_in"]))
    print(f"{len(pol)} policies, {sum(p['x'].shape[0] for p in pol)} evaluations, "
          f"{int(sum(p['x'].size for p in pol))} rollouts")
    out = {}

    def table2_cell(enc, instr, suite):
        tag = "Table 2 (BERT)" if instr == "BERT" else "Table 2 (own text tower)"
        hit = [p for p in pol if p["enc"] == enc and p["instr"] == instr and p["suite"] == suite and tag in p["used"]]
        assert len(hit) == 1, (enc, instr, suite, len(hit))
        return hit[0]

    # Tab. 2 and S8
    t2, s8, cells = {}, {}, {}
    for instr, rowset in (("BERT", BERT_ROWS), ("own text tower", OWN_ROWS)):
        t2[instr], s8[instr] = {}, {}
        for enc, _ in rowset:
            t2[instr][enc], s8[instr][enc] = {}, {}
            for s, sname, _ in SUITES:
                p = table2_cell(enc, instr, s)
                cells[(enc, instr, s)] = p["x"]
                v = success(p["x"])
                ref = p["row"]["value_in_paper"]
                if ref and abs(round(v, 1) - float(ref)) > 1e-9:
                    raise AssertionError(f"{p['row']['policy']}: {v:.4f} vs value_in_paper {ref}")
                t2[instr][enc][sname] = v
            t2[instr][enc]["mean"] = float(np.mean([t2[instr][enc][n] for _, n, _ in SUITES]))
    for s, sname, _ in SUITES:
        for grp in ("orig", "f7"):
            members = {(enc, instr): cells[(enc, instr, s)] for instr, rowset in (("BERT", BERT_ROWS), ("own text tower", OWN_ROWS))
                       for enc, g in rowset if g == grp}
            key = ("policy_" if grp == "orig" else "policy_f7_") + s
            bs = bootstrap_column({"%s|%s" % k: v for k, v in members.items()}, key)
            for (enc, instr), x in members.items():
                lo, hi = np.percentile(bs["%s|%s" % (enc, instr)], [2.5, 97.5])
                task = per_task(x)
                s8[instr][enc][sname] = {"success": t2[instr][enc][sname], "ci": (float(lo), float(hi)),
                                         "worst": float(task.min()), "ge80": int((task >= 80 - 1e-9).sum()), "per_task": task.tolist()}
    for instr in s8:
        for enc in s8[instr]:
            s8[instr][enc]["mean"] = t2[instr][enc]["mean"]
    out["table2"], out["tableS8"] = t2, s8
    # Tab. S9: every task, BERT rows
    out["tableS9"] = {sname: {enc: s8["BERT"][enc][sname]["per_task"] for enc, _ in BERT_ROWS} for _, sname, _ in SUITES}
    # Sec. S3: SigLIP 2 + Stage 2 (Long, Goal, Spatial), compared there with ALPS (the BERT cells of Tab. 2)
    s3 = {}
    for p in pol:
        if "Sec. S3" in p["used"]:
            s3[SUITE_NAME[p["suite"]]] = success(p["x"])
    out["secS3_policy"] = {"SigLIP 2-B/16 + Stage 2": s3}
    # retrainings of ALPS
    rc = {}
    for p in pol:
        if "ALPS retrainings" in p["used"]:
            rc[(p["suite"], p["seed"], "bert" if p["instr"] == "BERT" else "text")] = p["x"]
    assert len(rc) == 24, len(rc)
    mean_d, lo, hi, d_seed, succ = retraining_bootstrap(rc)
    bert_mean = np.mean([succ[(s, k, "bert")] for s, _, _ in SUITES for k in (0, 1, 2)])
    text_mean = np.mean([succ[(s, k, "text")] for s, _, _ in SUITES for k in (0, 1, 2)])
    out["retrainings"] = {"bert_mean": bert_mean, "text_mean": text_mean, "gain": mean_d, "ci95": (lo, hi),
                          "per_seed_gain": d_seed, "success": {"%s|seed %d|%s" % k: v for k, v in succ.items()}}
    report_paper(out)
    if args.json:
        json.dump(_jsonable(out), open(args.json, "w"), indent=1)
    if args.check:
        return check(out, json.load(open(args.check)), args.check)
    return 0


def report_paper(o):
    sn = [n for _, n, _ in SUITES]
    print("\nTab. 2: success in %, mean of three evaluations x 10 tasks x 50 initial states")
    for instr in ("BERT", "own text tower"):
        print(f"  instruction embedded by {instr}")
        for enc, row in o["table2"][instr].items():
            print("    %-16s" % enc + "".join("%9.1f" % row[n] for n in sn) + "   mean %.1f" % row["mean"])
    print("\nTab. S8: success [95% interval] worst task, tasks >= 80")
    for instr in ("BERT", "own text tower"):
        for enc, row in o["tableS8"][instr].items():
            cells = ["%.1f [%.0f, %.0f] %.0f %d" % (row[n]["success"], *row[n]["ci"], row[n]["worst"], row[n]["ge80"]) for n in sn]
            print("    %-16s %-15s " % (enc, instr) + " | ".join(cells))
    print("\nSec. S3: SigLIP 2 + Stage 2 with BERT: " + ", ".join("%s %.1f" % kv for kv in o["secS3_policy"]["SigLIP 2-B/16 + Stage 2"].items()))
    r = o["retrainings"]
    print("\nRetrainings of ALPS (3 per suite and embedding): BERT %.1f%%, own text tower %.1f%%, gain %+.1f points, 95%% interval "
          "[%+.1f, %+.1f]; per training seed %s" % (r["bert_mean"], r["text_mean"], r["gain"], *r["ci95"],
                                                    {k: round(float(v), 1) for k, v in r["per_seed_gain"].items()}))


def check(o, ref, ref_path="paper_numbers.json"):
    """Compare the computed numbers, formatted as printed in the paper, with paper_numbers.json."""
    sn = [n for _, n, _ in SUITES]
    bad, n = [], 0

    def eq(what, got, want):
        nonlocal n
        n += 1
        if got != want:
            bad.append(f"{what}: computed {got}, paper {want}")

    sgn = lambda v: "%+.1f" % v
    for instr in ("BERT", "own text tower"):
        for enc, row in ref["table2"][instr].items():
            for k in sn + ["mean"]:
                eq(f"Tab. 2 {instr} {enc} {k}", "%.1f" % o["table2"][instr][enc][k], row[k])
        for enc, row in ref["tableS8"][instr].items():
            c = o["tableS8"][instr][enc]
            for k in sn:
                eq(f"Tab. S8 {instr} {enc} {k} success", "%.1f" % c[k]["success"], row[k]["success"])
                eq(f"Tab. S8 {instr} {enc} {k} interval", "[%.0f, %.0f]" % c[k]["ci"], row[k]["ci"])
                eq(f"Tab. S8 {instr} {enc} {k} worst task", "%.0f" % c[k]["worst"], row[k]["worst"])
                eq(f"Tab. S8 {instr} {enc} {k} tasks >= 80", "%d" % c[k]["ge80"], row[k]["ge80"])
            eq(f"Tab. S8 {instr} {enc} mean", "%.1f" % c["mean"], row["mean"])
    for sname, cols in ref["tableS9"].items():
        for enc, vals in cols.items():
            if enc == "mean":
                for e2, v in vals.items():
                    eq(f"Tab. S9 {sname} mean {e2}", "%.1f" % o["table2"]["BERT"][e2][sname], v)
                continue
            for t, v in enumerate(vals):
                eq(f"Tab. S9 {sname} {enc} task {t}", "%.0f" % o["tableS9"][sname][enc][t], v)
    for k, v in ref["secS3_policy"]["SigLIP 2-B/16 + Stage 2"].items():
        eq(f"Sec. S3 SigLIP 2 + Stage 2 {k}", "%.1f" % o["secS3_policy"]["SigLIP 2-B/16 + Stage 2"][k], v)
    for k, v in ref["secS3_policy"]["ALPS"].items():
        eq(f"Sec. S3 ALPS {k}", "%.1f" % o["table2"]["BERT"]["ALPS"][k], v)
    r, rr = o["retrainings"], ref["retrainings"]
    eq("retrainings BERT mean", "%.1f" % r["bert_mean"], rr["bert_mean"])
    eq("retrainings own text tower mean", "%.1f" % r["text_mean"], rr["text_mean"])
    eq("retrainings gain", "%+.1f" % r["gain"], rr["gain"])
    eq("retrainings interval", "[%+.1f, %+.1f]" % r["ci95"], rr["ci95"])
    print(f"\ncheck against {os.path.normpath(ref_path)}: {n - len(bad)} of {n} printed values reproduced")
    for b in bad:
        print("  MISMATCH " + b)
    return 1 if bad else 0


# -------------------------------------------------------------------------------------------------------------- runs mode
def _describe(cfg):
    nk = cfg["policy"]["image_encoder"]["network_kwargs"]
    enc = nk.get("encoder") or os.path.basename(str(nk.get("encoder_ckpt", "?")))
    w = nk.get("weights") or ""
    if w and enc not in ("alps", "alps_stage1", "cradio_v3_b"):
        enc += "+" + os.path.basename(w)
    tef = str(cfg.get("task_embedding_format", "bert"))
    instr = "BERT" if tef == "bert" else os.path.basename(tef.split(":", 1)[-1])
    return enc, instr


def runs(args):
    found = []
    for root in args.dirs:
        for cfg_path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            d = os.path.dirname(cfg_path)
            x, tasks, tags = load_evaluations(d, require_all=False)
            if x is None:
                continue
            cfg = json.load(open(cfg_path))
            enc, instr = _describe(cfg)
            suite = str(cfg["benchmark_name"]).lower().replace("libero_", "")
            found.append(dict(dir=d, x=x, tasks=tasks, tags=tags, enc=enc, instr=instr, suite=suite, seed=cfg["seed"],
                              epochs=cfg["train"]["n_epochs"]))
    if not found:
        print("no run directory with evaluation files found")
        return 1
    res = []
    for p in sorted(found, key=lambda p: ([s for s, _, _ in SUITES].index(p["suite"]) if p["suite"] in SUITE_NAME else 9,
                                          p["enc"], p["instr"], p["seed"])):
        task = per_task(p["x"])
        if p["x"].shape[1:] == (10, 50):
            lo, hi = np.percentile(bootstrap_column({"p": p["x"]}, "policy_" + p["suite"])["p"], [2.5, 97.5])
        else:
            lo = hi = float("nan")
        per_eval = {t: success(p["x"][k:k + 1]) for k, t in enumerate(p["tags"])}
        rec = dict(dir=p["dir"], suite=SUITE_NAME.get(p["suite"], p["suite"]), encoder=p["enc"], instruction=p["instr"],
                   seed=p["seed"], epochs=p["epochs"], evaluations=per_eval, success=success(p["x"]), ci95=(lo, hi),
                   worst_task=float(task.min()), tasks_ge80=int((task >= 80 - 1e-9).sum()), per_task=task.tolist())
        res.append(rec)
        note = "" if len(p["tags"]) == 3 else "  (%d of 3 evaluations)" % len(p["tags"])
        print("%-15s %-22s %-24s seed %s: %.1f [%.0f, %.0f] worst %.0f, >= 80: %d | %s%s" % (
            rec["suite"], rec["encoder"][:22], rec["instruction"][:24], rec["seed"], rec["success"], lo, hi, rec["worst_task"],
            rec["tasks_ge80"], " ".join("%s %.1f" % kv for kv in per_eval.items()), note))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("paper", help="the numbers of the paper from the records of its policies")
    a.add_argument("--config", default=PAPER_RECORDS, help="folder with policies.csv and raw/ (default: ../paper_records)")
    a.add_argument("--check", nargs="?", const=PAPER_NUMBERS, default=None,
                   help="compare every printed value with paper_numbers.json (default file: ../paper_numbers.json)")
    a.add_argument("--json", help="write every computed number to this file")
    b = sub.add_parser("runs", help="per-policy numbers of the run directories below the given folders")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--json")
    args = ap.parse_args()
    sys.exit(paper(args) if args.mode == "paper" else runs(args))


if __name__ == "__main__":
    main()
