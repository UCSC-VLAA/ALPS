# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Merges the planning shards of one DINO-WM world model into one JSON line (Table 3 and Table S11).

  python merge_plan_shards.py <shard log> ... > <task>_<encoder>.json

Shard logs are the logs of dino_wm/scripts/plan.sh, named ..._<A>_<B>.log for the instances [A, B). Per shard only the LAST
attempt in the log counts (the text after the last line that starts with START) and it must end with "END ... rc=0".
One exception: an attempt that ended with rc 134 or 139 (a crash while the process exits) counts if its run had finished
("wandb: Find logs at" before END), its logs.json holds the final_eval record, and the per-instance values of the log
reproduce that record (within 1e-6); such shards are listed under accepted_after_exit_crash.
PushT / Wall / PointMaze: the final evaluation = the last "Success rate:" line of the attempt and the dict printed after it
({'success': array([...]), 'state_dist': array([...])}, one entry per instance of the shard, in order).
Granular / Rope: the last B - A "CD:" lines (final Chamfer distance per instance) and the final_eval record of logs.json.
Output: n, missing shards, instances, per-instance arrays in instance order (success / state_dist, or cd), their means
(success_rate, mean_state_dist, mean_cd, median_cd) and the final_eval means weighted by the shard sizes.
"""
import json
import os
import re
import sys

import numpy as np


def arr(txt, key):
    m = re.search(r"'%s': array\(\[(.*?)\]" % key, txt, re.S)
    if not m:
        return None
    vals = [v for v in re.split(r"[\s,]+", m.group(1).strip()) if v]
    if key == "success":
        return [v == "True" for v in vals]
    return [float(v) for v in vals]


def last_attempt(txt):
    return txt[txt.rfind("\nSTART") + 1:] if "\nSTART" in txt else txt


def accepted_after_exit_crash(seg, k, shard_log):
    if not (re.search(r"^END .*rc=(134|139)", seg, re.M) and "wandb: Find logs at" in seg[:seg.rfind("\nEND")]):
        return False
    dd = re.findall(r"Planning result saved dir: (\S+)", seg)
    lj = os.path.join(dd[-1], "logs.json") if dd else ""
    if not os.path.exists(lj):
        lj = shard_log[:-4] + ".logs.json"
    fin = [json.loads(line) for line in open(lj) if line.strip() and "final_eval/" in line] if os.path.exists(lj) else []
    cd = [float(x) for x in re.findall(r"^CD:\s+(\S+)", seg, re.M)]
    i = seg.rfind("Success rate:")
    if fin and "final_eval/mean_chamfer_distance" in fin[-1] and len(cd) >= k and "'state_dist'" not in seg[max(i, 0):]:
        return abs(float(np.mean(cd[-k:])) - fin[-1]["final_eval/mean_chamfer_distance"]) < 1e-6
    if fin and i >= 0 and "'state_dist'" in seg[i:]:
        s, d = arr(seg[i:], "success"), arr(seg[i:], "state_dist")
        return bool(s and d and len(s) == k and abs(float(np.mean(s)) - fin[-1]["final_eval/success_rate"]) < 1e-6
                    and abs(float(np.mean(d)) - fin[-1]["final_eval/mean_state_dist"]) < 1e-4)
    return False


def merge(logs):
    shards, accepted = [], []
    for f in logs:
        A, B = map(int, re.search(r"_(\d+)_(\d+)\.log$", f).groups())
        txt = open(f, errors="ignore").read() if os.path.exists(f) else ""
        seg = last_attempt(txt)
        ok = bool(re.search(r"^END .*rc=0", seg, re.M))
        if not ok and accepted_after_exit_crash(seg, B - A, f):
            ok = True; accepted.append(os.path.basename(f))
        shards.append((A, B, f, seg, ok))
    shards.sort()
    out = {"n": 0, "missing": [], "instances": []}
    succ, dist, cds, fe, fe_n = [], [], [], {}, 0
    for A, B, f, seg, ok in shards:
        k = B - A
        if not ok:
            out["missing"].append(os.path.basename(f)); continue
        out["instances"] += list(range(A, B)); out["n"] += k
        i = seg.rfind("Success rate:")
        if i >= 0 and "'state_dist'" in seg[i:]:
            tail = seg[i:]
            s, d = arr(tail, "success"), arr(tail, "state_dist")
            assert s is not None and len(s) == k and d is not None and len(d) == k, (f, k, s and len(s), d and len(d))
            succ += s; dist += d
        cd = [float(x) for x in re.findall(r"^CD:\s+(\S+)", seg, re.M)]
        if len(cd) >= k and "'state_dist'" not in seg[max(i, 0):]:
            cds += cd[-k:]
        dd = re.findall(r"Planning result saved dir: (\S+)", seg)
        lj = os.path.join(dd[-1], "logs.json") if dd else ""
        if not os.path.exists(lj):
            lj = f[:-4] + ".logs.json"
        if os.path.exists(lj):
            rec = [json.loads(line) for line in open(lj) if line.strip()]
            fin = [x for x in rec if any(key.startswith("final_eval/") for key in x)]
            if fin:
                for key, v in fin[-1].items():
                    if key.startswith("final_eval/"):
                        fe[key] = fe.get(key, 0.0) + float(v) * k
                fe_n += k
    if succ:
        out["success"] = [int(x) for x in succ]; out["state_dist"] = dist
        out["success_rate"] = float(np.mean(succ)); out["mean_state_dist"] = float(np.mean(dist))
    if cds:
        out["cd"] = cds; out["mean_cd"] = float(np.mean(cds)); out["median_cd"] = float(np.median(cds))
    if fe_n:
        out.update({key: v / fe_n for key, v in fe.items()})
    if accepted:
        out["accepted_after_exit_crash"] = accepted
    return out


if __name__ == "__main__":
    print(json.dumps(merge(sys.argv[1:])))
