#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""mesh_symmetry.py -- orientation-symmetry class per LIBERO object category, from the asset geometry.

For every category used by libero_spatial / libero_object: parse the category xml (ElementTree, walking nested
bodies), load every visual mesh geom (.msh legacy MuJoCo binary / .obj / .stl) with its mesh scale and geom/body
pos+quat, giving the vertex cloud in the object's ROOT BODY frame. Rotate it by the body quaternion the object
actually has in the env (task_facts.json 'category_quats', read from a restored demo state) so that world +z is
the up axis, then measure how well the cloud maps onto itself under yaw rotations about the best axis:
  mean nearest-neighbour distance (cKDTree against the full cloud) of the rotated cloud, for 180 deg, 90 deg and
  three generic angles (37/73/137 deg); axis centre fitted by Nelder-Mead (bbox centre start); everything
  normalised by the cloud's xy radius, with the mesh's own vertex spacing (2nd-NN distance) as the noise floor.
Rule (residuals as a fraction of the xy radius; scanned meshes are never exact, so the rule is RELATIVE):
  cyl  : generic-angle residual < 6% of radius and generic ~ 180-deg residual (ratio < 1.3: no yaw is preferred)
  z4   : else 90-deg ~ 180-deg residual (< 1.3) and 180-deg residual < 6%   (square section; treated as z2 downstream)
  z2   : else 180-deg residual < 6% of radius                                (rectangular / oval section)
  none : otherwise
'uncertain' when the deciding residual is > 2% of the radius (or, for 'none', the 180-deg residual is < 8%).
A best-fit tilted revolution axis is reported as a diagnostic (how much of the generic residual is axis tilt).
Textures/labels are ignored (geometry only).
Writes mesh_symmetry.json (consumed by label_states.py --manifest). Usage: LIBERO_ASSETS=... LIBERO_LABELS=... python mesh_symmetry.py
"""
import os, sys, json, struct
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from scipy.optimize import minimize

ROOT = os.environ.get("LIBERO_ASSETS", "LIBERO/libero/libero/assets")   # the assets folder of the LIBERO repository
OUT = os.environ.get("LIBERO_LABELS", ".")                              # folder with task_facts.json; mesh_symmetry.json is written here
CAT_XML = {
    "akita_black_bowl": "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", "plate": "stable_scanned_objects/plate/plate.xml",
    "glazed_rim_porcelain_ramekin": "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml",
    "basket": "stable_scanned_objects/basket/basket.xml", "cookies": "stable_hope_objects/cookies/cookies.xml",
    "alphabet_soup": "stable_hope_objects/alphabet_soup/alphabet_soup.xml", "bbq_sauce": "stable_hope_objects/bbq_sauce/bbq_sauce.xml",
    "butter": "stable_hope_objects/butter/butter.xml", "chocolate_pudding": "stable_hope_objects/chocolate_pudding/chocolate_pudding.xml",
    "cream_cheese": "stable_hope_objects/cream_cheese/cream_cheese.xml", "ketchup": "stable_hope_objects/ketchup/ketchup.xml",
    "milk": "stable_hope_objects/milk/milk.xml", "orange_juice": "stable_hope_objects/orange_juice/orange_juice.xml",
    "salad_dressing": "stable_hope_objects/salad_dressing/salad_dressing.xml", "tomato_sauce": "stable_hope_objects/tomato_sauce/tomato_sauce.xml",
    "wooden_cabinet": "articulated_objects/wooden_cabinet.xml", "flat_stove": "articulated_objects/flat_stove.xml",
}
PRIOR = {  # what the class name / photo suggests, for the report only
    "akita_black_bowl": "bowl", "plate": "plate", "glazed_rim_porcelain_ramekin": "ramekin", "basket": "rectangular basket with 2 handles",
    "cookies": "flat rectangular box", "alphabet_soup": "can", "bbq_sauce": "bottle", "butter": "rectangular box", "chocolate_pudding": "rectangular box",
    "cream_cheese": "rectangular box", "ketchup": "bottle", "milk": "gable-top carton", "orange_juice": "gable-top carton", "salad_dressing": "bottle",
    "tomato_sauce": "can", "wooden_cabinet": "3-drawer cabinet (fixture)", "flat_stove": "stove with side knob (fixture)",
}


def read_msh(path):
    with open(path, "rb") as f:
        nv, nn, nt, nf = struct.unpack("<4i", f.read(16))
        v = np.frombuffer(f.read(12 * nv), dtype="<f4").reshape(nv, 3)
    return v.astype(np.float64)


def read_obj(path):
    vs = []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    return np.array(vs)


def read_stl(path):
    with open(path, "rb") as f:
        head = f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tri = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    return np.unique(tri.reshape(-1, 3), axis=0).astype(np.float64)


def quat_to_mat(q):  # wxyz
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def arr(s, default):
    return np.array([float(x) for x in s.split()]) if s else np.array(default, dtype=float)


def load_cloud(xml_path):
    root = ET.parse(xml_path).getroot()
    xdir = os.path.dirname(xml_path)
    meshdir = ""
    for c in root.iter("compiler"):
        meshdir = c.get("meshdir", "")
    meshes = {}
    for m in root.iter("mesh"):
        meshes[m.get("name")] = (m.get("file"), arr(m.get("scale"), [1, 1, 1]))
    pts, ngeom = [], 0

    def walk(body, R, p):
        nonlocal ngeom
        for g in body.findall("geom"):
            if g.get("type") != "mesh":
                continue
            fname, scale = meshes[g.get("mesh")]
            cands = [os.path.join(xdir, meshdir, fname), os.path.join(xdir, fname)]
            fpath = next((c for c in cands if os.path.exists(c)), None)
            if fpath is None:
                raise FileNotFoundError(fname)
            v = read_msh(fpath) if fpath.endswith(".msh") else read_stl(fpath) if fpath.lower().endswith(".stl") else read_obj(fpath)
            v = v * scale
            Rg = quat_to_mat(arr(g.get("quat"), [1, 0, 0, 0]))
            v = v @ Rg.T + arr(g.get("pos"), [0, 0, 0])
            pts.append(v @ R.T + p)
            ngeom += 1
        for b in body.findall("body"):
            Rb = quat_to_mat(arr(b.get("quat"), [1, 0, 0, 0]))
            pb = arr(b.get("pos"), [0, 0, 0])
            walk(b, R @ Rb, R @ pb + p)

    wb = root.find("worldbody")
    # robosuite's MujocoXMLObject takes the (first) body named 'object' as the object root; its own pos/quat are dropped.
    obj = next((b for b in wb.iter("body") if b.get("name") == "object"), None)
    if obj is None:
        obj = wb
    walk(obj, np.eye(3), np.zeros(3))
    return np.concatenate(pts, 0), ngeom


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tilt_mat(tx, ty):
    """rotation taking +z to the axis tilted by small angles tx (about y) and ty (about x), radians."""
    cx, sx, cy, sy = np.cos(ty), np.sin(ty), np.cos(tx), np.sin(tx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rx @ Ry


def analyse(v, rng):
    v = np.unique(np.round(v, 7), axis=0)                        # .msh stores per-face-corner copies of vertices
    tree = cKDTree(v)
    n = min(6000, len(v))
    qi = rng.choice(len(v), n, replace=False)
    q = v[qi]
    spacing = float(tree.query(q, k=2)[0][:, 1].mean())          # mesh vertex spacing = noise floor
    c0 = 0.5 * (v[:, :2].max(0) + v[:, :2].min(0))                # bbox centre start

    def score(center, ang, stat="mean", tilt=(0.0, 0.0)):
        c = np.array([center[0], center[1], 0.0])
        Rt = tilt_mat(*tilt)
        d = tree.query((q - c) @ (Rt @ rotz(ang) @ Rt.T).T + c)[0]
        return float(d.mean() if stat == "mean" else np.percentile(d, 90))

    opts = dict(xatol=1e-4, fatol=1e-6, maxiter=300)
    c_cyl = minimize(lambda c: score(c, np.deg2rad(137)), c0, method="Nelder-Mead", options=opts).x
    c_z2 = minimize(lambda c: score(c, np.pi), c0, method="Nelder-Mead", options=opts).x
    rxy = float(np.sqrt(((v[:, :2] - c_z2) ** 2).sum(1)).max())
    s = dict(gen=max(score(c_cyl, np.deg2rad(a)) for a in (37, 73, 137)), r90=score(c_z2, np.pi / 2), r180=score(c_z2, np.pi),
             gen_p90=max(score(c_cyl, np.deg2rad(a), "p90") for a in (37, 73, 137)), r90_p90=score(c_z2, np.pi / 2, "p90"), r180_p90=score(c_z2, np.pi, "p90"))
    # diagnostic: best-fit (possibly tilted) revolution axis -- how much of the generic-angle residual is axis tilt?
    x0 = np.array([c_cyl[0], c_cyl[1], 0.0, 0.0])
    fit = minimize(lambda x: score(x[:2], np.deg2rad(137), tilt=(x[2], x[3])), x0, method="Nelder-Mead", options=opts)
    gen_tilt = max(score(fit.x[:2], np.deg2rad(a), tilt=(fit.x[2], fit.x[3])) for a in (37, 73, 137))
    rel = {k: s[k] / rxy for k in ("gen", "r90", "r180")}
    ratio_gen_180 = s["gen"] / max(s["r180"], 1e-9)
    ratio_90_180 = s["r90"] / max(s["r180"], 1e-9)
    # decision (relative): a yaw symmetry exists when the corresponding residual is a small fraction of the xy radius;
    # 'cyl' when no yaw angle is preferred over 180 deg (generic ~ 180), 'z2' when 180 deg is clearly preferred.
    if ratio_gen_180 < 1.3 and rel["gen"] < 0.06:
        cls, key = "cyl", "gen"
    elif ratio_90_180 < 1.3 and rel["r180"] < 0.06:
        cls, key = "z4", "r180"
    elif rel["r180"] < 0.06:
        cls, key = "z2", "r180"
    else:
        cls, key = "none", "r180"
    uncertain = (rel[key] > 0.02) if cls != "none" else (rel["r180"] < 0.08)
    thr = max(3 * spacing, 0.02 * rxy)
    return dict(symmetry_class=cls, uncertain=bool(uncertain), spacing_m=spacing, radius_xy_m=rxy, thr_m=thr,
                scores_m={k: round(x, 5) for k, x in s.items()}, scores_rel_radius={k: round(x, 4) for k, x in rel.items()},
                ratio_generic_over_180=round(ratio_gen_180, 2), ratio_90_over_180=round(ratio_90_180, 2),
                best_axis_fit=dict(gen_rel_radius=round(gen_tilt / rxy, 4), tilt_deg=[round(float(np.rad2deg(fit.x[2])), 2), round(float(np.rad2deg(fit.x[3])), 2)]),
                axis_centre_cyl=[round(float(x), 4) for x in c_cyl], axis_centre_z2=[round(float(x), 4) for x in c_z2],
                extent_world_xyz=[round(float(x), 4) for x in (v.max(0) - v.min(0))], n_vertices=int(len(v)))


def main():
    facts = json.load(open(os.path.join(OUT, "task_facts.json")))
    cq = {}
    for t in facts["tasks"]:
        for c, d in t["category_quats"].items():
            cq.setdefault(c, d)
    rng = np.random.RandomState(0)
    out = {}
    for cat, rel in CAT_XML.items():
        xml = os.path.join(ROOT, rel)
        try:
            v, ng = load_cloud(xml)
            if cat not in cq:
                raise KeyError("no body quat for category in task_facts.json")
            Rw = quat_to_mat(np.array(cq[cat]["quat_wxyz"]))
            vw = v @ Rw.T                                    # root-body frame -> world (up = +z)
            res = analyse(vw, rng)
            res.update(xml=rel, n_mesh_geoms=ng, body_quat_wxyz=cq[cat]["quat_wxyz"], up_in_body=cq[cat]["up_in_body"], prior=PRIOR.get(cat))
            if res["symmetry_class"] == "z4":
                res["note"] = "90-degree symmetric (square section); use z2 for a 2-class scheme"
            elif res["symmetry_class"] == "none" and cat in ("wooden_cabinet", "flat_stove"):
                res["note"] = "articulated fixture: drawers/knob on one side"
            else:
                res["note"] = "geometry only; printed labels break the symmetry visually" if cat in ("alphabet_soup", "tomato_sauce", "bbq_sauce", "ketchup", "salad_dressing", "milk", "orange_juice", "butter", "cream_cheese", "chocolate_pudding", "cookies") else ""
            rr = res["scores_rel_radius"]
            res["evidence"] = "mean NN residual of the yaw-rotated mesh as %% of xy radius (%.0f mm): generic %.1f%%, 90deg %.1f%%, 180deg %.1f%% (generic/180 = %.2f); best tilted-axis generic %.1f%% (tilt %s deg); vertex spacing %.1f mm" % (
                1e3 * res["radius_xy_m"], 100 * rr["gen"], 100 * rr["r90"], 100 * rr["r180"], res["ratio_generic_over_180"], 100 * res["best_axis_fit"]["gen_rel_radius"], res["best_axis_fit"]["tilt_deg"], 1e3 * res["spacing_m"])
            print("%-30s %-5s%s ext=%s  %s  prior=%s" % (cat, res["symmetry_class"], "?" if res["uncertain"] else " ", res["extent_world_xyz"], res["evidence"], PRIOR.get(cat)))
        except Exception as ex:
            res = dict(symmetry_class="unknown", uncertain=True, error=repr(ex), xml=rel, prior=PRIOR.get(cat), note="mesh analysis failed")
            print(cat, "ERR", repr(ex))
        out[cat] = res
    json.dump(out, open(os.path.join(OUT, "mesh_symmetry.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
