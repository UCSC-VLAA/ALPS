#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""label_states.py -- ground-truth object / robot state labels + LIBERO predicates for every harness episode
of libero_spatial + libero_object (episodes.json ep00000..ep00999) at the harness stride (2).

Outputs (all under --out):
  labels/ep%05d.npz        one file per episode (keys documented in KEYS_DOC below)
  alignment_audit.json     20 random episodes: arm-qpos / eef error hdf5-vs-restored with the +1 shift (and
                           without it, for contrast), fraction of label_exact frames, frames vs episodes.json n_steps
  task_facts.json          per task: objects, categories, movable/goal flags, regions the env defines, predicate
                           names, articulated joints, and the body quaternion of one instance of every object
                           category (consumed by mesh_symmetry.py to measure symmetry about the WORLD up axis)
  labels_manifest.json     (--manifest) task_facts + orientation-symmetry class per object category from mesh_symmetry.json

Method: build each task env ONCE from its bddl
(OffScreenRenderEnv, EGL headless, no rendering), then for each kept raw frame restore the MuJoCo state stored in
the hdf5 demo (sim.set_state_from_flattened + sim.forward) and read body poses / joint qpos / LIBERO predicates
(libero.libero.envs.predicates.eval_predicate_fn on env.object_states_dict, exactly what _check_success uses).

Alignment: recorded obs[u] is the observation AFTER actions[u] -> obs[u] <-> states[u+1]. Harness frame i
(= obs[2i], stride 2) is therefore labelled from states[2i+1]. The last recorded frame of an ODD-length demo has no
stored state: it is labelled from states[N-1] (one 50 ms control step stale) and flagged label_exact=False.
Action windows: obs[2i] -> obs[2j] is driven by raw_actions[2i+1 .. 2j] (inclusive).

Usage (LIBERO environment, headless EGL):
  MUJOCO_GL=egl OMP_NUM_THREADS=1 python label_states.py --libero_root <LIBERO repo> --data_root <libero_datasets> \
      --episodes_json <episodes>/episodes.json --out <labels dir>                 # labels + audit + task_facts
  python label_states.py --out <labels dir> --manifest                            # labels_manifest.json (no env)
mesh_symmetry.json (the symmetry input of --manifest) is in this folder; mesh_symmetry.py recomputes it from the assets.
"""
import os, sys, time, json, argparse, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import h5py

LIBERO_ROOT = DATA_ROOT = OUT_ROOT = EPISODES_JSON = None      # set from the command line in main()
SYM_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_symmetry.json")
SUITES = ("libero_spatial", "libero_object")
STRIDE = 2
DT = 0.05  # 20 Hz control

KEYS_DOC = {
    "raw_idx": "(T,) int: raw hdf5 obs index of harness frame i = 2*i",
    "state_idx": "(T,) int: hdf5 states[] row the labels were read from = min(raw_idx+1, N-1)",
    "label_exact": "(T,) bool: False only for the last frame of an odd-length demo (labelled from states[N-1], 50 ms stale)",
    "obj_names": "(K,) str: movable objects (env.objects_dict order) then fixtures (env.fixtures_dict order)",
    "obj_category": "(K,) str: LIBERO object category (asset type) per object",
    "obj_movable": "(K,) bool", "obj_of_interest": "(K,) bool: bddl obj_of_interest (goal objects)",
    "obj_pos": "(T,K,3) f32 world xyz of body root (sim.data.body_xpos)",
    "obj_quat_wxyz": "(T,K,4) f32 MuJoCo body_xquat (w,x,y,z), body->world",
    "eef_pos": "(T,3) f32 == robot0_eef_pos observable (site_xpos of eef site)",
    "eef_quat_xyzw": "(T,4) f32 == robot0_eef_quat observable (hand body, xyzw)",
    "arm_qpos": "(T,7) f32 robot0 joint1..7", "gripper_qpos": "(T,2) f32 finger joints",
    "extra_joint_names": "(J,) str articulated-fixture joints (drawers, stove button); empty for libero_object",
    "extra_joint_qpos": "(T,J) f32",
    "success": "(T,) bool: env._check_success() (bddl goal) at that frame",
    "pred_names": "(P,) str 'fn(a,b)' fixed predicate set per suite (see task_facts.json)",
    "pred_vals": "(T,P) bool: LIBERO eval_predicate_fn on the restored state",
    "raw_actions": "(N,7) f32 hdf5 actions at 20 Hz (OSC_POSE dpos(3), drot(3), gripper(1)); actions[u] drives obs[u-1]->obs[u]",
    "raw_eef_pos": "(N,3) f32 hdf5 obs/ee_pos (recorded post-action eef pos of obs[u], == sim(states[u+1]))",
    "raw_eef_quat_xyzw": "(N,4) f32 hdf5 robot_states[:,5:9]", "raw_gripper_qpos": "(N,2) f32 hdf5 obs/gripper_states",
    "raw_ts": "(N,) f32 = 0.05*u (harness convention; hdf5 sim time = states[:,0] = 0.25 + 0.05*u because create_dataset.py dropped 5 warm-up steps)",
    "raw_sim_time": "(N,) f32 states[:,0]",
    "h5_ee_pos": "(T,3) f32 hdf5 obs/ee_pos[raw_idx] (cross-check copy of the recorded value)",
    "meta_json": "() str json: episode/task facts, alignment/conventions, per-episode cross-check errors",
}


def joint_table(sim):
    import mujoco
    m = sim.model._model
    rows = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        jt = int(m.jnt_type[j])
        dim = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(jt, 1)
        rows.append((name, jt, int(m.jnt_qposadr[j]), dim))
    return rows, int(m.nq), int(m.nv)


def restore(sim, state):
    sim.set_state_from_flattened(state)
    sim.forward()


class TaskCtx:
    """One env per task, reused for all 50 demos."""

    def __init__(self, suite, task):
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.bddl_utils as BDDLUtils
        from libero.libero.envs.predicates import eval_predicate_fn
        import robosuite.utils.transform_utils as T
        self.T = T
        self.eval_predicate_fn = eval_predicate_fn
        self.suite, self.task = suite, task
        self.h5path = os.path.join(DATA_ROOT, suite, task + "_demo.hdf5")
        self.f = h5py.File(self.h5path, "r")
        self.bddl_rel = str(self.f["data"].attrs["bddl_file_name"])
        self.bddl = os.path.join(LIBERO_ROOT, self.bddl_rel)
        t0 = time.time()
        self.env = OffScreenRenderEnv(bddl_file_name=self.bddl, camera_heights=128, camera_widths=128)
        self.env.seed(0)
        self.env.reset()
        self.t_env = time.time() - t0
        e = self.e = self.env.env
        sim = self.sim = e.sim
        self.parsed = BDDLUtils.robosuite_parse_problem(self.bddl)
        self.movable = list(e.objects_dict.keys())
        self.fixtures = list(e.fixtures_dict.keys())
        self.names = self.movable + self.fixtures
        self.body_ids = np.array([e.obj_body_id[n] for n in self.names])
        self.obj_movable = np.array([n in e.objects_dict for n in self.names])
        ooi = set(e.obj_of_interest)
        self.obj_of_interest = np.array([n in ooi for n in self.names])
        self.categories = [str(getattr(e.get_object(n), "category_name", "?")) for n in self.names]
        jt, self.nq, self.nv = joint_table(sim)
        self.joint_table = jt
        r = e.robots[0]
        self.eef_site_id = r.eef_site_id
        self.eef_body_id = sim.model.body_name2id(r.robot_model.eef_name)
        self.arm_idx = np.array(r._ref_joint_pos_indexes)
        self.grip_idx = np.array(r._ref_gripper_joint_pos_indexes)
        robot_adr = set(self.arm_idx.tolist()) | set(self.grip_idx.tolist())
        extra = [(n, adr) for (n, typ, adr, dim) in jt if dim == 1 and adr not in robot_adr]
        self.extra_names = [n for n, _ in extra]
        self.extra_adr = np.array([adr for _, adr in extra], dtype=int)
        self.regions = {k: dict(parent=v.parent_name, size=[float(x) for x in np.atleast_1d(v.size)]) for k, v in e.object_sites_dict.items()}
        # ---- fixed predicate set
        self.preds = self.predicate_set()
        for fn, a, b in self.preds:
            assert a in e.object_states_dict and b in e.object_states_dict, (fn, a, b)
        self.pred_fns = [(fn, e.object_states_dict[a], e.object_states_dict[b]) for fn, a, b in self.preds]
        self.pred_names = ["%s(%s,%s)" % p for p in self.preds]
        # ---- one body quaternion per object category (demo_0, first exact state) -> world-up axis in body frame
        st = self.f["data/demo_0/states"][1]
        assert st.shape[0] == 1 + self.nq + self.nv, (st.shape, self.nq, self.nv)
        restore(sim, st)
        self.category_quats = {}
        for n, c in zip(self.names, self.categories):
            q = np.array(sim.data.body_xquat[e.obj_body_id[n]])
            R = T.quat2mat(T.convert_quat(q, to="xyzw"))
            self.category_quats.setdefault(c, dict(example=n, quat_wxyz=[float(x) for x in q], up_in_body=[float(x) for x in (R.T @ np.array([0, 0, 1.0]))]))

    def predicate_set(self):
        e = self.e
        if self.suite == "libero_spatial":
            cab = [k for k, v in e.object_sites_dict.items() if v.parent_name == "wooden_cabinet_1"]
            preds = [("on", "akita_black_bowl_1", "plate_1"), ("on", "akita_black_bowl_1", "cookies_1"), ("on", "akita_black_bowl_2", "plate_1")]
            preds += [("in", "akita_black_bowl_1", k) for k in cab if k.endswith("_region")]     # drawer containers
            preds += [("on", "akita_black_bowl_1", k) for k in cab if k.endswith("_side")]       # cabinet top surface
            preds += [("on", "akita_black_bowl_1", "flat_stove_1"),                             # literal (body-centre based, see notes)
                      ("on", "akita_black_bowl_1", "flat_stove_1_cook_region"),                 # LIBERO-native form (burner site)
                      ("on", "akita_black_bowl_1", "glazed_rim_porcelain_ramekin_1")]           # initial config of task 7
            return preds
        if self.suite == "libero_object":
            return [("in", n, "basket_1_contain_region") for n in self.movable if n != "basket_1"]
        raise ValueError(self.suite)

    def readout(self):
        sim, e, T = self.sim, self.e, self.T
        pos = sim.data.body_xpos[self.body_ids].copy()
        quat = sim.data.body_xquat[self.body_ids].copy()
        eef_pos = np.array(sim.data.site_xpos[self.eef_site_id])
        eef_quat = T.convert_quat(np.array(sim.data.body_xquat[self.eef_body_id]), to="xyzw")
        qpos = sim.data.qpos
        return pos, quat, eef_pos, eef_quat, qpos[self.arm_idx].copy(), qpos[self.grip_idx].copy(), qpos[self.extra_adr].copy()

    def preds_eval(self):
        return [bool(self.eval_predicate_fn(fn, a, b)) for fn, a, b in self.pred_fns]

    def facts(self):
        return dict(suite=self.suite, task=self.task, hdf5=self.h5path, bddl=self.bddl_rel, language=" ".join(self.parsed["language_instruction"]),
                    goal=self.parsed["goal_state"], obj_of_interest=list(self.e.obj_of_interest),
                    objects=[dict(name=n, category=c, movable=bool(m), goal_object=bool(g)) for n, c, m, g in zip(self.names, self.categories, self.obj_movable, self.obj_of_interest)],
                    regions=self.regions, extra_joint_names=self.extra_names, pred_names=self.pred_names, predicates=[list(p) for p in self.preds],
                    state_layout=dict(state_len=1 + self.nq + self.nv, nq=self.nq, nv=self.nv, arm_qpos_adr=self.arm_idx.tolist(), gripper_qpos_adr=self.grip_idx.tolist(),
                                      joints=[dict(name=n, type=typ, qpos_adr=adr, dim=dim) for n, typ, adr, dim in self.joint_table]),
                    category_quats=self.category_quats, env_create_reset_s=self.t_env)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        self.f.close()


def label_episode(ctx, ep, audit=False):
    e, sim = ctx.e, ctx.sim
    demo = ep["meta"]["demo"]
    g = ctx.f["data/demo_%d" % demo]
    states = g["states"][()]
    N = states.shape[0]
    actions = g["actions"][()]
    ee_pos = g["obs/ee_pos"][()]
    joint = g["obs/joint_states"][()]
    grip = g["obs/gripper_states"][()]
    robot_states = g["robot_states"][()]
    keep = np.arange(0, N, STRIDE)
    T2 = len(keep)
    if T2 != ep["n_steps"]:
        raise RuntimeError("frame count %d != episodes.json n_steps %d (N=%d)" % (T2, ep["n_steps"], N))
    s_idx = np.minimum(keep + 1, N - 1)
    label_exact = (keep + 1) <= (N - 1)
    K, P, J = len(ctx.names), len(ctx.pred_fns), len(ctx.extra_adr)
    obj_pos = np.zeros((T2, K, 3), np.float32); obj_quat = np.zeros((T2, K, 4), np.float32)
    eef_pos_a = np.zeros((T2, 3), np.float32); eef_quat_a = np.zeros((T2, 4), np.float32)
    arm_a = np.zeros((T2, 7), np.float32); grip_a = np.zeros((T2, len(ctx.grip_idx)), np.float32); extra_a = np.zeros((T2, J), np.float32)
    succ = np.zeros(T2, bool); pv = np.zeros((T2, P), bool)
    t1 = time.time()
    for i, t in enumerate(s_idx):
        restore(sim, states[t])
        p, q, ep_, eq, aq, gq, xq = ctx.readout()
        obj_pos[i], obj_quat[i], eef_pos_a[i], eef_quat_a[i], arm_a[i], grip_a[i], extra_a[i] = p, q, ep_, eq, aq, gq, xq
        pv[i] = ctx.preds_eval()
        succ[i] = bool(e._check_success())
    t_label = time.time() - t1
    ex = label_exact
    hq = robot_states[keep, 5:9]
    xcheck = dict(arm_qpos_vs_h5_maxabs_exact=float(np.abs(arm_a[ex] - joint[keep][ex]).max()),
                  gripper_vs_h5_maxabs_exact=float(np.abs(grip_a[ex] - grip[keep][ex]).max()),
                  ee_pos_vs_h5_maxabs_exact=float(np.abs(eef_pos_a[ex] - ee_pos[keep][ex]).max()),
                  eef_quat_vs_h5_maxabs_exact=float(np.minimum(np.abs(eef_quat_a[ex] - hq[ex]).max(1), np.abs(eef_quat_a[ex] + hq[ex]).max(1)).max()),
                  arm_qpos_vs_h5_maxabs_stale=(float(np.abs(arm_a[~ex] - joint[keep][~ex]).max()) if (~ex).any() else None))
    aud = None
    if audit:
        # contrast: labels WITHOUT the +1 shift (states[2i]) vs the recorded obs[2i]
        arm0 = np.zeros_like(arm_a); ee0 = np.zeros_like(eef_pos_a); grip0 = np.zeros_like(grip_a)
        for i, t in enumerate(keep):
            restore(sim, states[t])
            _, _, ep_, _, aq, gq, _ = ctx.readout()
            arm0[i], ee0[i], grip0[i] = aq, ep_, gq
        aud = dict(episode=ep["episode"], suite=ctx.suite, task=ctx.task, demo=demo, n_raw=N, n_frames=T2, n_steps_json=ep["n_steps"], n_steps_match=bool(T2 == ep["n_steps"]),
                   n_exact=int(ex.sum()), frac_exact=float(ex.mean()),
                   arm_qpos_maxabs_shift1=xcheck["arm_qpos_vs_h5_maxabs_exact"], arm_qpos_maxabs_shift0=float(np.abs(arm0 - joint[keep]).max()),
                   arm_qpos_medabs_shift1=float(np.median(np.abs(arm_a[ex] - joint[keep][ex]))), arm_qpos_medabs_shift0=float(np.median(np.abs(arm0 - joint[keep]))),
                   gripper_maxabs_shift1=xcheck["gripper_vs_h5_maxabs_exact"], gripper_maxabs_shift0=float(np.abs(grip0 - grip[keep]).max()),
                   ee_pos_maxabs_shift1=xcheck["ee_pos_vs_h5_maxabs_exact"], ee_pos_maxabs_shift0=float(np.abs(ee0 - ee_pos[keep]).max()),
                   stale_last_frame_arm_qpos_err=xcheck["arm_qpos_vs_h5_maxabs_stale"])
    meta = dict(episode=ep["episode"], suite=ctx.suite, task=ctx.task, demo=demo, hdf5=ctx.h5path, bddl=ctx.bddl_rel, stride=STRIDE, n_raw=N, n_frames=T2,
                language=" ".join(ctx.parsed["language_instruction"]), goal=ctx.parsed["goal_state"], obj_of_interest=list(e.obj_of_interest),
                alignment="obs[u] <-> states[u+1]; frame i = obs[2i] labelled from states[2i+1]; last frame of odd-length demo from states[N-1] (label_exact=False)",
                action_windows="obs[2i] -> obs[2j] is driven by raw_actions[2i+1 .. 2j] inclusive",
                conventions=dict(obj_quat="MuJoCo body_xquat wxyz (body->world)", eef_quat="robosuite xyzw", frame="world", raw_ts="0.05*u; sim time = raw_sim_time"),
                label_xcheck=xcheck, label_total_s=t_label, keys=KEYS_DOC)
    out = os.path.join(OUT_ROOT, "labels", "%s.npz" % ep["episode"])
    np.savez_compressed(out, raw_idx=keep, state_idx=s_idx, label_exact=label_exact,
                        obj_names=np.array(ctx.names), obj_category=np.array(ctx.categories), obj_movable=ctx.obj_movable, obj_of_interest=ctx.obj_of_interest,
                        obj_pos=obj_pos, obj_quat_wxyz=obj_quat, eef_pos=eef_pos_a, eef_quat_xyzw=eef_quat_a, arm_qpos=arm_a, gripper_qpos=grip_a,
                        extra_joint_names=np.array(ctx.extra_names, dtype=str), extra_joint_qpos=extra_a, success=succ,
                        pred_names=np.array(ctx.pred_names), pred_vals=pv,
                        raw_actions=actions.astype(np.float32), raw_eef_pos=ee_pos.astype(np.float32), raw_eef_quat_xyzw=robot_states[:, 5:9].astype(np.float32),
                        raw_gripper_qpos=grip.astype(np.float32), raw_ts=(DT * np.arange(N)).astype(np.float32), raw_sim_time=states[:, 0].astype(np.float32),
                        h5_ee_pos=ee_pos[keep].astype(np.float32), meta_json=json.dumps(meta))
    return dict(n_frames=T2, n_raw=N, n_exact=int(ex.sum()), t_label=t_label, xcheck=xcheck, size=os.path.getsize(out), success_final=bool(succ[-1]),
                success_first=bool(succ[0]), preds_first=pv[0].tolist(), preds_last=pv[-1].tolist()), aud


def make_manifest():
    facts = json.load(open(os.path.join(OUT_ROOT, "task_facts.json")))
    sym_path = SYM_JSON
    sym = json.load(open(sym_path)) if os.path.exists(sym_path) else {}
    cats = sorted({o["category"] for t in facts["tasks"] for o in t["objects"]})
    symmetry = {}
    for c in cats:
        s = sym.get(c)
        symmetry[c] = dict(symmetry_class=(s or {}).get("symmetry_class", "unknown"), uncertain=(s or {}).get("uncertain", True),
                           note=(s or {}).get("note", "no mesh analysis"), evidence=(s or {}).get("evidence"))
    man = dict(version="libero_labels_v1", created=time.strftime("%Y-%m-%d %H:%M:%S"),
               episodes="ep00000..ep00499 = libero_spatial (task order below, demo_0..49), ep00500..ep00999 = libero_object; episodes.json order",
               label_files="labels/ep%05d.npz", stride=STRIDE, alignment=facts["alignment"], keys=KEYS_DOC,
               symmetry_classes=dict(cyl="revolution about the world up axis (any yaw is the same shape)", z2="180-degree yaw symmetry (rectangular section)",
                                     z4="90-degree yaw symmetry (square section; a subclass of z2)", none="no yaw symmetry"),
               symmetry_note="geometric symmetry of the collision/visual mesh about the world up axis (measured with mesh_symmetry.py after applying the "
                             "object's actual body rotation in the env); printed labels/textures break these symmetries VISUALLY even when the "
                             "geometry is symmetric, so for image-based orientation prediction treat cyl/z2 as the ambiguity class of the geometry only.",
               object_symmetry=symmetry,
               tasks=[dict(index=i, suite=t["suite"], task=t["task"], episodes="ep%05d..ep%05d" % (50 * i, 50 * i + 49), language=t["language"], goal=t["goal"],
                           bddl=t["bddl"], objects=t["objects"], obj_of_interest=t["obj_of_interest"], regions=t["regions"], extra_joint_names=t["extra_joint_names"],
                           pred_names=t["pred_names"], predicates=t["predicates"], object_symmetry={o["name"]: symmetry[o["category"]]["symmetry_class"] for o in t["objects"]},
                           n_episodes=t["n_episodes"], n_frames=t["n_frames"], n_raw_frames=t["n_raw_frames"]) for i, t in enumerate(facts["tasks"])],
               predicate_notes=facts["predicate_notes"], totals=facts["totals"])
    out = os.path.join(OUT_ROOT, "labels_manifest.json")
    json.dump(man, open(out, "w"), indent=1)
    print("wrote", out, os.path.getsize(out), "bytes; symmetry:", {c: symmetry[c]["symmetry_class"] for c in cats})


def main():
    global LIBERO_ROOT, DATA_ROOT, OUT_ROOT, EPISODES_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", help="the LIBERO repository (for the bddl files and the env)")
    ap.add_argument("--data_root", help="the LIBERO hdf5 datasets (libero_spatial/, libero_object/)")
    ap.add_argument("--episodes_json", help="episodes.json written by build_episodes.py")
    ap.add_argument("--out", required=True, help="output folder (labels/, task_facts.json, labels_manifest.json)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (debug)")
    ap.add_argument("--n_audit", type=int, default=20)
    ap.add_argument("--manifest", action="store_true", help="only build labels_manifest.json from task_facts.json + mesh_symmetry.json")
    a = ap.parse_args()
    LIBERO_ROOT, DATA_ROOT, OUT_ROOT, EPISODES_JSON = a.libero_root, a.data_root, a.out, a.episodes_json
    if a.manifest:
        make_manifest()
        return
    sys.path.insert(0, LIBERO_ROOT)
    eps = [e for e in json.load(open(EPISODES_JSON))["episodes"] if e["meta"]["suite"] in SUITES]
    assert len(eps) == 1000 and eps[0]["episode"] == "ep00000" and eps[-1]["episode"] == "ep00999", len(eps)
    for i, e in enumerate(eps):
        assert e["episode"] == "ep%05d" % i
    if a.limit:
        eps = eps[:a.limit]
    os.makedirs(os.path.join(OUT_ROOT, "labels"), exist_ok=True)
    rng = np.random.RandomState(0)
    audit_ids = set(rng.choice(len(eps), min(a.n_audit, len(eps)), replace=False).tolist())
    t_start = time.time()
    tasks, facts_tasks, audits, failures = {}, [], [], []
    tot_frames = tot_raw = tot_exact = tot_bytes = 0
    order = []
    for idx, ep in enumerate(eps):
        key = (ep["meta"]["suite"], ep["meta"]["task"])
        if key not in tasks:
            for k in list(tasks):     # one live env at a time
                facts_tasks.append(tasks[k].facts_final())
                tasks[k].close(); del tasks[k]
            try:
                ctx = TaskCtx(*key)
            except Exception:
                failures.append(dict(episode=ep["episode"], task=key, error=traceback.format_exc()))
                print("TASK FAILED", key, flush=True)
                traceback.print_exc()
                continue
            ctx.n_eps = ctx.n_frames = ctx.n_raw = 0
            ctx.facts_final = lambda c=ctx: dict(c.facts(), n_episodes=c.n_eps, n_frames=c.n_frames, n_raw_frames=c.n_raw)
            tasks[key] = ctx
            order.append(key)
            print("[%s] task %s/%s env %.1fs preds=%s" % (ep["episode"], key[0], key[1], ctx.t_env, ctx.pred_names), flush=True)
        ctx = tasks.get(key)
        if ctx is None:
            failures.append(dict(episode=ep["episode"], task=key, error="task env failed"))
            continue
        try:
            r, aud = label_episode(ctx, ep, audit=(idx in audit_ids))
        except Exception:
            failures.append(dict(episode=ep["episode"], task=key, error=traceback.format_exc()))
            print("EPISODE FAILED", ep["episode"], flush=True)
            traceback.print_exc()
            continue
        ctx.n_eps += 1; ctx.n_frames += r["n_frames"]; ctx.n_raw += r["n_raw"]
        tot_frames += r["n_frames"]; tot_raw += r["n_raw"]; tot_exact += r["n_exact"]; tot_bytes += r["size"]
        if aud:
            audits.append(aud)
        if idx % 50 == 0 or idx == len(eps) - 1:
            print("  %s N=%d T=%d exact=%d %.1fms/frame arm_err=%.1e ee_err=%.1e succ0=%d succT=%d size=%d  elapsed %.0fs" % (
                ep["episode"], r["n_raw"], r["n_frames"], r["n_exact"], 1e3 * r["t_label"] / r["n_frames"], r["xcheck"]["arm_qpos_vs_h5_maxabs_exact"],
                r["xcheck"]["ee_pos_vs_h5_maxabs_exact"], r["success_first"], r["success_final"], r["size"], time.time() - t_start), flush=True)
    for k in list(tasks):
        facts_tasks.append(tasks[k].facts_final())
        tasks[k].close(); del tasks[k]
    total_s = time.time() - t_start
    # ---- task facts (consumed by mesh_symmetry.py / --manifest)
    facts = dict(alignment="obs[u] <-> states[u+1]; harness frame i = obs[2i] labelled from states[2i+1]; last frame of odd-length demos from states[N-1] flagged label_exact=False",
                 predicate_notes=dict(
                     spatial="on(a,b) with b an object = LIBERO ObjectState.check_ontop: b.z <= a.z, contact(a,b), |xy(a)-xy(b)| < 3 cm (body centres); "
                             "in(a,region) = SiteObjectState.check_contain: a's body centre inside the region box site (in_box, z lower bound -1 cm); "
                             "on(a,<site>) = SiteObject.under: a's centre within the site xy half-sizes and 0..10 cm above it AND contact(parent fixture, a). "
                             "on(akita_black_bowl_1,flat_stove_1) is the literal body-centre test and is FALSE even when the bowl sits on the burner "
                             "(burner is 0.15 m from the stove body origin); on(akita_black_bowl_1,flat_stove_1_cook_region) is the LIBERO-native test used by libero_goal.",
                     object="in(x,basket_1_contain_region): x's body centre inside the basket's contain_region box site (0.061 x 0.061 x 0.069 half-sizes, centred 7.2 cm above the basket origin); "
                            "evaluated for the 6 non-basket movables of each task (target first, then distractors in bddl order)."),
                 tasks=facts_tasks, totals=dict(episodes=len(eps) - len(failures), frames=tot_frames, raw_frames=tot_raw, exact_frames=tot_exact, bytes=tot_bytes, total_s=total_s),
                 failures=failures)
    json.dump(facts, open(os.path.join(OUT_ROOT, "task_facts.json"), "w"), indent=1)
    # ---- alignment audit
    if audits:
        summary = dict(n_audited=len(audits), all_n_steps_match=all(x["n_steps_match"] for x in audits),
                       max_arm_qpos_err_shift1=max(x["arm_qpos_maxabs_shift1"] for x in audits), min_arm_qpos_err_shift0=min(x["arm_qpos_maxabs_shift0"] for x in audits),
                       max_gripper_err_shift1=max(x["gripper_maxabs_shift1"] for x in audits), max_ee_pos_err_shift1=max(x["ee_pos_maxabs_shift1"] for x in audits),
                       min_ee_pos_err_shift0=min(x["ee_pos_maxabs_shift0"] for x in audits),
                       mean_frac_exact=float(np.mean([x["frac_exact"] for x in audits])), n_with_stale_last_frame=sum(x["n_exact"] < x["n_frames"] for x in audits),
                       all_episodes=dict(n=len(eps) - len(failures), frames=tot_frames, exact_frames=tot_exact, frac_exact=tot_exact / max(tot_frames, 1),
                                         n_steps_match_all="every labelled episode asserted n_frames == episodes.json n_steps (a mismatch is recorded as a failure)"),
                       rule="labels use states[2i+1] for harness frame i (obs[2i]); shift1 errors must be ~1e-7 rad (arm) / ~5e-4 m (eef, observable quantisation), shift0 = one control step of motion")
        json.dump(dict(summary=summary, episodes=audits), open(os.path.join(OUT_ROOT, "alignment_audit.json"), "w"), indent=1)
        print("AUDIT", json.dumps(summary))
    print("DONE episodes=%d failures=%d frames=%d raw=%d exact=%d (%.4f) bytes=%d total_s=%.1f" % (
        len(eps) - len(failures), len(failures), tot_frames, tot_raw, tot_exact, tot_exact / max(tot_frames, 1), tot_bytes, total_s), flush=True)


if __name__ == "__main__":
    main()
