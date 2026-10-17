# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Settings shared by the DINO-WM scripts; sourced, not run. Environment: see env.sh.example.
: "${DINO_WM:?set DINO_WM (see env.sh.example)}"
: "${ALPS_ROOT:?set ALPS_ROOT}"
: "${DATASET_DIR:?set DATASET_DIR}"
: "${WM_ROOT:?set WM_ROOT}"
export ALPS_ROOT DATASET_DIR
SEED=${SEED:-1}
SUBSETS=$WM_ROOT/subsets
STATS=$WM_ROOT/stats
RUNS=$WM_ROOT/runs
RESULTS=$WM_ROOT/results
ENCODERS="alps alps_stage1 dinov2_b14 dinov3_b16 siglip2_b16 cradio_v3_b"

run_name() { echo "${1}_${2}_wm_s${SEED}"; }           # <task> <encoder>

# T = fixed endpoint (update), PLAN_CFG = DINO-WM planning config, NEV = planning instances, SHARD = instances per shard,
# SIM = 1 for the PyFleX tasks (shards start one after the other), PLAN_EXTRA = overrides of the paper's planning runs
task_settings() {
  case $1 in
    pusht)      T=31000; PLAN_CFG=plan_pusht;      NEV=50; SHARD=10; SIM=0; PLAN_EXTRA="+planner.sub_planner.rollout_chunk=100 +tf32=true" ;;
    wall)       T=15500; PLAN_CFG=plan_wall;       NEV=50; SHARD=10; SIM=0; PLAN_EXTRA="+planner.sub_planner.rollout_chunk=100 +tf32=true" ;;
    point_maze) T=15500; PLAN_CFG=plan_point_maze; NEV=50; SHARD=10; SIM=0; PLAN_EXTRA="+planner.sub_planner.rollout_chunk=100 +tf32=true" ;;
    granular)   T=7750;  PLAN_CFG=plan_granular;   NEV=10; SHARD=2;  SIM=1; PLAN_EXTRA="+planner.sub_planner.rollout_chunk=100 +tf32=true planner.sub_planner.eval_every=1000" ;;
    rope)       T=7750;  PLAN_CFG=plan_rope;       NEV=50; SHARD=10; SIM=1; PLAN_EXTRA="+planner.sub_planner.rollout_chunk=100 +tf32=true planner.sub_planner.eval_every=1000" ;;
    *) echo "unknown task $1" >&2; return 1 ;;
  esac
}
