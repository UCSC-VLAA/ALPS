# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Settings shared by the Reacher scripts; sourced, not run. Environment: see env.sh.example.
: "${LEWM_ROOT:?set LEWM_ROOT (see env.sh.example)}"
: "${SWM_ROOT:?set SWM_ROOT}"
: "${ALPS_ROOT:?set ALPS_ROOT}"
: "${STABLEWM_HOME:?set STABLEWM_HOME}"
: "${REACHER_OUT:?set REACHER_OUT}"
export LEWM_ROOT SWM_ROOT ALPS_ROOT STABLEWM_HOME
REACHER_DIR=${REACHER_DIR:-$ALPS_ROOT/eval/world_models/reacher}
export PYTHONPATH=$REACHER_DIR:${PYTHONPATH:-}
PYTHON=${PYTHON:-python}
SEED=${SEED:-1}
ENCODERS="alps alps_stage1 dinov2_b14_196 dinov3_b16 siglip2_b16 cradio_v3_b"
STATS=$REACHER_OUT/stats
RESULTS=$REACHER_OUT/results
run_name() { echo "reacher_wm_${1}_s${SEED}"; }       # <encoder>; the save_pretrained folder under $STABLEWM_HOME/checkpoints
