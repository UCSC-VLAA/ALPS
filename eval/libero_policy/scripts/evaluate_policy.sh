#!/usr/bin/env bash
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
#
# The three evaluations of a trained policy, as for every policy of the paper: eval50 and repA without a seed, repB with
# EVAL_SEED=1; 50 rollouts per task each (evaluate_policy.py). One GPU; the simulators render with EGL on the same GPU.
# usage: evaluate_policy.sh <run dir> [gpu=0] [more evaluate_policy.py options, e.g. --encoder-weights <file>]
# environment: LIBERO_ROOT (the patched LIBERO checkout, required); ALPS_REPO (default: this repository)
set -euo pipefail
[ $# -ge 1 ] || { grep '^# usage\|^# environment' "$0"; exit 2; }
RUN_DIR=$(cd "$1" && pwd); GPU=${2:-0}; shift; [ $# -ge 1 ] && shift
: "${LIBERO_ROOT:?set LIBERO_ROOT to the patched LIBERO checkout}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ALPS_REPO=${ALPS_REPO:-$(cd "$HERE/../../.." && pwd)}
cd "$LIBERO_ROOT"
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# renderers on the evaluation GPU; one BLAS / OpenMP thread per simulator process (as in the runs)
export MUJOCO_EGL_DEVICE_ID=$GPU EGL_DEVICE_ID=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for spec in "eval50 -" "repA -" "repB 1"; do
  TAG=${spec% *}; S=${spec#* }
  SEED_ARGS=(); [ "$S" = "-" ] || SEED_ARGS=(--eval-seed "$S")
  echo "$(date -u +%FT%TZ) EVAL $TAG ${SEED_ARGS[*]:-} $RUN_DIR"
  python "$HERE/evaluate_policy.py" --run-dir "$RUN_DIR" --device "cuda:$GPU" --tag "$TAG" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} "$@"
done
python "$HERE/analyze_policies.py" runs "$RUN_DIR"
