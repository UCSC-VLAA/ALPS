#!/bin/bash
# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
#
# SSv2 attentive probe of one encoder, data-parallel over the listed GPUs (torch.distributed.run).
#
#   usage: run_ssv2.sh <enc> <gpu list, e.g. 0,1,2,3,4,5,6,7> <accum> [ssv2_probe.py arguments]
#
# The global batch is 256 = number of GPUs x per-GPU batch x <accum> (the paper: 8 x 32 x 1 or 4 x 32 x 2).
# Environment:
#   SSV2_DATA    directory with train.csv, holdout.csv, val.csv (make_ssv2_csv.py)                    required
#   VJEPA2_ROOT  checkout of github.com/facebookresearch/vjepa2 at 204698b45b3712590f06245fbfba32d3be539812  required
#   OUT_DIR      output directory (default runs/ssv2/<enc>)
#   PYTHON       interpreter (default python)
#   MASTER_PORT  rendezvous port (default: random in 20000-39999)
# Example (the protocol of the paper; the first run writes runs/ssv2/alps_stage1/probe_init.pt):
#   bash eval/video/run_ssv2.sh alps_stage1 0,1,2,3,4,5,6,7 1
#   bash eval/video/run_ssv2.sh alps 0,1,2,3 2 --init_from runs/ssv2/alps_stage1/probe_init.pt
set -euo pipefail
if [ $# -lt 3 ]; then
  sed -n '5,18p' "$0" >&2
  exit 2
fi
ENC=$1; GPUS=$2; ACC=$3; shift 3
: "${SSV2_DATA:?set SSV2_DATA to the directory with train.csv, holdout.csv and val.csv}"
: "${VJEPA2_ROOT:?set VJEPA2_ROOT to a checkout of github.com/facebookresearch/vjepa2}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT=${OUT_DIR:-runs/ssv2/$ENC}
PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
mkdir -p "$OUT"
export VJEPA2_ROOT SSV2_DATA CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"${PYTHON:-python}" -m torch.distributed.run --nproc_per_node="$N" --master_port="$PORT" "$HERE/ssv2_probe.py" \
  --enc "$ENC" --out "$OUT" --data "$SSV2_DATA" --accum "$ACC" "$@" 2>&1 | tee -a "$OUT/train.log"
