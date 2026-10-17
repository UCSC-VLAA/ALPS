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
set -euo pipefail
usage() {
  cat <<'EOF'
Train one LIBERO policy on frozen encoder tokens with the recipe of the ALPS paper: LIBERO's BC-Transformer, multitask on
the 50 demonstrations of each of the 10 tasks, the frozen encoder + attention pool of frozen_encoder_pool.yaml, no evaluation
during training (the last checkpoint is evaluated afterwards with evaluate_policy.sh).

usage: train_policy.sh <encoder> <suite: 10 | goal | spatial | object> <training seed> [<instruction embeddings .npz>]
  encoder     alps | alps_stage1 | clip_b16_qg | siglip2_b16 | dinov2_b14_timm | dinov3_b16 | cradio_v3_b
  embeddings  omit for LIBERO's frozen BERT; an .npz of embed_instructions.py for an encoder's own text tower
environment
  LIBERO_ROOT      the LIBERO checkout patched with patches/libero_8f1084e3.patch (required)
  ENCODER_WEIGHTS  alps / alps_stage1: safetensors file (default $ALPS_WEIGHTS / $ALPS_STAGE1_WEIGHTS); cradio_v3_b: the
                   checkpoint (default $RADIO_CKPT; RADIO_REPO must point to a clone of NVlabs/RADIO); siglip2_b16: optional
                   state dict of SigLIP 2 refined by Stage 2 (Sec. S3)
  GPU (0)  EPOCHS (25 on suite 10, else 50)  NUM_WORKERS (4)
  RUN_TAG          default <encoder>_<suite>_e<EPOCHS>_s<seed>[_<npz name>][_<weights name, timm encoders>]
  EXTRA_OVERRIDES  more hydra overrides. The defaults are a truly frozen encoder (encoder_train_mode=eval) and the released
                   model's arithmetic (alps_port_numerics=false); the paper's runs correspond to
                   policy.image_encoder.network_kwargs.encoder_train_mode=follow (+ alps_port_numerics=true for ALPS;
                   C-RADIOv3-B: the defaults), which run_paper_policies.sh passes.
The run directory is $LIBERO_ROOT/experiments/<benchmark>/Multitask/BCTransformerPolicy_seed<seed>_<RUN_TAG>/run_001.
Paths must not contain spaces or '=' (hydra overrides).
EOF
}
[ $# -ge 3 ] || { usage; exit 2; }
ENC=$1; SUITE=$2; SEED=$3; EMB=${4:-}
: "${LIBERO_ROOT:?set LIBERO_ROOT to the patched LIBERO checkout}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ALPS_REPO=${ALPS_REPO:-$(cd "$HERE/../../.." && pwd)}
absfile() { [ -f "$1" ] || { echo "no file $1" >&2; exit 2; }; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
case $SUITE in
  10) BM=LIBERO_10; DEF_EP=25 ;;
  goal|spatial|object) BM=LIBERO_$(echo "$SUITE" | tr '[:lower:]' '[:upper:]'); DEF_EP=50 ;;
  *) echo "unknown suite '$SUITE' (10 | goal | spatial | object)"; exit 2 ;;
esac
EP=${EPOCHS:-$DEF_EP}; GPU=${GPU:-0}; NW=${NUM_WORKERS:-4}
W=${ENCODER_WEIGHTS:-}
case $ENC in
  alps) W=${W:-${ALPS_WEIGHTS:-}} ;;
  alps_stage1) W=${W:-${ALPS_STAGE1_WEIGHTS:-}} ;;
  cradio_v3_b) W=${W:-${RADIO_CKPT:-}}; : "${RADIO_REPO:?set RADIO_REPO to a clone of github.com/NVlabs/RADIO}" ;;
esac
case $ENC in alps|alps_stage1|cradio_v3_b) [ -n "$W" ] || { echo "$ENC needs ENCODER_WEIGHTS"; exit 2; } ;; esac
TAG=${RUN_TAG:-${ENC}_${SUITE}_e${EP}_s${SEED}}
OVR=()
if [ -n "$W" ]; then W=$(absfile "$W"); OVR+=("policy.image_encoder.network_kwargs.weights=$W"); fi
if [ -n "$EMB" ]; then
  EMB=$(absfile "$EMB"); OVR+=("task_embedding_format=file:$EMB")
  [ -n "${RUN_TAG:-}" ] || TAG=${TAG}_$(basename "$EMB" .npz)
fi
case $ENC in   # a timm encoder with other weights (SigLIP 2 + Stage 2): its own run name
  alps|alps_stage1|cradio_v3_b) ;;
  *) if [ -n "$W" ] && [ -z "${RUN_TAG:-}" ]; then B=$(basename "$W"); TAG=${TAG}_${B%.*}; fi ;;
esac
cd "$LIBERO_ROOT"
EXPDIR=experiments/$BM/Multitask/BCTransformerPolicy_seed${SEED}_$TAG
if ls -d "$EXPDIR"/run_* > /dev/null 2>&1 && [ -z "${ALLOW_NEW_RUN:-}" ]; then
  echo "$EXPDIR already has a run (ALLOW_NEW_RUN=1 trains another one)"; exit 1
fi
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "$(date -u +%FT%TZ) TRAIN $TAG: $BM, encoder $ENC, seed $SEED, $EP epochs, GPU $GPU"
# LIBERO_EXP_SUFFIX: own experiment directory per policy; LIBERO_SKIP_EP0=1: no zero-shot loss pass at epoch 0 (both as in the runs)
# shellcheck disable=SC2086
LIBERO_EXP_SUFFIX=$TAG LIBERO_SKIP_EP0=1 CUDA_VISIBLE_DEVICES=$GPU python libero/lifelong/main.py \
  seed="$SEED" benchmark_name=$BM policy=bc_transformer_policy lifelong=multitask policy/image_encoder=frozen_encoder_pool \
  policy.image_encoder.network_kwargs.encoder="$ENC" policy.image_encoder.network_kwargs.flip_ud=true \
  ${OVR[@]+"${OVR[@]}"} \
  train.num_workers="$NW" eval.num_workers=0 train.n_epochs="$EP" lifelong.eval_in_train=false eval.eval_every="$EP" eval.eval=false \
  ${EXTRA_OVERRIDES:-}
RUNDIR=$(ls -dt "$EXPDIR"/run_* 2>/dev/null | head -1)
if [ -z "$RUNDIR" ] || [ ! -f "$RUNDIR/multitask_model_ep$EP.pth" ]; then echo "TRAIN_FAIL $TAG: no $EXPDIR/run_*/multitask_model_ep$EP.pth"; exit 1; fi
echo "$TAG" > "$RUNDIR/cell_id.txt"
echo "$(date -u +%FT%TZ) TRAIN_OK $TAG -> $LIBERO_ROOT/$RUNDIR"
echo "next: bash $HERE/evaluate_policy.sh $LIBERO_ROOT/$RUNDIR $GPU"
