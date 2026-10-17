#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Clones LeWorldModel (with le_wm.patch: episode sharding of eval.py) and stable-worldmodel at the commits of the paper:
#   bash setup_lewm.sh <new folder for le-wm> <new folder for stable-worldmodel>
set -euo pipefail
LEWM=${1:?le-wm folder}; SWM=${2:?stable-worldmodel folder}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for d in "$LEWM" "$SWM"; do if [ -e "$d" ]; then echo "$d exists; choose a new folder" >&2; exit 1; fi; done
git clone https://github.com/lucas-maes/le-wm.git "$LEWM"
git -C "$LEWM" checkout -q 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
git -C "$LEWM" apply "$HERE/le_wm.patch"
git clone https://github.com/galilai-group/stable-worldmodel.git "$SWM"
git -C "$SWM" checkout -q 63988116d34cde56aea1240d5e58eb158ac67dc0
echo "le-wm in $LEWM (8edfeb33 + le_wm.patch), stable-worldmodel in $SWM (63988116)"
