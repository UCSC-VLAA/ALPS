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
# LIBERO for the ALPS policy experiments: clone LIBERO at the commit of the paper, apply patches/libero_8f1084e3.patch, write
# LIBERO's path config, and patch robomimic 0.2.0 of the active Python environment (see README.md for the environment).
# usage: LIBERO_ROOT=<dir for the checkout> LIBERO_DATA=<datasets dir> bash setup_libero.sh
#   LIBERO_CONFIG_PATH: folder of LIBERO's config.yaml (default ~/.libero; an existing config.yaml is kept)
set -euo pipefail
: "${LIBERO_ROOT:?set LIBERO_ROOT (where LIBERO is cloned)}"; : "${LIBERO_DATA:?set LIBERO_DATA (the LIBERO datasets folder)}"
COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PATCHES=$HERE/../patches
CFG_DIR=${LIBERO_CONFIG_PATH:-$HOME/.libero}
if [ ! -d "$LIBERO_ROOT/.git" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi
git -C "$LIBERO_ROOT" checkout -q "$COMMIT"
if git -C "$LIBERO_ROOT" apply --reverse --check "$PATCHES/libero_8f1084e3.patch" 2>/dev/null; then
  echo "LIBERO patch already applied"
else
  git -C "$LIBERO_ROOT" apply "$PATCHES/libero_8f1084e3.patch"; echo "LIBERO patch applied"
fi
mkdir -p "$CFG_DIR" "$LIBERO_DATA"
if [ -f "$CFG_DIR/config.yaml" ]; then
  echo "kept $CFG_DIR/config.yaml:"; cat "$CFG_DIR/config.yaml"
else
  B=$(cd "$LIBERO_ROOT/libero/libero" && pwd); D=$(cd "$LIBERO_DATA" && pwd)
  printf 'assets: %s/assets\nbddl_files: %s/bddl_files\nbenchmark_root: %s\ndatasets: %s\ninit_states: %s/init_files\n' \
    "$B" "$B" "$B" "$D" "$B" > "$CFG_DIR/config.yaml"
  echo "wrote $CFG_DIR/config.yaml"
fi
# robomimic 0.2.0 SequenceDataset.__getstate__ (as in the environment of the runs; inert with the fork-context workers)
if SITE=$(python -c 'import os, robomimic; print(os.path.dirname(os.path.dirname(robomimic.__file__)))' 2>/dev/null); then
  if grep -q "def __getstate__" "$SITE/robomimic/utils/dataset.py"; then echo "robomimic already patched"
  elif patch -p1 --dry-run -s -d "$SITE" < "$PATCHES/robomimic-0.2.0_dataset.patch" > /dev/null; then
    patch -p1 -s -d "$SITE" < "$PATCHES/robomimic-0.2.0_dataset.patch"; echo "robomimic patched in $SITE"
  else echo "robomimic in $SITE is not 0.2.0 as expected: patch not applied"; fi
else
  echo "robomimic is not installed in this Python yet: install the environment (README.md), then run this script again"
fi
echo "LIBERO_ROOT=$LIBERO_ROOT  (export it, and LIBERO_CONFIG_PATH=$CFG_DIR if not the default)"
