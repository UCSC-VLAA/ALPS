# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Helpers of the CPU tests: load the world-model modules by path (they live in folders that are copied into other
repositories, so they are not a Python package here)."""
import importlib.util
import os
import sys

WM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # eval/world_models
ROOT = os.path.dirname(os.path.dirname(WM))                               # repository root (ALPS_ROOT)
OVERLAY = os.path.join(WM, "dino_wm", "overlay")
REACHER = os.path.join(WM, "reacher")
TOOLS = os.path.join(WM, "tools")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                                              # the alps package


def load(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def registry():
    os.environ.setdefault("ALPS_ROOT", ROOT)
    return load(os.path.join(ROOT, "eval", "encoders.py"), "alps_eval_encoders")
