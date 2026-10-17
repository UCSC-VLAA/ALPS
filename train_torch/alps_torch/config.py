# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Minimal configs: a Python file with get_config() returning a (nested) Config; command-line overrides
`--config.a.b=value` (Python literals, otherwise strings), as with the ml_collections flags of train_jax."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple


class Config(dict):
    """dict with attribute access; nested dicts become Configs."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Config(mod.get_config())


def _literal(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    for o in overrides:
        key, _, val = o.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node:
                node[p] = Config()
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = _literal(val)
    return cfg


def parse_args(argv: Optional[List[str]] = None) -> Tuple[Config, str]:
    """--config=<file> --workdir=<dir> [--config.key=value ...]"""
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path = workdir = None
    overrides = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if "=" not in a and i + 1 < len(argv) and a in ("--config", "--workdir"):
            a = f"{a}={argv[i + 1]}"
            i += 1
        if a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
        elif a.startswith("--workdir="):
            workdir = a.split("=", 1)[1]
        elif a.startswith("--config."):
            overrides.append(a[len("--config."):])
        else:
            raise SystemExit(f"unknown argument {a!r} (use --config=, --workdir=, --config.<key>=<value>)")
        i += 1
    if cfg_path is None or workdir is None:
        raise SystemExit("usage: --config=<config file> --workdir=<dir> [--config.<key>=<value> ...]")
    cfg = apply_overrides(load_config(cfg_path), overrides)
    return cfg, workdir


def dump(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)
