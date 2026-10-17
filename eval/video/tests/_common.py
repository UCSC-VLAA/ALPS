# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Helpers of the tests: the repository root on sys.path, skipping, and plain-python runs of a test module."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def skip(reason):
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(reason)
    raise _Skip(reason)                             # plain-python runs (run_all)


class _Skip(Exception):
    pass


def run_all(namespace):
    """Run the test_* functions of a module without pytest; returns the number of failures."""
    fails = 0
    for k, fn in sorted(namespace.items()):
        if k.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", k)
            except _Skip as e:
                print("SKIP", k, "-", e)
            except Exception as e:                  # noqa: BLE001
                fails += 1
                print("FAIL", k, "-", type(e).__name__, e)
    return fails
