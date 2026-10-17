# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Runs LeWorldModel's eval.py (with le_wm.patch) with the MuJoCo / dm_control EGL rendering pinned to one GPU.

  TARGET_GPU_UUID=<uuid> CUDA_VISIBLE_DEVICES=<uuid> python eval_pinned.py --config-name=reacher.yaml <eval.py overrides>

The EGL device list ignores CUDA_VISIBLE_DEVICES, and a GPU whose memory is full can be missing from it, so the EGL device
is looked up by the UUID of the GPU (nvidia-smi --query-gpu=uuid). Only the rendering is affected; the observations are
the same EGL renderings. eval.py is taken from the folder of LeWorldModel (env LEWM_ROOT, default: the current folder).
"""
import ctypes
import os
import runpy
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def main():
    target = os.environ["TARGET_GPU_UUID"].replace("GPU-", "").replace("-", "").lower()
    from mujoco.egl import egl_ext as E
    from OpenGL import EGL
    qb = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p,
                          ctypes.POINTER(ctypes.c_ssize_t))(EGL.eglGetProcAddress("eglQueryDeviceBinaryEXT"))
    idx = None
    for k, d in enumerate(E.eglQueryDevicesEXT()):
        buf = (ctypes.c_ubyte * 16)(); n = ctypes.c_ssize_t(0)
        if qb(ctypes.cast(d, ctypes.c_void_p), 0x335C, 16, ctypes.cast(buf, ctypes.c_void_p), ctypes.byref(n)) and n.value == 16 \
                and bytes(buf).hex() == target:
            idx = k; break
    if idx is None:
        sys.exit("[pinned] GPU %s is not among the EGL devices" % target)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(idx)
    print("[pinned] MUJOCO_EGL_DEVICE_ID=%d for GPU %s" % (idx, target), flush=True)
    from dm_control._render.pyopengl import egl_renderer  # noqa: F401  (creates this process's EGL display now, on that device)
    import hdf5plugin  # noqa: F401  (Blosc filter of the pixels of reacher.h5)
    sys.argv[0] = os.path.join(os.environ.get("LEWM_ROOT", os.getcwd()), "eval.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
