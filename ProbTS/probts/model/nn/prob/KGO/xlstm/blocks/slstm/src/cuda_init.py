# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel
# Modified for KGO: CUDA toolchain compatibility and kernel-loading diagnostics.

import os
import re
import subprocess
from typing import Sequence, Union
import logging

import torch
from torch.utils.cpp_extension import load as _load

LOGGER = logging.getLogger(__name__)


def _nvcc_major_minor():
    """Return (major, minor) of the system nvcc, or None if unavailable."""
    nvcc = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(nvcc, "bin", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _extra_nvcc_compat_flags():
    """CUDA 12.5+ flips the default of -static-global-template-stub to true, which
    breaks linking of __global__ template instantiations used by the sLSTM kernels.
    Explicitly force it off on affected toolchains. Harmless on older nvcc (we gate
    on version)."""
    ver = _nvcc_major_minor()
    if ver is not None and ver >= (12, 5):
        return ["-static-global-template-stub=false"]
    return []


def defines_to_cflags(defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]]):
    cflags = []
    print(defines)
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    try:
        _inc_paths = torch.utils.cpp_extension.include_paths(cuda=True)
    except TypeError:
        _inc_paths = torch.utils.cpp_extension.include_paths()
    os.environ["CUDA_LIB"] = os.path.join(os.path.split(_inc_paths[-1])[0], "lib")


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]

    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            # "-dbg=1",
            '-Xptxas="-v"',
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_89,code=sm_89",
            # PTX fallback: JIT-compiled by the driver on newer GPUs (Hopper, Blackwell).
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            *_extra_nvcc_compat_flags(),
            *extra_cflags,
            *extra_cuda_cflags,
        ],
    }
    print(myargs)
    myargs.update(**kwargs)
    # PyTorch serializes extension builds with its own file lock.
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod
