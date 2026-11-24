"""CUDA C kernels compiled on the user's machine at load time, with no build step.

NVRTC (`libnvrtc`, which PyTorch's CUDA wheels already install) compiles the source for the GPU in use; the driver
(`libcuda`) loads the result and launches it on torch's current stream, so a launch can be captured in a CUDA graph.
No `nvcc`, C++ compiler or CUDA toolkit is needed. The kernels include no headers: a pip-only install has none on
NVRTC's search path, so fp16 conversions are inline PTX.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import sys

import torch

_libs: dict[str, ctypes.CDLL] = {}


def enabled() -> bool:
    """The CUDA C kernels are on unless `LAMQ_CUDA_C=0` (this fork's default, tuned on RTX A4500s; upstream runs
    Triton only)."""
    return os.environ.get("LAMQ_CUDA_C", "1") not in ("", "0")


_modules: dict[tuple, tuple] = {}  # (source, device index, options) -> (module, cubin): one compile each


def _nvrtc() -> ctypes.CDLL:
    if "nvrtc" not in _libs:
        cands = []
        for p in sys.path:  # the nvidia-cuda-nvrtc wheel that torch depends on
            cands += sorted(glob.glob(os.path.join(p, "nvidia", "*", "lib", "libnvrtc.so*")))
        cands = [c for c in cands if "builtins" not in c and ".alt." not in c] + ["libnvrtc.so"]
        for c in cands:
            try:
                _libs["nvrtc"] = ctypes.CDLL(c)
                break
            except OSError:
                continue
        else:
            raise OSError("libnvrtc not found: install a CUDA build of PyTorch")
    return _libs["nvrtc"]


def _cuda() -> ctypes.CDLL:
    if "cuda" not in _libs:
        _libs["cuda"] = ctypes.CDLL("libcuda.so.1" if os.name != "nt" else "nvcuda.dll")
    return _libs["cuda"]


def _check(r: int, what: str):
    if r != 0:
        s = ctypes.c_char_p()
        _cuda().cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what}: CUDA error {r} ({s.value.decode() if s.value else '?'})")


def compile_cubin(src: str, name: str, arch: int, options=()) -> bytes:
    """`src` compiled for `sm_{arch}`. Raises with NVRTC's log on a compile error."""
    nv = _nvrtc()
    prog = ctypes.c_void_p()
    # The program is named after its source's hash: NVRTC 13.0 once returned an empty binary (and reported success)
    # for one source named "moe_side.cu" that compiled under any other name; the cause is unknown.
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    if nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), f"{name}_{tag}.cu".encode(), 0, None, None) != 0:
        raise RuntimeError("nvrtcCreateProgram failed")
    try:
        opts = [f"--gpu-architecture=sm_{arch}", "--std=c++17", *options]
        r = nv.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts]))
        if r != 0:
            size = ctypes.c_size_t()
            nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nv.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC could not compile {name}:\n{log.value.decode(errors='replace')}")
        size = ctypes.c_size_t()
        if nv.nvrtcGetCUBINSize(prog, ctypes.byref(size)) != 0 or size.value == 0:
            raise RuntimeError(f"NVRTC reported success for {name} but returned no binary")
        buf = ctypes.create_string_buffer(size.value)
        if nv.nvrtcGetCUBIN(prog, buf) != 0:
            raise RuntimeError(f"nvrtcGetCUBIN failed for {name}")
        return buf.raw
    finally:
        nv.nvrtcDestroyProgram(ctypes.byref(prog))


class Kernel:
    """One `extern "C" __global__` function of `src`, compiled for `device`. Call with tensors (passed as pointers),
    Python ints (C `int`) and floats (C `float`), in the kernel's parameter order."""

    def __init__(self, src: str, func: str, device: torch.device, options=("--use_fast_math",), smem: int = 0):
        self.device = torch.device(device)
        key = (src, self.device.index, tuple(options))
        if key not in _modules:  # one compile per source: its kernels share the module
            with torch.cuda.device(self.device):
                torch.empty(1, device=self.device)  # torch's primary context, current on this thread
                cc = torch.cuda.get_device_capability(self.device)
                cubin = compile_cubin(src, func, cc[0] * 10 + cc[1], options)
                mod = ctypes.c_void_p()
                _check(_cuda().cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
            _modules[key] = (mod, cubin)
        self.module, self.cubin = _modules[key]
        self.func = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            _check(_cuda().cuModuleGetFunction(ctypes.byref(self.func), self.module, func.encode()), "cuModuleGetFunction")
        if smem > 48 * 1024:  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _check(_cuda().cuFuncSetAttribute(self.func, 8, smem), "cuFuncSetAttribute")

    def __call__(self, grid, block, args, smem: int = 0):
        vals = []
        for a in args:
            if isinstance(a, torch.Tensor):
                vals.append(ctypes.c_void_p(a.data_ptr()))
            elif a is None:  # an optional pointer: NULL
                vals.append(ctypes.c_void_p(0))
            elif isinstance(a, int):
                vals.append(ctypes.c_int(a))
            elif isinstance(a, float):
                vals.append(ctypes.c_float(a))
            else:
                raise TypeError(f"kernel argument of type {type(a).__name__}")
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        with torch.cuda.device(self.device):  # the module lives in this device's context
            stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            _check(_cuda().cuLaunchKernel(self.func, *g, *b, smem, stream, ptrs, None), "cuLaunchKernel")
