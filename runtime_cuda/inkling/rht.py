"""Randomized Hadamard transforms in one Triton launch, on tensor cores at ~float32 accuracy (tf32x3).

`H = A ⊗ B` (A: Sylvester ±1, size a = 2^p; B: Paley/Sylvester ±1, size b), so for the row-major `[a, b]` view V of
a vector, `H v = A V B^T`. In: `y = A (V ∘ S) B^T / sqrt(n)`; out: `z = (A^T V B) ∘ S / sqrt(n)`. Program
(row, jb, ib) computes output rows `[ib*IB, +IB)` and columns `[jb*16, +16)` of the `[a, b]` result, building the
needed `V B'` columns chunk by chunk. `tf32x3` splits each operand into two TF32 parts (error ~1e-7 relative); TF32
alone would cost ~1e-3.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .formats import hadamard_factors_np


@triton.jit
def _rht_kernel(
    X, S, A, B, Y, T, stride_x, stride_y, stride_s, inv_sqrt_n,
    a: tl.constexpr, b: tl.constexpr, BP: tl.constexpr, IB: tl.constexpr, KC: tl.constexpr, OUT: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    jb = tl.program_id(1)
    ib = tl.program_id(2)
    s_row = r // T
    inv_sqrt_n = inv_sqrt_n.to(tl.float32)
    j = jb * 16 + tl.arange(0, 16)  # output columns
    j_ok = j < b
    kb = tl.arange(0, BP)  # B contraction index
    kb_ok = kb < b
    i = ib * IB + tl.arange(0, IB)  # output rows
    # B' block [BP, 16]: B' = B^T (in) or B (out)
    if OUT:
        Bp = tl.load(B + kb[:, None] * b + j[None, :], mask=kb_ok[:, None] & j_ok[None, :], other=0.0)
    else:
        Bp = tl.load(B + j[None, :] * b + kb[:, None], mask=kb_ok[:, None] & j_ok[None, :], other=0.0)
    acc = tl.zeros((IB, 16), tl.float32)
    for k0 in tl.static_range(0, a, KC):
        kr = k0 + tl.arange(0, KC)  # rows of V in this chunk
        vidx = kr[:, None] * b + kb[None, :]
        V = tl.load(X + r * stride_x + vidx, mask=kb_ok[None, :], other=0.0).to(tl.float32)
        if not OUT:
            V = V * tl.load(S + s_row * stride_s + vidx, mask=kb_ok[None, :], other=0.0)
        W = tl.dot(V, Bp, input_precision="tf32x3")  # [KC, 16] = (V B')[kr, j]
        if OUT:
            Ap = tl.load(A + kr[None, :] * a + i[:, None])  # A^T[i, k] = A[k, i]
        else:
            Ap = tl.load(A + i[:, None] * a + kr[None, :])
        acc = tl.dot(Ap, W, acc, input_precision="tf32x3")
    oidx = i[:, None] * b + j[None, :]
    z = acc * inv_sqrt_n
    if OUT:
        z = z * tl.load(S + s_row * stride_s + oidx, mask=j_ok[None, :], other=0.0)
    tl.store(Y + r * stride_y + oidx, z, mask=j_ok[None, :])


_FACT: dict = {}
# rows up to which the CUDA C kernel (rot_cuda.py) runs: all of them. It is ~1.4-2.5x faster than this kernel in a
# decode CUDA graph and 2-2.7x at 1024-token prefill on an RTX A4500 (ties at n = 1024)
CUDA_C_MAX_ROWS = 1 << 30


def _factors(n: int, device):
    key = (n, str(device))
    if key not in _FACT:
        A, B = hadamard_factors_np(n)
        _FACT[key] = (torch.tensor(A, dtype=torch.float32, device=device).contiguous(),
                      torch.tensor(B, dtype=torch.float32, device=device).contiguous())
    return _FACT[key]


def rht(x: torch.Tensor, signs: torch.Tensor, out: bool) -> torch.Tensor:
    """x `[S, T, n]` (or `[T, n]`), signs `[S, n]` (or `[n]`): `x[s] @ R_in(signs[s]).T` (out=False) or
    `x[s] @ R_out(signs[s])` (out=True). Returns float32 of x's shape."""
    shape = x.shape
    n = shape[-1]
    if x.dim() == 2:
        x = x[None]
    S, T = x.shape[0], x.shape[1]
    signs = signs.reshape(-1, n)
    assert signs.shape[0] in (1, S)
    x2 = x.reshape(S * T, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if S * T <= CUDA_C_MAX_ROWS:  # the CUDA C rotation (rot_cuda.py) when NVRTC is on
        from . import nvrtc

        if nvrtc.enabled():
            from .rot_cuda import rotation

            sg = signs if signs.dtype == torch.float32 and signs.stride(-1) == 1 else signs.float().contiguous()
            xf = x2 if x2.dtype == torch.float32 else x2.float()
            return rotation(n, out, x.device)(xf, sg, T if signs.shape[0] > 1 else S * T).reshape(shape)
    A, B = _factors(n, x.device)
    a, b = A.shape[0], B.shape[0]
    assert a >= 16 and a & (a - 1) == 0, a
    # 16 output rows a program and a contraction chunk of up to 64 rows: same speed as (32, 32) for a <= 32, and
    # 9.2 -> 6.3 us for n = 3072 (a = 256) inside a CUDA graph on an RTX A4500
    IB = 16
    y = torch.empty(S * T, n, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _rht_kernel[(S * T, triton.cdiv(b, 16), a // IB)](
            x2, signs, A, B, y, T if signs.shape[0] > 1 else S * T, x2.stride(0), y.stride(0), signs.stride(0),
            1.0 / math.sqrt(n), a=a, b=b, BP=max(16, triton.next_power_of_2(b)), IB=IB, KC=min(a, 64), OUT=out,
            num_warps=4,
        )
    return y.reshape(shape)
