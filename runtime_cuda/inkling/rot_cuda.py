"""CUDA C randomized Hadamard transform for decode (a few rows), the counterpart of `rht.rht` (on with `LAMQ_CUDA_C`).

`H = A ⊗ B` with A a natural-order Sylvester matrix (a = 2^p <= 256) and B a small ±1 matrix (Paley or Sylvester),
so for the row-major `[a, b]` view V of a vector: in `y = A (V ∘ S) B^T / sqrt(n)`, out `z = (A^T V B) ∘ S / sqrt(n)`
(A is symmetric). Both are `W = V' Bsel` (Bsel = B^T in, B out) followed by a Walsh-Hadamard transform along a.

Thread t of block (row, jb) owns one element of the `[a, b]` result: row i = t % a, column jb * JT + t / a
(JT = 256 / a). It sums its b-term dot product from shared memory, then the a-point transform runs across threads:
warp shuffles for the low five bits of i, shared memory for the rest (a = 256 only). Blocks never wait on each
other, and there is one barrier (after staging) for a <= 32. Plain float32; the Triton version (`rht.py`) uses tf32x3
tensor cores in ~10 programs a vector and takes 4-7 us inside a CUDA graph on an RTX A4500, mostly latency.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .formats import hadamard_factors_np
from .nvrtc import Kernel

THREADS = 256

_SRC = r"""
// Fused neighbours (template parameters RPRO / REPI):
//   RPRO 1: the input is x * sigmoid(aux)                       (the attention output gate, before o_proj)
//   RPRO 2: the input is softcap(g, c0) sigmoid(g) softcap(u, c1) for x = [g | u] rows of 2n (the MoE down side's
//          SwiGLU); blocks jb == 0 also store it to act (NVFP4 experts read it unrotated)
//   REPI 1: one row (decode): the output goes through the static causal conv (state prev [3][n], weights wc [4][n],
//          updated in place) and, with resid, onto the residual (o_proj's output, `conv4_` + the residual add)
__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + expf(-v)); }
__device__ __forceinline__ float softcap(float v, float cap) { return tanhf(v / cap) * cap; }

// One row: xr -> yr, block column group jb. bneg [JB * RJT][RNBW]: bit k of column j set where Bsel[k][j] = -1.
template <int RA, int RB, int RJT, int RNBW, int ROUT, int RPRO, int REPI>
__device__ __forceinline__ void rot_row(const float* __restrict__ xr, const float* __restrict__ sr,
        const unsigned* __restrict__ bneg, float* __restrict__ yr, int jb, float scale, const float* __restrict__ gate,
        float c0, float c1, float* __restrict__ act, float* __restrict__ prev, const float* __restrict__ wc,
        const float* __restrict__ resid, float* sm) {
    float* V = sm;                       // [RA][RB + 1]: odd row stride, so the lanes (rows) of a warp hit distinct banks
    const int tid = threadIdx.x, i = tid % RA, jl = tid / RA;
    {   // every global load issued before any store: a load-then-store loop waits out one L2 round trip an element
        constexpr int NV = (RA * RB / 4 + 255) / 256;
        float4 xv[NV], sv[NV], av[NV];
        const float* ar = RPRO == 1 ? gate : xr + RA * RB;  // the gate, or the SwiGLU up half
        #pragma unroll
        for (int u = 0; u < NV; u++) {
            const int q = tid + u * 256;
            if (q < RA * RB / 4) {
                xv[u] = __ldg(reinterpret_cast<const float4*>(xr) + q);
                if (!ROUT) sv[u] = __ldg(reinterpret_cast<const float4*>(sr) + q);
                if (RPRO) av[u] = __ldg(reinterpret_cast<const float4*>(ar) + q);
            }
        }
        #pragma unroll
        for (int u = 0; u < NV; u++) {
            const int q = tid + u * 256;
            if (q < RA * RB / 4) {
                float e[4] = {xv[u].x, xv[u].y, xv[u].z, xv[u].w};
                const float f[4] = {sv[u].x, sv[u].y, sv[u].z, sv[u].w};
                const float g[4] = {av[u].x, av[u].y, av[u].z, av[u].w};
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (RPRO == 1) e[c] = e[c] * sigm(g[c]);
                    if (RPRO == 2) e[c] = softcap(e[c], c0) * sigm(e[c]) * softcap(g[c], c1);
                }
                if (RPRO == 2 && act != nullptr && jb == 0)
                    reinterpret_cast<float4*>(act)[q] = make_float4(e[0], e[1], e[2], e[3]);
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int o = 4 * q + c;
                    V[(o / RB) * (RB + 1) + o % RB] = ROUT ? e[c] : e[c] * f[c];
                }
            }
        }
    }
    unsigned nb[RNBW];  // the column's signs, in registers (one shared-memory read a term instead of two)
    #pragma unroll
    for (int q = 0; q < RNBW; q++) nb[q] = __ldg(bneg + (jb * RJT + jl) * RNBW + q);
    __syncthreads();
    // W[i][j] = sum_k ±V[i][k], four accumulators to shorten the dependency chain
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < RB; k++)
        a4[k % 4] += __int_as_float(__float_as_int(V[i * (RB + 1) + k]) ^ (((nb[k / 32] >> (k % 32)) & 1u) << 31));
    float w = (a4[0] + a4[1]) + (a4[2] + a4[3]);
    // RA-point Walsh-Hadamard transform along i (natural order: H[i][i'] = (-1)^popc(i & i')): the low five bits of
    // i are lanes (shuffles), higher bits warps (shared memory)
    constexpr int LW = RA < 32 ? RA : 32;
    #pragma unroll
    for (int h = 1; h < LW; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, w, h);
        w = (i & h) ? o - w : w + o;
    }
    if constexpr (RA > 32) {
        float* Wsm = V + RA * (RB + 1);    // [RA] (RJT = 1)
        #pragma unroll
        for (int h = 32; h < RA; h <<= 1) {
            Wsm[i] = w;
            __syncthreads();
            const float o = Wsm[i ^ h];
            __syncthreads();
            w = (i & h) ? o - w : w + o;
        }
    }
    const int j = jb * RJT + jl;
    if (j < RB) {
        const int o = i * RB + j;
        float z = ROUT ? w * scale * sr[o] : w * scale;
        if (REPI == 1) {  // as conv.py's _conv4_kernel (static), then the residual
            constexpr int C = RA * RB;
            const float p0 = prev[o], p1 = prev[C + o], p2 = prev[2 * C + o];
            const float yv = (p0 * wc[o] + p1 * wc[C + o] + p2 * wc[2 * C + o] + z * wc[3 * C + o] + z) * 0.4472135954999579f;
            prev[o] = p1; prev[C + o] = p2; prev[2 * C + o] = z;
            z = resid != nullptr ? resid[o] + yv : yv;
        }
        yr[o] = z;
    }
}
"""

_ROT_KERNEL = r"""
extern "C" __global__ void __launch_bounds__(256) rot(const float* __restrict__ x, int stride_x,
        const float* __restrict__ signs, int stride_s, int T, const unsigned* __restrict__ bneg, float* __restrict__ y,
        int stride_y, float scale, const float* __restrict__ aux, int stride_aux, float c0, float c1,
        float* __restrict__ act, int stride_act, float* __restrict__ prev, const float* __restrict__ wc,
        const float* __restrict__ resid) {
    extern __shared__ __align__(16) float sm[];
    const long long r = blockIdx.x;
    rot_row<A, B, JT, NBW, OUT, PRO, EPI>(x + r * stride_x, signs + (r / T) * stride_s, bneg, y + r * stride_y,
        blockIdx.y, scale, aux + r * stride_aux, c0, c1, act != nullptr ? act + r * stride_act : nullptr, prev, wc,
        resid, sm);
}
"""


class Rotation:
    """The decode rotation for one size and direction, with optional fused neighbours (`pro`, `epi`: see _SRC)."""

    def __init__(self, n: int, out: bool, device, pro: int = 0, epi: int = 0):
        A, B = hadamard_factors_np(n)
        a, b = A.shape[0], B.shape[0]
        assert a & (a - 1) == 0 and a * b == n and 256 % a == 0 and n % 4 == 0, (n, a, b)
        jt = THREADS // a  # columns a block: one thread an (row, column)
        jb = (b + jt - 1) // jt
        bsel = B if out else B.T  # bsel[k][j], ±1
        nbw = (b + 31) // 32
        bits = np.zeros((jb * jt, nbw), dtype=np.uint32)
        for j in range(b):
            for k in range(b):
                if bsel[k, j] < 0:
                    bits[j, k // 32] |= np.uint32(1 << (k % 32))
        self.bneg = torch.tensor(bits.view(np.int32), device=device)
        self.n, self.out, self.grid_y, self.scale = n, out, jb, 1.0 / math.sqrt(n)
        self.smem = 4 * (a * (b + 1) + a)
        self.pro, self.epi = pro, epi
        self.a, self.b, self.jt, self.nbw = a, b, jt, nbw
        defs = (f"#define A {a}\n#define B {b}\n#define JT {jt}\n#define NBW {nbw}\n#define OUT {int(out)}\n"
                f"#define PRO {pro}\n#define EPI {epi}\n")
        # IEEE expf / tanhf / division (no --use_fast_math): the fused SwiGLU and gate match PyTorch's
        self.kernel = Kernel(defs + _SRC + _ROT_KERNEL, "rot", device, options=(), smem=self.smem)

    def __call__(self, x2: torch.Tensor, signs: torch.Tensor, T: int, aux=None, caps=(1.0, 1.0), act=None,
                 prev=None, wc=None, resid=None) -> torch.Tensor:
        """x2 `[R, n]` float32 rows (`[R, 2n]` for the SwiGLU prologue), signs `[R // T or 1, n]` float32, all rows
        16-byte aligned. `aux` `[R, n]`: the gate (pro 1). `act` `[R, n]`: also store the SwiGLU output (pro 2).
        `prev` / `wc` / `resid`: the conv state, weights and residual (epi 1, R == 1)."""
        R = x2.shape[0]
        need = [x2, signs] + ([aux] if self.pro == 1 else []) + ([act] if act is not None else [])
        for t in need:
            assert t.dtype == torch.float32 and t.stride(-1) == 1 and t.data_ptr() % 16 == 0 and t.stride(0) % 4 == 0, (
                t.dtype, t.stride())
        assert x2.shape[1] == (2 * self.n if self.pro == 2 else self.n), (x2.shape, self.n)
        if self.epi == 1:
            assert R == 1 and prev.is_contiguous() and wc.is_contiguous() and prev.dtype == wc.dtype == torch.float32
            assert resid is None or (resid.dtype == torch.float32 and resid.is_contiguous())
        y = torch.empty(R, self.n, dtype=torch.float32, device=x2.device)
        self.kernel((R, self.grid_y), (THREADS,), [
            x2, x2.stride(0), signs, signs.stride(0), T, self.bneg, y, y.stride(0), self.scale,
            aux if aux is not None else x2, aux.stride(0) if aux is not None else 0, float(caps[0]), float(caps[1]),
            act, act.stride(0) if act is not None else 0, prev, wc, resid], self.smem)
        return y


_ROT: dict = {}


def rotation(n: int, out: bool, device, pro: int = 0, epi: int = 0) -> Rotation:
    key = (n, out, str(device), pro, epi)
    if key not in _ROT:
        _ROT[key] = Rotation(n, out, device, pro, epi)
    return _ROT[key]


_COMBINE_SRC = r"""
// out[c] = resid[c] + conv(sum_k w[k] dn[k][c]): the MoE combine, the static causal conv (state prev [3][C] and
// weights wc [4][C], updated in place, as conv.py's _conv4_kernel) and the residual add of one decode token
extern "C" __global__ void __launch_bounds__(256) combine_conv_add(const float* __restrict__ dn, int stride_dn,
        const float* __restrict__ w, float* __restrict__ prev, const float* __restrict__ wc,
        const float* __restrict__ resid, float* __restrict__ out, int C) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float v = 0.f;
    #pragma unroll
    for (int k = 0; k < K; k++) v = fmaf(dn[k * stride_dn + c], w[k], v);
    const float p0 = prev[c], p1 = prev[C + c], p2 = prev[2 * C + c];
    const float y = (p0 * wc[c] + p1 * wc[C + c] + p2 * wc[2 * C + c] + v * wc[3 * C + c] + v) * 0.4472135954999579f;
    prev[c] = p1; prev[C + c] = p2; prev[2 * C + c] = v;
    out[c] = resid[c] + y;
}
"""

_COMBINE: dict = {}


def combine_conv_add(dn: torch.Tensor, w: torch.Tensor, prev: torch.Tensor, wc: torch.Tensor, resid: torch.Tensor):
    """One token: `resid + conv4_(prev, (dn * w[:, None]).sum(0), wc)`; dn `[K, C]`, w `[K]` (or `[1, K]`)."""
    K, C = dn.shape
    for t in (dn, w, prev, wc, resid):
        assert t.dtype == torch.float32 and t.stride(-1) == 1, (t.dtype, t.stride())
    assert w.numel() == K and prev.is_contiguous() and wc.is_contiguous() and resid.numel() == C and resid.is_contiguous()
    key = (K, str(dn.device))
    if key not in _COMBINE:
        _COMBINE[key] = Kernel(f"#define K {K}\n" + _COMBINE_SRC, "combine_conv_add", dn.device, options=())
    out = torch.empty(1, C, dtype=torch.float32, device=dn.device)
    _COMBINE[key](((C + 255) // 256,), (256,), [dn, dn.stride(0), w.contiguous(), prev, wc, resid, out, C])
    return out



class StageRotations:
    """One launch for a decode stage's output rotations and, optionally, one row of dot products: the rows (each
    with its own size and signs, read in place from the stage's concatenated output) replace a `torch.stack` and one
    rotation launch a size, and the dot job (the attention sinks, `q_in @ sink_w.T`) two cuBLAS launches."""

    def __init__(self, rows, device, dot_rows: int = 0, dot_n: int = 0):
        """rows: list of (n, signs_out `[n]`); the dot job computes `dot_rows` dot products of length `dot_n`."""
        assert rows or dot_rows
        self.rows, self.dot_rows, self.dot_n = [(n, sg.float().contiguous()) for n, sg in rows], dot_rows, dot_n
        params, cases, start, smem = [], [], 0, 0
        self.consts = []
        for q, (n, sg) in enumerate(self.rows):
            rt = rotation(n, True, device)
            end = start + rt.grid_y
            smem = max(smem, rt.smem)
            self.consts.append((rt.bneg, rt.scale))
            params.append(f"const float* X{q}, const float* S{q}, float* Y{q}, const unsigned* BN{q}, float SC{q}")
            cases.append(f"    if (b < {end}) {{ rot_row<{rt.a}, {rt.b}, {rt.jt}, {rt.nbw}, 1, 0, 0>(X{q}, S{q}, BN{q}, Y{q}, "
                         f"b - {start}, SC{q}, nullptr, 0.f, 0.f, nullptr, nullptr, nullptr, nullptr, sm); return; }}")
            start = end
        self.rot_blocks = start
        self.blocks = start + dot_rows
        params.append("const float* DX, const float* DW, float* DY")
        smem = max(smem, 32)
        self.smem = smem
        src = _SRC + f"""
extern "C" __global__ void __launch_bounds__(256) stage_rot({", ".join(params)}) {{
    extern __shared__ __align__(16) float sm[];
    const int b = blockIdx.x;
{chr(10).join(cases)}
    {{  // dot job: DY[h] = sum_c DX[c] DW[h][c]
        const int h = b - {start};
        float a = 0.f;
        for (int c = threadIdx.x; c < {max(dot_n, 1)}; c += 256) a = fmaf(DX[c], DW[(long long)h * {max(dot_n, 1)} + c], a);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (threadIdx.x % 32 == 0) sm[threadIdx.x / 32] = a;
        __syncthreads();
        if (threadIdx.x == 0) {{
            float t = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; w++) t += sm[w];
            DY[h] = t;
        }}
    }}
}}
"""
        self.kernel = Kernel(src, "stage_rot", device, options=(), smem=self.smem)

    def __call__(self, xs, dot_x=None, dot_w=None):
        """xs: one `[n]` (or `[1, n]`) float32 input a row, 16-byte aligned. Returns (rotated rows, dot outputs
        `[1, dot_rows]` or None)."""
        assert len(xs) == len(self.rows)
        args, outs = [], []
        for x, (n, sg), (bneg, scale) in zip(xs, self.rows, self.consts):
            assert x.dtype == torch.float32 and x.numel() == n and x.is_contiguous() and x.data_ptr() % 16 == 0
            y = torch.empty(1, n, dtype=torch.float32, device=x.device)
            outs.append(y)
            args += [x, sg, y, bneg, scale]
        dy = None
        if self.dot_rows:
            assert dot_x.dtype == dot_w.dtype == torch.float32 and dot_x.is_contiguous() and dot_w.is_contiguous()
            assert dot_x.numel() == self.dot_n and dot_w.shape == (self.dot_rows, self.dot_n)
            dy = torch.empty(1, self.dot_rows, dtype=torch.float32, device=dot_x.device)
        args += [dot_x, dot_w, dy]
        self.kernel((self.blocks,), (THREADS,), args, self.smem)
        return outs, dy
