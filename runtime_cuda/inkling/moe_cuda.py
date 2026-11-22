"""CUDA C decode GEMV for one MoE side (experimental; `LAMQ_CUDA_C=1` turns it on, Triton stays the default).

The same job as `gemv._moe_gemv_kernel` for trellis experts: every (row block, token-expert slot) block decodes
its expert's rows in that expert's format. The difference is where the codebook lives: each block copies its
slab's table (and the activation) into shared memory, which Triton cannot do, so the random table lookups hit
shared memory instead of gathering through L1. A half-warp owns `RPS` rows; its lanes stride over the row's chunks
and reduce with shuffles. The source is generated per side (format constants baked in, like Triton constexprs) and
compiled with NVRTC (`nvrtc.py`). NVFP4 experts are not handled: a side that has them stays on Triton.
"""
from __future__ import annotations

import os

import torch

from .formats import FMT_PARAMS
from .nvrtc import Kernel

THREADS = 256


def enabled() -> bool:
    return os.environ.get("LAMQ_CUDA_C", "0") not in ("", "0")


def _spec(fmt: str, lut: torch.Tensor):
    """(k, g, logK, hash, CH, NW, table): the plain table (one lookup per group), viewed as one word per entry."""
    k, g, K, hm = FMT_PARAMS[fmt]
    CH = {1: 32, 2: 16, 3: 32, 4: 8}[k]
    span = CH * k - k * g + 12  # bits a chunk reads
    nw = (span + 31) // 32
    word = {1: torch.int16, 2: torch.int32, 4: torch.int64}[g]
    table = lut.contiguous().view(word).reshape(-1)
    return k, g, K.bit_length() - 1, hm, CH, nw, table


_PRELUDE = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
__device__ __forceinline__ float h2f_lo(u32 v) { float a; asm("{.reg .b16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l;}" : "=f"(a) : "r"(v)); return a; }
__device__ __forceinline__ float h2f_hi(u32 v) { float a; asm("{.reg .b16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, h;}" : "=f"(a) : "r"(v)); return a; }
__device__ __forceinline__ float h1f(u16 v) { float a; asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(v)); return a; }

// One slab's rows [r0, r0 + RPS) of expert e times the activation in shared memory (transposed: column j of chunk c
// at s_x[j * nch + c], so the lanes of a half-warp read consecutive words).
template <int K, int G, int LOGK, u32 HASH, int CH, int NW, int RPS, int LPR, typename TAB>
__device__ __forceinline__ void slab_rows(const float* __restrict__ xs, const u32* __restrict__ codes,
        const u16* __restrict__ scales, const TAB* __restrict__ tab, int ntab, float* __restrict__ yrow,
        long long e, int r0, int m, int n, unsigned char* smem) {
    TAB* s_tab = reinterpret_cast<TAB*>(smem);
    float* s_x = reinterpret_cast<float*>(smem + TAB_BYTES);
    const int nch = n / CH, wpr = n * K / 32;
    for (int i = threadIdx.x; i < ntab; i += blockDim.x) s_tab[i] = tab[i];
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_x[(i % CH) * nch + i / CH] = xs[i];
    __syncthreads();
    const int hl = threadIdx.x % LPR;
    float acc[RPS];
    #pragma unroll
    for (int r = 0; r < RPS; r++) acc[r] = 0.f;
    const u32* ebase = codes + e * m * wpr;
    for (int c = hl; c < nch; c += LPR) {
        float xr[CH];
        #pragma unroll
        for (int j = 0; j < CH; j++) xr[j] = s_x[j * nch + c];
        const int wb = (n - CH * (c + 1)) * K / 32;
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
            const u32* p = ebase + (long long)min(r0 + r, m - 1) * wpr + wb;
            u32 w[NW + 1];
            #pragma unroll
            for (int i = 0; i < NW; i++) w[i] = (wb + i < wpr) ? __ldg(p + i) : 0u;
            w[NW] = 0u;
            float a = 0.f;
            #pragma unroll
            for (int gi = 0; gi < CH / G; gi++) {
                const int bit = (CH - (gi + 1) * G) * K;
                const u64 two = ((u64)w[bit / 32 + 1] << 32) | w[bit / 32];
                const u32 st = (u32)(two >> (bit % 32)) & 4095u;
                const u32 idx = HASH ? (st * HASH) >> (32 - LOGK) : st;
                const TAB v = s_tab[idx];
                if constexpr (G == 1) {
                    a += h1f((u16)v) * xr[gi];
                } else if constexpr (G == 2) {
                    a += h2f_lo((u32)v) * xr[2 * gi] + h2f_hi((u32)v) * xr[2 * gi + 1];
                } else {
                    const u32 lo = (u32)v, hi = (u32)((u64)v >> 32);
                    a += h2f_lo(lo) * xr[4 * gi] + h2f_hi(lo) * xr[4 * gi + 1] + h2f_lo(hi) * xr[4 * gi + 2] + h2f_hi(hi) * xr[4 * gi + 3];
                }
            }
            acc[r] += a;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
        float s = acc[r];
        #pragma unroll
        for (int o = LPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (hl == 0 && r0 + r < m) yrow[r0 + r] = s * h1f(scales[e * m + r0 + r]);
    }
}
"""

_TAB_TYPE = {1: "u16", 2: "u32", 4: "u64"}


class MoESideCUDA:
    """Decode (one slot per token-expert pair) for the trellis slabs of a `gemv.MoESideGEMV`."""

    def __init__(self, slabs, slab_of: torch.Tensor, loc_of: torch.Tensor, m: int, n: int, rps: int = 4, lpr: int = 16):
        assert 1 <= len(slabs) <= 4
        self.m, self.n, self.rps, self.lpr, self._slabs = m, n, rps, lpr, slabs
        self.slab_of, self.loc_of = slab_of, loc_of
        self.args, cases, tab_bytes = [], [], 0
        for j, (fmt, codes, scales, lut) in enumerate(slabs):
            k, g, logk, hm, CH, nw, table = _spec(fmt, lut)
            assert n % CH == 0 and (n * k) % 32 == 0, (fmt, n)
            tt = _TAB_TYPE[g]
            tab_bytes = max(tab_bytes, table.numel() * table.element_size())
            self.args += [codes, scales, table, table.numel()]
            cases.append(f"case {j}: slab_rows<{k}, {g}, {logk}, {hm}u, {CH}, {nw}, ROWS_PER_HW, LANES_PER_ROW, {tt}>(xs, (const u32*)C{j}, "
                         f"(const u16*)S{j}, (const {tt}*)T{j}, N{j}, yrow, e, r0, m, n, smem); break;")
        self.smem = tab_bytes + n * 4
        params = ", ".join(f"const void* C{j}, const void* S{j}, const void* T{j}, int N{j}" for j in range(len(slabs)))
        src = (f"#define TAB_BYTES {tab_bytes}\n#define ROWS_PER_HW {rps}\n#define LANES_PER_ROW {lpr}\n" + _PRELUDE + f"""
extern "C" __global__ void __launch_bounds__({THREADS}) moe_side(const float* __restrict__ x, int x_div, int stride_x,
        const void* __restrict__ eids, int eid64, const int* __restrict__ slab_of, const int* __restrict__ loc_of,
        float* __restrict__ y, int stride_y, int m, int n, {params}) {{
    extern __shared__ __align__(16) unsigned char smem[];
    const int slot = blockIdx.y;
    const int eid = eid64 ? (int)((const long long*)eids)[slot] : ((const int*)eids)[slot];  // the router's are int64
    const int r0 = blockIdx.x * ({THREADS} / LANES_PER_ROW) * ROWS_PER_HW + (threadIdx.x / LANES_PER_ROW) * ROWS_PER_HW;
    float* yrow = y + (long long)slot * stride_y;
    const float* xs = x + (long long)(slot / x_div) * stride_x;
    const long long e = loc_of[eid];
    switch (slab_of[eid]) {{
        {chr(10).join(cases)}
        default: break;
    }}
}}
""")
        self.kernel = Kernel(src, "moe_side", slab_of.device, smem=self.smem)
        self.rows_per_block = (THREADS // lpr) * rps

    def __call__(self, x: torch.Tensor, eids: torch.Tensor, x_div: int, out: torch.Tensor | None = None) -> torch.Tensor:
        S = eids.numel()
        assert x.dtype == torch.float32 and x.stride(-1) == 1, (x.dtype, x.stride())
        assert eids.dtype in (torch.int32, torch.int64), eids.dtype
        y = out if out is not None else torch.empty(S, self.m, dtype=torch.float32, device=x.device)
        grid = ((self.m + self.rows_per_block - 1) // self.rows_per_block, S)
        self.kernel(grid, (THREADS,), [x, x_div, x.stride(0), eids, int(eids.dtype == torch.int64), self.slab_of, self.loc_of, y, y.stride(0),
                                       self.m, self.n, *self.args], self.smem)
        return y
