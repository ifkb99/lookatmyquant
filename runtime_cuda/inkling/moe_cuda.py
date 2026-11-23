"""CUDA C decode GEMV for one MoE side (on by default in this fork; `LAMQ_CUDA_C=0` runs Triton).

The same job as `gemv._moe_gemv_kernel` for trellis experts: every (row block, token-expert slot) block decodes
its expert's rows in that expert's format. The difference is where the codebook lives: each block copies its
slab's table (and the activation) into shared memory, which Triton cannot do, so the random table lookups hit
shared memory instead of gathering through L1. A half-warp owns `RPS` rows; its lanes stride over the row's chunks
and reduce with shuffles. The source is generated per side (format constants baked in, like Triton constexprs) and
compiled with NVRTC (`nvrtc.py`). NVFP4 experts (slab 4) read the unrotated activation and keep their two small
tables (E2M1 values, E4M3 block scales) in shared memory the same way.
"""
from __future__ import annotations

import torch

from .formats import FMT_PARAMS
from .nvrtc import Kernel, enabled  # noqa: F401  (enabled: re-exported)

THREADS = 256




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

__device__ __forceinline__ float tab1(float v) { return v; }
__device__ __forceinline__ float tab1(u16 v) { return h1f(v); }

// The block's table and activation into shared memory, every global load issued before any store (a load-then-store
// loop waits out one L2 round trip an element). The activation is stored transposed (column j of chunk c at
// s_x[j * nch + c]) so the lanes of a half-warp read consecutive words. (Interleaved copies of the table, one a lane
// group, to spread the random lookups over more banks measured slower: 2 copies 5%, 8 copies 50%.)
template <int CH, typename TAB>
__device__ __forceinline__ void stage_inputs(const TAB* __restrict__ tab, int tab16, const float* __restrict__ xs, int n,
        TAB* s_tab, float* s_x) {
    constexpr int TV = (TAB_BYTES / 16 + NTHREADS - 1) / NTHREADS;
    constexpr int XV = (MAX_N / 4 + NTHREADS - 1) / NTHREADS;
    const int nx4 = n / 4, nch = n / CH;
    uint4 tv[TV];
    float4 xv[XV];
    #pragma unroll
    for (int j = 0; j < TV; j++) { const int i = threadIdx.x + j * NTHREADS; if (i < tab16) tv[j] = __ldg(reinterpret_cast<const uint4*>(tab) + i); }
    #pragma unroll
    for (int j = 0; j < XV; j++) { const int i = threadIdx.x + j * NTHREADS; if (i < nx4) xv[j] = __ldg(reinterpret_cast<const float4*>(xs) + i); }
    #pragma unroll
    for (int j = 0; j < TV; j++) {
        const int i = threadIdx.x + j * NTHREADS;
        if (i < tab16) reinterpret_cast<uint4*>(s_tab)[i] = tv[j];
    }
    #pragma unroll
    for (int j = 0; j < XV; j++) {
        const int i = threadIdx.x + j * NTHREADS;
        if (i < nx4) {
            const float v[4] = {xv[j].x, xv[j].y, xv[j].z, xv[j].w};
            #pragma unroll
            for (int q = 0; q < 4; q++) s_x[((4 * i + q) % CH) * nch + (4 * i + q) / CH] = v[q];
        }
    }
}

// One slab's rows [r0, r0 + RPS) of expert e times the activation in shared memory. Trellis rows store their column
// groups from the high end (12-bit states, hashed or not); Lloyd-Max rows (LM) are a plain bitstream of K-bit indices
// into a float32 level table.
template <bool LM, int K, int G, int LOGK, u32 HASH, int CH, int NW, int RPS, int LPR, typename TAB>
__device__ __forceinline__ void slab_rows(const float* __restrict__ xs, const u32* __restrict__ codes,
        const u16* __restrict__ scales, const TAB* __restrict__ tab, int ntab, float* __restrict__ yrow,
        long long e, int r0, int m, int n, unsigned char* smem) {
    TAB* s_tab = reinterpret_cast<TAB*>(smem);
    float* s_x = reinterpret_cast<float*>(smem + TAB_BYTES);
    const int nch = n / CH, wpr = n * K / 32;
    stage_inputs<CH, TAB>(tab, ntab * (int)sizeof(TAB) / 16, xs, n, s_tab, s_x);
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
        const int wb = LM ? c * (CH * K / 32) : (n - CH * (c + 1)) * K / 32;
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
            const u32* p = ebase + (long long)min(r0 + r, m - 1) * wpr + wb;
            u32 w[NW + 1];
            #pragma unroll
            for (int i = 0; i < NW; i++) w[i] = (wb + i < wpr) ? __ldg(p + i) : 0u;
            w[NW] = 0u;
            // one fused multiply-add a weight, straight into the row's accumulator (`a += p * x + q * y` compiles to
            // separate multiplies and adds)
            float a = acc[r];
            #pragma unroll
            for (int gi = 0; gi < CH / G; gi++) {
                const int bit = LM ? gi * K : (CH - (gi + 1) * G) * K;
                const u64 two = ((u64)w[bit / 32 + 1] << 32) | w[bit / 32];
                const u32 st = (u32)(two >> (bit % 32)) & (LM ? (1u << K) - 1u : 4095u);
                const u32 idx = HASH ? (st * HASH) >> (32 - LOGK) : st;
                const TAB v = s_tab[idx];
                if constexpr (G == 1) {
                    a = fmaf(tab1(v), xr[gi], a);
                } else if constexpr (G == 2) {
                    a = fmaf(h2f_lo((u32)v), xr[2 * gi], a);
                    a = fmaf(h2f_hi((u32)v), xr[2 * gi + 1], a);
                } else {
                    const u32 lo = (u32)v, hi = (u32)((u64)v >> 32);
                    a = fmaf(h2f_lo(lo), xr[4 * gi], a);
                    a = fmaf(h2f_hi(lo), xr[4 * gi + 1], a);
                    a = fmaf(h2f_lo(hi), xr[4 * gi + 2], a);
                    a = fmaf(h2f_hi(hi), xr[4 * gi + 3], a);
                }
            }
            acc[r] = a;
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

_NVFP4 = r"""
// NVFP4 expert e: 16-column blocks of two code words (8 E2M1 nibbles each, low nibble first) and one E4M3 scale byte;
// the expert's fp32 outer scale at the end. `xs` is the unrotated activation.
template <int RPS, int LPR>
__device__ __forceinline__ void nvfp4_rows(const float* __restrict__ xs, const u32* __restrict__ codes,
        const unsigned char* __restrict__ bscale, const float* __restrict__ outer, const float* __restrict__ e2m1,
        const float* __restrict__ e4m3, float* __restrict__ yrow, long long e, int r0, int m, int n, unsigned char* smem) {
    float* s_e2m1 = reinterpret_cast<float*>(smem);
    float* s_e4m3 = s_e2m1 + 16;
    float* s_x = reinterpret_cast<float*>(smem + TAB_BYTES);
    const int nch = n / 16, wpr = n / 8;
    for (int i = threadIdx.x; i < 16 + 256; i += blockDim.x) s_e2m1[i] = i < 16 ? e2m1[i] : e4m3[i - 16];
    stage_inputs<16, float>(nullptr, 0, xs, n, nullptr, s_x);
    __syncthreads();
    const int hl = threadIdx.x % LPR;
    float acc[RPS];
    #pragma unroll
    for (int r = 0; r < RPS; r++) acc[r] = 0.f;
    for (int c = hl; c < nch; c += LPR) {
        float xr[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) xr[j] = s_x[j * nch + c];
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
            const long long row = e * m + min(r0 + r, m - 1);
            const uint2 w = __ldg(reinterpret_cast<const uint2*>(codes + row * wpr) + c);
            float a = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; j++) a += s_e2m1[(w.x >> (4 * j)) & 15u] * xr[j] + s_e2m1[(w.y >> (4 * j)) & 15u] * xr[8 + j];
            acc[r] += a * s_e4m3[__ldg(bscale + row * nch + c)];
        }
    }
    const float o = outer[e];
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
        float s = acc[r];
        #pragma unroll
        for (int d = LPR / 2; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
        if (hl == 0 && r0 + r < m) yrow[r0 + r] = s * o;
    }
}
"""

_TAB_TYPE = {1: "u16", 2: "u32", 4: "u64"}

NV_SLAB = 4  # as gemv.MoESideGEMV.NV_SLAB


class MoESideCUDA:
    """Decode (one slot per token-expert pair) for the slabs of a `gemv.MoESideGEMV`: trellis `slabs` and optional
    `nvfp4` = (codes [E, m, n/2] uint8, block scales [E, m, n/16] uint8, outer [E] float32)."""

    def __init__(self, slabs, slab_of: torch.Tensor, loc_of: torch.Tensor, m: int, n: int, rps: int = 4, lpr: int = 16,
                 groups: int = 1, nvfp4=None, minb: int = 3):
        # minb: blocks an SM for __launch_bounds__ (3 caps registers at 80: 2 -> 3 blocks an SM, ~8% faster on an A4500)
        assert len(slabs) <= 4 and (slabs or nvfp4 is not None)
        self.m, self.n, self.rps, self.lpr, self._slabs, self._nvfp4 = m, n, rps, lpr, slabs, nvfp4
        assert n % 4 == 0 and n <= 3072, n
        assert m % groups == 0, (m, groups)
        self.groups = groups
        self.slab_of, self.loc_of = slab_of, loc_of
        self.args, cases, tab_bytes = [], [], 0
        for j, (fmt, codes, scales, lut) in enumerate(slabs):
            k, g, logk, hm, CH, nw, table = _spec(fmt, lut)
            assert n % CH == 0 and (n * k) % 32 == 0, (fmt, n)
            tt = _TAB_TYPE[g]
            nbytes = table.numel() * table.element_size()
            assert nbytes % 16 == 0 and table.data_ptr() % 16 == 0, (fmt, nbytes)  # staged with 16-byte loads
            tab_bytes = max(tab_bytes, nbytes)
            self.args += [codes, scales, table, table.numel()]
            cases.append(f"case {j}: slab_rows<false, {k}, {g}, {logk}, {hm}u, {CH}, {nw}, ROWS_PER_HW, LANES_PER_ROW, {tt}>(xs, "
                         f"(const u32*)C{j}, (const u16*)S{j}, (const {tt}*)T{j}, N{j}, yrow, e, r0, m, n, smem); break;")
        params = ", ".join(f"const void* C{j}, const void* S{j}, const void* T{j}, int N{j}" for j in range(len(slabs)))
        self.has_nv = nvfp4 is not None
        if self.has_nv:
            from .formats import nvfp4_tables

            assert n % 16 == 0, n
            nc, nsb, nout = nvfp4
            self.args += [nc.contiguous(), nsb.contiguous(), nout.float().contiguous(), *nvfp4_tables(nc.device)]
            params += (", " if params else "") + "const void* CN, const void* SN, const void* ON, const void* E2, const void* E4"
            tab_bytes = max(tab_bytes, (16 + 256) * 4)
            cases.append(f"case {NV_SLAB}: nvfp4_rows<ROWS_PER_HW, LANES_PER_ROW>(xr, (const u32*)CN, (const unsigned char*)SN, "
                         "(const float*)ON, (const float*)E2, (const float*)E4, yrow, e, r0, m, n, smem); break;")
        self.smem = tab_bytes + n * 4
        bounds = f"{THREADS}, {minb}" if minb else f"{THREADS}"
        src = (f"#define TAB_BYTES {tab_bytes}\n#define ROWS_PER_HW {rps}\n#define LANES_PER_ROW {lpr}\n"
               f"#define NTHREADS {THREADS}\n#define MAX_N 3072\n" + _PRELUDE
               + (_NVFP4 if self.has_nv else "") + f"""
extern "C" __global__ void __launch_bounds__({bounds}) moe_side(const float* __restrict__ x, int x_div, int stride_x,
        const float* __restrict__ x_raw, int stride_xr,
        const void* __restrict__ eids, int eid64, const int* __restrict__ slab_of, const int* __restrict__ loc_of,
        float* __restrict__ y, int stride_y, int m, int n, int groups, {params}) {{
    // `m` rows a row group; with `groups` > 1 (SQYFR v3) each expert's rows come in `groups` groups of `m`, each with
    // its own format slab and slab index at slab_of / loc_of[expert * groups + group].
    extern __shared__ __align__(16) unsigned char smem[];
    const int slot = blockIdx.y;
    const int rpb = ({THREADS} / LANES_PER_ROW) * ROWS_PER_HW;
    const int bpg = (m + rpb - 1) / rpb;
    const int group = blockIdx.x / bpg;
    const int r0 = (blockIdx.x % bpg) * rpb + (threadIdx.x / LANES_PER_ROW) * ROWS_PER_HW;
    const int eid = (eid64 ? (int)((const long long*)eids)[slot] : ((const int*)eids)[slot]) * groups + group;  // the router's are int64
    float* yrow = y + (long long)slot * stride_y + (long long)group * m;
    const float* xs = x + (long long)(slot / x_div) * stride_x;
    const float* xr = x_raw + (long long)(slot / x_div) * stride_xr;  // NVFP4 experts: the unrotated activation
    const long long e = loc_of[eid];
    switch (slab_of[eid]) {{
        {chr(10).join(cases)}
        default: break;
    }}
}}
""")
        self.kernel = Kernel(src, "moe_side", slab_of.device, smem=self.smem)
        self.rows_per_block = (THREADS // lpr) * rps

    def __call__(self, x: torch.Tensor, eids: torch.Tensor, x_div: int, out: torch.Tensor | None = None,
                 x_raw: torch.Tensor | None = None) -> torch.Tensor:
        """`x` the rotated activation (trellis experts), `x_raw` the unrotated one (NVFP4 experts)."""
        S = eids.numel()
        if self.has_nv:
            assert x_raw is not None, "this side has NVFP4 experts: pass x_raw (the unrotated activation)"
        xr = x_raw if x_raw is not None else x
        for t in (x, xr):  # rows are staged with 16-byte loads
            assert t.dtype == torch.float32 and t.stride(-1) == 1 and t.data_ptr() % 16 == 0 and t.stride(0) % 4 == 0, (
                t.dtype, t.stride())
        assert eids.dtype in (torch.int32, torch.int64), eids.dtype
        y = out if out is not None else torch.empty(S, self.m, dtype=torch.float32, device=x.device)
        rows = self.m // self.groups
        grid = (self.groups * ((rows + self.rows_per_block - 1) // self.rows_per_block), S)
        self.kernel(grid, (THREADS,), [x, x_div, x.stride(0), xr, xr.stride(0), eids, int(eids.dtype == torch.int64), self.slab_of, self.loc_of, y, y.stride(0),
                                       rows, self.n, self.groups, *self.args], self.smem)
        return y
