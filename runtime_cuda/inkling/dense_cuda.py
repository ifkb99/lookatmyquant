"""CUDA C decode GEMV for a stage of packed matrices (on by default in this fork; `LAMQ_CUDA_C=0` for Triton), the counterpart of
`gemv.DenseStageGEMV`: up to 8 matrices of any formats (trellis or Lloyd-Max) that share an input width, one
launch. The grid's row blocks are split across the matrices; a block copies its matrix's table and input row into
shared memory and runs `moe_cuda`'s `slab_rows` (the attention inputs, the LM head, a single linear).
"""
from __future__ import annotations

import torch

from .moe_cuda import _PRELUDE, _TAB_TYPE, THREADS, _spec
from .nvrtc import Kernel


def _lm_spec(fmt: str, levels: torch.Tensor):
    """(k, g, logK, hash, CH, NW, table) of a Lloyd-Max matrix: CH indices a chunk (a whole number of words)."""
    k = int(fmt[2:])
    CH = 16 if k == 6 else 4  # lm6: 16 indices = 96 bits = 3 words; lm8: 4 = 1 word
    return k, 1, 0, 0, CH, CH * k // 32, levels.float().contiguous().reshape(-1)


def _shape(rows: int, sms: int) -> tuple[int, int]:
    """(rows a lane group, lanes a row) for a stage of `rows` output rows. Measured on an RTX A4500 (56 SMs) inside
    the decode graph: the LM head (~200K rows) runs best with 16 lanes and 4 rows; the attention inputs (8.6K rows)
    with a full warp a row; a single 2304-row projection with a full warp and 2 rows, which keeps >= 4 blocks per SM
    (4 rows: 72 blocks, 0.6x the speed)."""
    if rows >= 32768:
        return 4, 16
    return (4 if rows // 32 >= 4 * sms else 2), 32


class DenseStageCUDA:
    """`mats`: list of (fmt, m, codes, scales, table); `x_rows[i]`: the row of the stacked input matrix i reads.
    Returns the outputs concatenated, `[sum m]`, like `DenseStageGEMV`."""

    def __init__(self, mats, n: int, x_rows, rps: int | None = None, lpr: int | None = None):
        assert 1 <= len(mats) <= 8
        self.ms = [m for _, m, _, _, _ in mats]
        r, l = _shape(sum(self.ms), torch.cuda.get_device_properties(mats[0][2].device).multi_processor_count)
        self.n, self.rps, self.lpr = n, rps or r, lpr or l
        rps, lpr = self.rps, self.lpr
        self.offsets = [sum(self.ms[:i]) for i in range(len(mats))]
        self.total = sum(self.ms)
        rpb = (THREADS // lpr) * rps
        self.args, cases, tab_bytes, rb = [], [], 0, 0
        for i, (fmt, m, codes, scales, table) in enumerate(mats):
            lm = fmt.startswith("lm")
            k, g, logk, hm, CH, nw, tab = _lm_spec(fmt, table) if lm else _spec(fmt, table)
            assert n % CH == 0 and (n * k) % 32 == 0, (fmt, n)
            tt = "float" if lm else _TAB_TYPE[g]
            tab_bytes = max(tab_bytes, tab.numel() * tab.element_size())
            cw = codes.view(torch.int32) if codes.dtype == torch.uint8 else codes
            start, rb = rb, rb + (m + rpb - 1) // rpb
            self.args += [cw, scales.reshape(-1), tab, tab.numel(), m, rb, int(x_rows[i]), self.offsets[i]]
            cases.append(f"""    {"if" if i == 0 else "else if"} (blockIdx.x < RB{i}) {{
        const int r0 = (blockIdx.x - {"0" if i == 0 else f"RB{i - 1}"}) * ROWS_PER_BLOCK + (threadIdx.x / LANES_PER_ROW) * ROWS_PER_HW;
        slab_rows<{str(lm).lower()}, {k}, {g}, {logk}, {hm}u, {CH}, {nw}, ROWS_PER_HW, LANES_PER_ROW, {tt}>(
            x + (long long)XR{i} * stride_x, (const u32*)C{i}, (const u16*)S{i}, (const {tt}*)T{i}, N{i}, y + YO{i}, 0,
            r0, M{i}, n, smem);
    }}""")
        self.blocks = rb
        self.smem = tab_bytes + n * 4
        params = ", ".join(f"const void* C{i}, const void* S{i}, const void* T{i}, int N{i}, int M{i}, int RB{i}, "
                           f"int XR{i}, int YO{i}" for i in range(len(mats)))
        src = (f"#define TAB_BYTES {tab_bytes}\n#define ROWS_PER_HW {rps}\n#define LANES_PER_ROW {lpr}\n"
               f"#define ROWS_PER_BLOCK {rpb}\n" + _PRELUDE + f"""
extern "C" __global__ void __launch_bounds__({THREADS}) dense_stage(const float* __restrict__ x, int stride_x,
        float* __restrict__ y, int n, {params}) {{
    extern __shared__ __align__(16) unsigned char smem[];
{chr(10).join(cases)}
}}
""")
        self.kernel = Kernel(src, "dense_stage", codes.device, smem=self.smem)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x `[R, n]` float32 (stacked inputs); returns the concatenated outputs `[total]`."""
        assert x.dtype == torch.float32 and x.stride(-1) == 1 and x.shape[-1] == self.n, (x.dtype, x.shape, x.stride())
        y = out if out is not None else torch.empty(self.total, dtype=torch.float32, device=x.device)
        self.kernel((self.blocks,), (THREADS,), [x, x.stride(0), y, self.n, *self.args], self.smem)
        return y

    def split(self, y: torch.Tensor):
        return [y[o : o + m] for o, m in zip(self.offsets, self.ms)]
