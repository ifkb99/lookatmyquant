"""Packed attention linears and LM head for the standalone model (`model.py`): the codes stay on the GPU and run
through the decode GEMV (`gemv.py`, one or two token rows) or the fused decode + matmul (`triton_kernels.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .formats import ID_TO_FMT, rht_in, rht_out
from .gemv import packed_gemv
from .triton_kernels import packed_matmul

GEMV_MAX_ROWS = 2  # token rows per matrix up to which the decode GEMV beats the tensor-core kernel

FMT_ID = {v: k for k, v in ID_TO_FMT.items()}


def _t(a, device) -> torch.Tensor:
    a = np.asarray(a)
    if a.dtype == np.uint32:
        a = a.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _signs(a, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a).astype(np.float32)).to(device)


DEQUANT_MIN_ROWS = 32  # dense calls with at least this many rows decode the matrix once and use cuBLAS
DEQUANT_MAX_WEIGHTS = 1 << 24  # ... up to 64 MB of float32 weights (the LM head's large buckets keep the fused kernel)


class _Mat:
    """One packed matrix (or a stack of same-format experts) and how to run it."""

    def __init__(self, fmt, m, n, codes, scales, lut=None, levels=None):
        self.fmt, self.m, self.n = fmt, m, n
        self.codes, self.scales, self.lut, self.levels = codes, scales, lut, levels

    def __call__(self, x16, **kw):
        if "block_expert" not in kw and x16.shape[0] <= GEMV_MAX_ROWS:
            return self.gemv(x16, out=kw.get("out"))
        if ("block_expert" not in kw and x16.shape[0] >= DEQUANT_MIN_ROWS and self.m * self.n <= DEQUANT_MAX_WEIGHTS
                and self.codes.dim() <= 3 and (self.codes.dim() < 3 or self.codes.shape[0] == 1) and x16.is_cuda):
            # prefill: decode the matrix once (float32, row scales applied) and run one cuBLAS float32 matmul; the
            # fused kernel decodes every weight again for each 16-token block. 2.5-14x faster at 1024 tokens on an
            # RTX A4500, and closer to float64 (~4e-7 relative against ~1e-5: no float16 split of the activation)
            from .triton_kernels import packed_dequant

            w = packed_dequant(self.codes, self.scales, self.fmt, self.m, self.n, lut=self.lut, levels=self.levels)
            out = kw.get("out")
            if out is not None:
                return torch.mm(x16.float(), w.T, out=out)
            return x16.float() @ w.T
        return packed_matmul(x16, self.codes, self.scales, self.fmt, self.m, self.n, lut=self.lut, levels=self.levels, **kw)

    def gemv(self, x, expert_ids=None, x_div=1, out=None, zero_skipped=True):
        table = self.levels if self.fmt.startswith("lm") else self.lut
        if expert_ids is None and x.shape[0] == 1 and x.dtype == torch.float32 and x.stride(-1) == 1:
            from .nvrtc import enabled

            if enabled():  # the experimental CUDA C kernel, a one-matrix stage (dense_cuda.py)
                if getattr(self, "_cuda_c", None) is None:
                    from .dense_cuda import DenseStageCUDA

                    self._cuda_c = DenseStageCUDA([(self.fmt, self.m, self.codes, self.scales, table)], self.n, [0])
                y = self._cuda_c(x, out=None if out is None else out.view(-1))
                return y.view(1, self.m) if out is None else out
        return packed_gemv(x, self.codes, self.scales, self.fmt, self.m, self.n, table, expert_ids=expert_ids,
                           x_div=x_div, out=out, zero_skipped=zero_skipped)


class FastPackedLinear(nn.Module):
    def __init__(self, meta: dict, arrays, luts: dict[str, torch.Tensor], device):
        super().__init__()
        s, m, n, fmt = meta["safe"], int(meta["m"]), int(meta["n"]), meta["fmt"]
        self.in_features, self.out_features = n, m
        codes = _t(arrays[f"lin_{s}_codes"], device)
        scales = _t(arrays[f"lin_{s}_scales"], device).reshape(1, m)
        if fmt.startswith("lm"):
            self.mat = _Mat(fmt, m, n, codes, scales, levels=_t(arrays[f"lin_{s}_levels"].astype(np.float32), device))
        else:
            self.mat = _Mat(fmt, m, n, codes.reshape(1, m, -1), scales, lut=luts[fmt])
        self.signs_in = _signs(arrays[f"lin_{s}_signs_in"], device)
        self.signs_out = _signs(arrays[f"lin_{s}_signs_out"], device) if meta.get("out_rot") else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = rht_in(x.reshape(-1, shape[-1]), self.signs_in)
        y = self.mat(x2)
        if self.signs_out is not None:
            y = rht_out(y, self.signs_out)
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)


class FastPackedLMHead(nn.Module):
    def __init__(self, packed_dir: Path, luts: dict[str, torch.Tensor], device, hidden: int = 2304):
        super().__init__()
        meta = json.loads((packed_dir / "head_meta.json").read_text())
        npz = np.load(packed_dir / "head_packed.npz")
        self.mats, self.signs = [], []
        for bm in meta:
            kb, rows, fmt = bm["kb"], int(bm["rows"]), bm["opt"]
            codes = _t(npz[f"b{kb}_codes"], device)
            scales = _t(npz[f"b{kb}_scales"], device).reshape(1, rows)
            if fmt.startswith("lm"):
                mat = _Mat(fmt, rows, hidden, codes, scales, levels=_t(npz[f"b{kb}_levels"].astype(np.float32), device))
            else:
                mat = _Mat(fmt, rows, hidden, codes.reshape(1, rows, -1), scales, lut=luts[fmt])
            self.mats.append(mat)
            self.signs.append(_signs(npz[f"b{kb}_signs_in"], device))
        order = torch.from_numpy(np.asarray(npz["order"], dtype=np.int64)).to(device)
        self.inv_order = torch.empty_like(order)
        self.inv_order[order] = torch.arange(len(order), device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1])
        parts = [mat(rht_in(x, s)) for mat, s in zip(self.mats, self.signs)]
        return torch.cat(parts, dim=-1)[:, self.inv_order].reshape(*h.shape[:-1], -1)
