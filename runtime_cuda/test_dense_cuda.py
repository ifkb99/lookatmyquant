"""The experimental CUDA C stage kernel (`dense_cuda.py`) against the reference decoders."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling.formats import FMT_PARAMS, decode_lm, decode_trellis  # noqa: E402
from runtime_cuda.test_kernels import _rand_trellis  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")
FMTS = list(FMT_PARAMS) + ["lm6", "lm8"]


def _rand_mat(rng, fmt, m, n):
    """(fmt, m, codes, scales, table) and the float64 dense matrix."""
    if fmt.startswith("lm"):
        k = int(fmt[2:])
        idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
        if k == 8:
            packed = idx.reshape(-1)
        else:  # lm6: 4 indices in 3 bytes, little-endian
            a = idx.reshape(-1, 4).astype(np.uint32)
            v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
            packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
        codes = torch.from_numpy(packed).to(DEV)
        scales = torch.from_numpy(rng.uniform(0.01, 0.05, size=(1, m)).astype(np.float16)).to(DEV)
        levels = torch.from_numpy(np.sort(rng.standard_normal(1 << k)).astype(np.float32)).to(DEV)
        return (fmt, m, codes, scales, levels), decode_lm(codes, scales[0], levels, m, n, k).double()
    codes, scales, lut = _rand_trellis(rng, 1, m, n, fmt)
    return (fmt, m, codes, scales, lut), decode_trellis(codes[0], scales[0], lut, n, fmt).double()


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("m,n", [(200, 2304), (3072, 2304), (2304, 1152)])
def test_dense_cuda_one(fmt, m, n):
    from runtime_cuda.inkling.dense_cuda import DenseStageCUDA

    rng = np.random.default_rng(m + n + len(fmt))
    spec, W = _rand_mat(rng, fmt, m, n)
    x = torch.randn(1, n, device=DEV)
    y = torch.full((m,), float("nan"), device=DEV)
    DenseStageCUDA([spec], n, [0])(x, out=y)
    want = W @ x[0].double()
    assert ((y.double() - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("specs,x_rows", [
    ([("rtr4", 3072), ("lm6", 1024), ("lm8", 1024), ("ht2", 384), ("ht3", 3072)], [0, 1, 2, 3, 4]),
    ([("lm8", 1024), ("lm8", 3072), ("lm8", 4096), ("lm6", 8192), ("lm6", 1000), ("rtr4", 3000), ("rtr3", 700),
      ("rtr2", 900)], list(range(8))),  # LM-head-like
    ([("rtr1", 64), ("ht1", 128), ("lm6", 16)], [0, 0, 0]),  # one shared input row, tiny matrices
])
def test_dense_cuda_stage(specs, x_rows):
    from runtime_cuda.inkling.dense_cuda import DenseStageCUDA

    rng = np.random.default_rng(len(specs))
    n = 2304
    mats, Ws = zip(*[_rand_mat(rng, f, m, n) for f, m in specs])
    x = torch.randn(max(x_rows) + 1, n, device=DEV)
    st = DenseStageCUDA(list(mats), n, x_rows)
    for i, y in enumerate(st.split(st(x))):
        want = Ws[i] @ x[x_rows[i]].double()
        assert ((y.double() - want).norm() / want.norm()).item() < 1e-5, i


def test_mat_gemv_uses_cuda_c(monkeypatch):
    """`_Mat.gemv` for one token row takes the CUDA C path with the switch on, and matches Triton's."""
    from runtime_cuda.inkling.packed_modules import _Mat

    rng = np.random.default_rng(3)
    n = 2304
    for fmt in ("rtr4", "lm6"):
        (f, m, codes, scales, table), W = _rand_mat(rng, fmt, 1000, n)
        mat = _Mat(f, m, n, codes, scales, **({"levels": table} if f.startswith("lm") else {"lut": table}))
        x = torch.randn(1, n, device=DEV)
        monkeypatch.setenv("LAMQ_CUDA_C", "0")
        a = mat.gemv(x)
        monkeypatch.setenv("LAMQ_CUDA_C", "1")
        b = mat.gemv(x)
        assert mat._cuda_c is not None and b.shape == a.shape == (1, m)
        assert ((b - a).norm() / a.norm()).item() < 1e-5
