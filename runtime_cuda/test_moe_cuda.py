"""The experimental CUDA C MoE decode kernel (`moe_cuda.py`, NVRTC-compiled) against the reference decoders."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling.formats import FMT_PARAMS, decode_trellis  # noqa: E402
from runtime_cuda.test_kernels import _rand_trellis  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")


def _side(rng, fmts, m, n, E):
    owner = rng.integers(0, len(fmts), size=E)
    owner[: len(fmts)] = np.arange(len(fmts))  # every slab owns an expert
    slabs, W = [], {}
    slab_of, loc_of = np.zeros(E, np.int32), np.zeros(E, np.int32)
    for j, fmt in enumerate(fmts):
        ids = np.nonzero(owner == j)[0]
        codes, scales, lut = _rand_trellis(rng, len(ids), m, n, fmt)
        slabs.append((fmt, codes, scales, lut))
        Wf = decode_trellis(codes, scales, lut, n, fmt)
        for loc, e in enumerate(ids):
            slab_of[e], loc_of[e], W[int(e)] = j, loc, Wf[loc].double()
    return slabs, torch.from_numpy(slab_of).to(DEV), torch.from_numpy(loc_of).to(DEV), W


@pytest.mark.parametrize("eid_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("fmts", [[f] for f in FMT_PARAMS] + [["ht2", "ht3", "rtr4", "rtr1"], ["ht3", "rtr4"]])
@pytest.mark.parametrize("m,n,x_div", [(2304, 2304, 8), (2304, 1152, 1), (200, 1152, 8)])
def test_moe_side_cuda(fmts, m, n, x_div, eid_dtype):
    from runtime_cuda.inkling.moe_cuda import MoESideCUDA

    rng = np.random.default_rng(len(fmts) * 1000 + m + n)
    E = 12
    slabs, slab_of, loc_of, W = _side(rng, fmts, m, n, E)
    side = MoESideCUDA(slabs, slab_of, loc_of, m, n)
    eids = torch.from_numpy(rng.integers(0, E, size=16)).to(DEV)  # int64, as the router returns them
    eids[: len(fmts)] = torch.arange(len(fmts), device=DEV)  # every format runs
    eids = eids.to(eid_dtype)
    x = torch.randn(16 // x_div, n, device=DEV)
    out = torch.full((16, m), float("nan"), device=DEV)
    side(x, eids, x_div, out=out)
    want = torch.stack([x[i // x_div].double() @ W[int(e)].T for i, e in enumerate(eids.tolist())])
    assert ((out.double() - want).norm() / want.norm()).item() < 1e-5


def test_moe_side_cuda_graph():
    """Captured once, replayed with a new activation and new expert ids into a NaN-filled output."""
    from runtime_cuda.inkling.moe_cuda import MoESideCUDA

    rng = np.random.default_rng(5)
    m, n, E = 2304, 1152, 12
    slabs, slab_of, loc_of, W = _side(rng, ["ht2", "ht3"], m, n, E)
    side = MoESideCUDA(slabs, slab_of, loc_of, m, n)
    x = torch.randn(8, n, device=DEV)
    eids = torch.zeros(8, dtype=torch.long, device=DEV)
    out = torch.full((8, m), float("nan"), device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        side(x, eids, 1, out=out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        side(x, eids, 1, out=out)
    x.copy_(torch.randn_like(x))
    eids.copy_(torch.from_numpy(rng.integers(0, E, size=8)))
    out.fill_(float("nan"))
    g.replay()
    want = torch.stack([x[i].double() @ W[int(e)].T for i, e in enumerate(eids.tolist())])
    assert ((out.double() - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("trellis", [[], ["ht2"], ["ht2", "ht3", "rtr4"]])
@pytest.mark.parametrize("m,n,x_div", [(2304, 2304, 8), (2304, 1152, 1)])
def test_moe_side_cuda_nvfp4(trellis, m, n, x_div):
    """NVFP4 experts (slab 4, the unrotated activation) alone or mixed with trellis slabs in one launch."""
    from runtime_cuda.inkling.formats import decode_nvfp4
    from runtime_cuda.inkling.moe_cuda import NV_SLAB, MoESideCUDA
    from runtime_cuda.test_kernels import _rand_nvfp4

    rng = np.random.default_rng(77 + len(trellis) + n)
    E = 12
    if trellis:
        slabs, slab_of, loc_of, W = _side(rng, trellis, m, n, E)
    else:
        slabs, slab_of, loc_of, W = [], torch.zeros(E, dtype=torch.int32, device=DEV), torch.zeros(E, dtype=torch.int32, device=DEV), {}
    nv = [e for e in range(E) if e % 2 == 1 or not trellis]  # every other expert is NVFP4
    nc, ns, no = _rand_nvfp4(rng, len(nv), m, n)
    Wnv = decode_nvfp4(nc, ns, no).double()
    for loc, e in enumerate(nv):
        slab_of[e], loc_of[e], W[e] = NV_SLAB, loc, Wnv[loc]
    side = MoESideCUDA(slabs, slab_of, loc_of, m, n, nvfp4=(nc, ns, no))
    eids = torch.from_numpy(rng.integers(0, E, size=16)).to(DEV)
    eids[:4] = torch.arange(4, device=DEV)  # trellis and NVFP4 experts both run
    x_rot, x_raw = torch.randn(16 // x_div, n, device=DEV), torch.randn(16 // x_div, n, device=DEV)
    out = torch.full((16, m), float("nan"), device=DEV)
    side(x_rot, eids, x_div, out=out, x_raw=x_raw)
    want = torch.stack([(x_raw if int(slab_of[e]) == NV_SLAB else x_rot)[i // x_div].double() @ W[int(e)].T
                        for i, e in enumerate(eids.tolist())])
    assert ((out.double() - want).norm() / want.norm()).item() < 1e-5
