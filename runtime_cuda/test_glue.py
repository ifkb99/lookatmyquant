"""Fork-specific glue kernels against PyTorch: the fused router and the per-query attention residual."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")


@pytest.mark.parametrize("T", [1, 2, 5, 8, 16])
def test_route_top8_fused(T):
    from runtime_cuda.inkling.fused import route_top8_fused

    g = torch.Generator(device=DEV).manual_seed(T)
    h = torch.randn(T, 2304, device=DEV, generator=g)
    W = torch.randn(128, 2304, device=DEV, generator=g) * 0.02
    b = torch.randn(128, device=DEV, generator=g) * 0.1
    sc = torch.sigmoid(h @ W.T)
    ref = torch.topk(sc + b, 8, dim=-1).indices
    rw = sc.gather(1, ref)
    rw = rw / rw.sum(-1, keepdim=True) * 2.5
    idx, w = route_top8_fused(h, W, b, 8, 2.5)
    for t in range(T):  # same experts and weights, in any order
        got = dict(zip(idx[t].tolist(), w[t].tolist()))
        want = dict(zip(ref[t].tolist(), rw[t].tolist()))
        assert got.keys() == want.keys()
        assert max(abs(got[e] - want[e]) for e in want) < 1e-6


@pytest.mark.parametrize("NQ,NS,norm", [(1, 2, True), (3, 5, True), (4, 7, False), (2, 4, True)])
def test_block_attn_res_per_query(NQ, NS, norm):
    from runtime_cuda.inkling.fused import block_attn_res_fused

    g = torch.Generator(device=DEV).manual_seed(NQ * 10 + NS)
    T, H = 3, 2304
    q = torch.randn(NQ, H, device=DEV, generator=g)
    states = [torch.randn(T, H, device=DEV, generator=g) for _ in range(NS)]
    lnw = torch.randn(NQ, H, device=DEV, generator=g) if norm else None
    got = block_attn_res_fused(q, states, lnw)
    X = torch.stack(states, 1)  # [T, NS, H]
    Xn = X * torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
    a = torch.softmax(torch.einsum("qh,tsh->qts", q, Xn) / H ** 0.5, -1)
    want = torch.einsum("qts,tsh->qth", a, Xn)
    if norm:
        want = want * torch.rsqrt(want.square().mean(-1, keepdim=True) + 1e-6) * lnw[:, None]
    assert ((got - want).norm() / want.norm()).item() < 1e-5
