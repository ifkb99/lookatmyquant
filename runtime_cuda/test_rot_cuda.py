"""The CUDA C decode rotation (`rot_cuda.py`) against a float64 randomized Hadamard transform."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling import nvrtc  # noqa: E402
from runtime_cuda.inkling.formats import _factors  # noqa: E402

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and nvrtc.enabled()), reason="needs CUDA and NVRTC")
DEV = torch.device("cuda")


def _ref(x, signs, out):
    """x [S, T, n], signs [S, n] (or [1, n]) in float64."""
    S, T, n = x.shape
    A, B = _factors(n, torch.device("cpu"), torch.float64)
    a, b = A.shape[0], B.shape[0]
    xc, sc = x.double().cpu(), signs.double().cpu()[:, None, :]
    if out:
        return (A.T @ xc.reshape(S, T, a, b) @ B).reshape(S, T, n) * sc / math.sqrt(n)
    return (A @ (xc * sc).reshape(S, T, a, b) @ B.T).reshape(S, T, n) / math.sqrt(n)


# every decode shape of Inkling-2-Mini (in: attention inputs, o_proj, MoE sides; out: the stage outputs) and one
# per-row sign layout with T > 1
@pytest.mark.parametrize("S,T,n,out", [(5, 1, 2304, False), (1, 1, 3072, False), (1, 1, 2304, False), (8, 1, 1152, False),
                                       (2, 1, 1024, True), (1, 1, 2304, True), (1, 1, 384, True), (2, 1, 3072, True),
                                       (8, 1, 2304, False), (2, 3, 1152, True)])
def test_rotation_matches_float64(S, T, n, out):
    from runtime_cuda.inkling.rot_cuda import rotation

    g = torch.Generator(device=DEV).manual_seed(n + S + T)
    x = torch.randn(S, T, n, device=DEV, generator=g)
    signs = torch.randn(S, n, device=DEV, generator=g).sign() * (1 + 0.01 * torch.randn(S, n, device=DEV, generator=g))
    y = rotation(n, out, DEV)(x.reshape(S * T, n), signs, T).reshape(S, T, n)
    ref = _ref(x, signs, out)
    rel = float((y.double().cpu() - ref).norm() / ref.norm())
    assert rel < 5e-7, rel


def test_rht_uses_it_for_decode_and_shared_signs():
    """`rht` routes few-row calls here; one sign vector shared by all rows (the MoE down side's 8 slots)."""
    from runtime_cuda.inkling.rht import rht

    x = torch.randn(8, 1152, device=DEV)
    signs = torch.randn(1152, device=DEV).sign()
    y = rht(x, signs, False)
    ref = _ref(x[None], signs[None], False)[0]
    assert float((y.double().cpu() - ref).norm() / ref.norm()) < 5e-7


def test_fused_gate_prologue():
    """pro 1: rotate attn * sigmoid(gate) (gate_mul before o_proj)."""
    from runtime_cuda.inkling.rot_cuda import rotation

    n = 3072
    attn, gate, signs = torch.randn(1, n, device=DEV), torch.randn(1, n, device=DEV) * 3, torch.randn(n, device=DEV).sign()
    y = rotation(n, False, DEV, pro=1)(attn, signs[None], 1, aux=gate)
    ref = _ref((attn * torch.sigmoid(gate))[None], signs[None], False)[0]
    assert float((y.double().cpu() - ref).norm() / ref.norm()) < 5e-7


def test_fused_swiglu_prologue():
    """pro 2: rotate the softcapped SwiGLU of each [gate | up] row, and store the activation (MoE down side)."""
    from runtime_cuda.inkling.model import softcapped_swiglu
    from runtime_cuda.inkling.rot_cuda import rotation

    n, caps = 1152, (12.0, 30.0)
    gu, signs = torch.randn(8, 2 * n, device=DEV) * 4, torch.randn(n, device=DEV).sign()
    act = torch.empty(8, n, device=DEV)
    y = rotation(n, False, DEV, pro=2)(gu, signs[None], 8, caps=caps, act=act)
    want = softcapped_swiglu(gu, *caps)
    assert torch.allclose(act, want, rtol=1e-6, atol=1e-6)
    ref = _ref(want[None], signs[None], False)[0]
    assert float((y.double().cpu() - ref).norm() / ref.norm()) < 5e-7


@pytest.mark.parametrize("with_resid", [True, False])
def test_fused_conv_residual_epilogue(with_resid):
    """epi 1: the output rotation, then conv4_ (state updated in place) and the residual add (o_proj's output)."""
    from runtime_cuda.inkling.conv import conv4_
    from runtime_cuda.inkling.rot_cuda import rotation

    n = 2304
    x, signs = torch.randn(1, n, device=DEV), torch.randn(n, device=DEV).sign()
    prev, wc, resid = torch.randn(3, n, device=DEV), torch.randn(4, n, device=DEV) * 0.5, torch.randn(n, device=DEV)
    prev_ref = prev.clone()
    want = conv4_(prev_ref, torch.from_numpy(_ref(x[None], signs[None], True)[0].numpy()).float().to(DEV), wc)
    if with_resid:
        want = resid + want
    y = rotation(n, True, DEV, epi=1)(x, signs[None], 1, prev=prev, wc=wc, resid=resid if with_resid else None)
    assert torch.allclose(y, want, rtol=1e-5, atol=1e-6), float((y - want).abs().max())
    assert torch.allclose(prev, prev_ref, rtol=1e-5, atol=1e-6)


def test_combine_conv_add():
    """The MoE combine, conv4_ (state updated in place) and the residual add of one decode token."""
    from runtime_cuda.inkling.conv import conv4_
    from runtime_cuda.inkling.rot_cuda import combine_conv_add

    K, C = 8, 2304
    dn, w = torch.randn(K, C, device=DEV), torch.rand(1, K, device=DEV)
    prev, wc, resid = torch.randn(3, C, device=DEV), torch.randn(4, C, device=DEV) * 0.5, torch.randn(1, C, device=DEV)
    prev_ref = prev.clone()
    want = resid + conv4_(prev_ref, (dn * w[0, :, None]).sum(0, keepdim=True), wc)
    got = combine_conv_add(dn, w, prev, wc, resid.reshape(-1))
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6) and torch.allclose(prev, prev_ref, rtol=1e-5, atol=1e-6)


def test_stage_rotations_with_dot():
    """One launch: output rotations of rows of different sizes, read in place from one buffer, plus a dot job."""
    from runtime_cuda.inkling.rot_cuda import StageRotations

    sizes = [3072, 1024, 1024, 384]
    buf = torch.randn(sum(sizes) + 2304, device=DEV)  # a stage's concatenated output, one unrotated row at the end
    offs = [sum(sizes[:i]) for i in range(len(sizes))]
    signs = [torch.randn(n, device=DEV).sign() for n in sizes]
    qx, sw = torch.randn(2304, device=DEV), torch.randn(24, 2304, device=DEV)
    sr = StageRotations(list(zip(sizes, signs)), DEV, dot_rows=24, dot_n=2304)
    outs, dots = sr([buf[o:o + n] for o, n in zip(offs, sizes)], qx, sw)
    for o, n, sg, y in zip(offs, sizes, signs, outs):
        ref = _ref(buf[o:o + n].reshape(1, 1, n), sg[None], True)[0]
        assert float((y.double().cpu() - ref).norm() / ref.norm()) < 5e-7, n
    assert torch.allclose(dots[0], sw @ qx, rtol=1e-5, atol=1e-5)
