"""`CudaSession`: a resumed request sees exactly the state the cache had when it first reached that position.

Compared bit for bit with `snapshot_cpu()`, not by KL: a cold prefill splits the prompt differently, so its state
differs at the 1e-7 level and routing near-ties amplify that; a correct resume involves no new arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACK = ROOT / "models" / "Inkling-2-Mini-LookAtMyQuant-8.9GB"
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and PACK.exists()), reason="needs CUDA and the 8.9 GB pack")


@pytest.fixture(scope="module")
def session():
    from runtime_cuda.inkling.model import Inkling2LoopPacked
    from runtime_cuda.inkling.session import CudaSession
    from runtime_mlx.prompt_cache_disk import CudaPromptCache

    model = Inkling2LoopPacked(PACK, "cuda")
    return CudaSession(model, 6144, prompt_cache=CudaPromptCache(max_entries=2, min_tokens=0))


def _ids(seed, n):
    return torch.randint(1000, 150000, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _run(session, ids, max_tokens, read=None, **kw):
    """Generate, reading `read` tokens (all by default) before closing."""
    out = []
    gen = session.stream(ids, max_tokens=max_tokens, **kw)
    for t in gen:
        out.append(t)
        if read is not None and len(out) == read:
            break
    gen.close()
    return out


def _state(session):
    """The state a later token can read: global rows below the position (all `snapshot_cpu` saves) and, for
    sliding ring buffers, the cells of the last `window` positions (other cells are stale or out of reach)."""
    c = session.cache
    snap = c.snapshot_cpu()
    t, L, W = snap["tensors"], snap["meta"]["seq_len"], c.cfg.sliding_window_size
    for s in range(len(c.kv)):
        if c.is_sliding[s]:
            cap = c.kv[s].shape[2]
            cells = torch.tensor(sorted({p % cap for p in range(max(0, L - W), L)}), dtype=torch.long)
            t[f"kv.{s}"] = t[f"kv.{s}"][:, :, cells]
    return t


def _same(a, b):
    assert a.keys() == b.keys()
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, bad[:5]


def _capture_at_checkpoint(session, monkeypatch):
    """The cache's full state each time a checkpoint is taken."""
    seen = []
    orig = session._checkpoint

    def hook(n):
        seen.append((n, _state(session)))
        orig(n)
    monkeypatch.setattr(session, "_checkpoint", hook)
    return seen


def _capture_after_resume(session, monkeypatch):
    seen = []
    orig = session._resume

    def hook(ids, use_cache):
        r = orig(ids, use_cache)
        seen.append((r, _state(session)))
        return r
    monkeypatch.setattr(session, "_resume", hook)
    return seen


def test_checkpoint_rewind_is_exact(session, monkeypatch):
    session.reset()
    p = _ids(1, 700)
    at_ckpt = _capture_at_checkpoint(session, monkeypatch)
    first = _run(session, p, 40)  # a long reply, abandoned by the next request
    after = _capture_after_resume(session, monkeypatch)
    _run(session, p + _ids(2, 50), 4)
    assert after[0][0] == len(p) - 1
    _same(at_ckpt[0][1], after[0][1])
    # a retry of the same prompt resumes at the same checkpoint and generates the same tokens
    assert _run(session, p, 40) == first


@pytest.mark.parametrize("temp,n", [(0.0, 300), (0.8, 300), (0.0, 2100)])
def test_overrun_repair_is_exact(session, temp, n):
    p = _ids(3, n)  # 2100: the sliding rings (2048 cells) have wrapped
    k = 1 + 8 + 3  # stop 3 tokens into the third batch: the graph ran 5 steps further
    kw = dict(temperature=temp, seed=5)
    session.reset()
    ref_tokens = _run(session, p, k, **kw)  # max_tokens = k: the last batch has 3 steps, no overrun
    assert session.repair is None
    ref = _state(session)
    session.reset()
    assert _run(session, p, 100, read=k, **kw) == ref_tokens
    assert session.repair is not None
    nxt = p + ref_tokens[:-1] + _ids(4, 20)
    assert session._resume(nxt, True) == len(p) + k - 1
    _same(ref, _state(session))


def test_rewind_past_ring_wrap_is_exact(session, monkeypatch):
    """A rewind over more than `capacity - window` abandoned tokens needs the ring cells the checkpoint saved."""
    session.reset()
    p = _ids(13, 2500)
    at_ckpt = _capture_at_checkpoint(session, monkeypatch)
    _run(session, p, 4)
    _run(session, p + _ids(14, 1200), 4)  # overwrites every ring cell
    after = _capture_after_resume(session, monkeypatch)
    _run(session, p + _ids(15, 30), 4)
    assert after[0][0] == len(p) - 1
    _same(at_ckpt[0][1], after[0][1])


def test_shorter_continuation_ignores_stale_rows(session):
    """Turn 2 after a long abandoned reply = turn 2 after a short one."""
    p, q = _ids(5, 500), _ids(6, 30)
    session.reset()
    _run(session, p, 60)
    long_ = _run(session, p + q, 12)
    session.reset()
    _run(session, p, 2)
    assert _run(session, p + q, 12) == long_


@pytest.mark.parametrize("na,nb", [(600, 400), (2300, 2200)])
def test_other_conversation_round_trip(session, monkeypatch, na, nb):
    """A, then B (A's checkpoint goes to the prompt cache), then A's next turn restores it exactly."""
    session.reset()
    session.prompt_cache.clear()
    a, b = _ids(7, na), _ids(8, nb)
    at_ckpt = _capture_at_checkpoint(session, monkeypatch)
    _run(session, a, 6)
    _run(session, b, 6)
    after = _capture_after_resume(session, monkeypatch)
    _run(session, a + _ids(9, 10), 4)
    assert after[0][0] == len(a) - 1
    _same(at_ckpt[0][1], after[0][1])


def test_sampled_reply_end_resume(session):
    """Temperature sampling: seeded draws repeat, and the cache holds the prompt and the reply read but its last."""
    session.reset()
    p = _ids(10, 300)
    out = _run(session, p, 30, read=10, temperature=0.7, seed=1)
    assert session.rows == p + out[:-1] and session.cache.seq_len == len(p) + 9
    assert session._resume(p + out + _ids(11, 5), True) == len(p) + 9
    assert _run(session, p, 10, temperature=0.7, seed=1) == out
    assert _run(session, p, 10, temperature=0.7, seed=2) != out


def test_no_cache_leaves_nothing(session):
    session.reset()
    p = _ids(12, 300)
    _run(session, p, 5, use_cache=False)
    assert session.rows == [] and not session.ckpts
    assert session._resume(p + [5], True) == 0


def test_graph_sampler_matches_sample_logits():
    """The in-graph draw picks the token the eager `sample_logits` picks for the same logits and uniform."""
    from types import SimpleNamespace

    from runtime_cuda.inkling.session import SampledDecodeGraph
    from runtime_cuda.inkling_cuda.drafter import sample_logits

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    for temp in (0.3, 1.0):
        draws = torch.rand(64, device=dev, generator=g)
        lg = torch.randn(1, 50000, device=dev, generator=g) * 4
        lg[0, 11] = 40.0  # would be drawn almost always, were it not blocked
        fake = SimpleNamespace(ban=torch.tensor([11], device=dev), temp=torch.tensor([temp], device=dev), draws=draws,
                               step=torch.zeros(1, dtype=torch.long, device=dev))
        differ = 0
        for i in range(64):
            fake.step.fill_(i)
            got, want = int(SampledDecodeGraph._sample(fake, lg)), sample_logits(lg.clone(), draws[i:i + 1], temp, 11)[0]
            assert got == SampledDecodeGraph._sample(fake, lg).item()  # reproducible
            assert abs(got - want) <= 1 and got != 11  # the CDFs round differently: at most a neighbour at a boundary
            differ += got != want
        assert differ <= 2
