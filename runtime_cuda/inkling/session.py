"""One persistent cache and decode graph per server, so a request resumes where an earlier one left off.

An agent loop resends its whole history every turn: system prompt, tools, earlier replies, tool output. The
next prompt extends the last one, so most of its KV state is already on the GPU. A `CudaSession` keeps it:

- `rows` lists the tokens whose state is in the cache, in order. The global-attention KV is written once per
  position and never changed, so the rows of any prefix stay valid until a different token overwrites them.
- What later tokens do overwrite is small: the sliding-window ring buffers (which reuse cells) and the causal-conv
  states (which shift every token). A *checkpoint* copies just those to pinned host memory at each prompt boundary
  (the state after `ids[:-1]`, before the last prompt token runs). Rewinding to a checkpoint copies them back and
  moves the position; nothing proportional to the context is copied.
- The end of the last reply is also a resume point, for clients that resend the reasoning (the template then
  re-renders the reply to the same tokens). The decode graph runs `sync_every` steps between host reads, so it
  can run up to `sync_every - 1` steps past where the client stopped reading. Before each batch the conv states
  are copied to a GPU backup; if the batch overran, the next request that resumes there restores the backup and
  replays the batch's real tokens. KV written by the overrun needs no repair: global rows past the end are
  rewritten before they are read, and the ring cells it overwrote hold positions outside the window.
- A request from a different conversation overwrites the rows. With a `CudaPromptCache`, the latest checkpoint
  of the old conversation is first saved there as a full snapshot, and a later request can restore it. With a
  cache directory, every checkpoint is saved there (as upstream does), so a restarted server can resume.

The cache is allocated once at the server's context size and each decode graph is captured once (per blocked-
token set), which also saves the per-request allocation and capture.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Iterator, Sequence

import torch

from .generate import GreedyDecodeGraph
from .model import Inkling2Cache, Inkling2LoopPacked


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def _cdf(w: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """`w.cumsum(1)` for one row, bit-reproducible: CUDA's single-pass scan of a long 1-D row adds in a different
    order every run, so seeded draws near a boundary between two tokens could differ. Rows of `block` scan with
    a fixed order; the row totals are scanned the same way and added on."""
    V = w.shape[1]
    nb = -(-V // block)
    rows = torch.nn.functional.pad(w, (0, nb * block - V)).reshape(nb, block).cumsum(1)
    tot = rows[:, -1].reshape(1, nb)
    return (rows + (tot.cumsum(1) - tot).reshape(nb, 1)).reshape(1, -1)[:, :V]


class SampledDecodeGraph:
    """A captured temperature-sampling decode step: the step, the full logits, and the draw of `sample_logits`
    (inverse CDF of softmax(logits / T) at a uniform `draws[step]`, with a reproducible CDF), all on the device.
    `temp` and `draws` are inputs, so one capture serves every temperature and seed. A step whose logits are not
    finite outputs -1 (and feeds it on), for the host to raise on."""

    def __init__(self, model: Inkling2LoopPacked, cache: Inkling2Cache, max_steps: int, blocked: Sequence[int] = (),
                 scratch: Inkling2Cache | None = None):
        self.model, self.cache, self.blocked = model, cache, frozenset(int(t) for t in blocked)
        dev = model.device
        self.ban = torch.tensor(sorted(self.blocked), dtype=torch.long, device=dev) if self.blocked else None
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(max_steps + 1, dtype=torch.long, device=dev)
        self.step = torch.zeros(1, dtype=torch.long, device=dev)
        self.draws = torch.zeros(max_steps + 1, device=dev)
        self.temp = torch.ones(1, device=dev)
        scratch = scratch if scratch is not None else model.make_cache(
            cache.capacity, global_kv_mode=cache.global_kv_mode, kv_vram_cells=cache.kv_vram_cells,
            prefill_chunk=cache.prefill_chunk)
        with torch.cuda.device(dev):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self._sample(model.logits(model.decode_step(self.tok, scratch)))
            torch.cuda.current_stream().wait_stream(st)
            del scratch
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                nxt = self._sample(model.logits(model.decode_step(self.tok, cache)))
                self.tok.copy_(nxt)
                self.out.index_copy_(0, self.step, nxt)
                self.step.add_(1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        lg = logits.float()
        if self.ban is not None:
            lg = lg.index_fill(-1, self.ban, float("-inf"))
        w = _cdf(lg.sub(lg.amax(dim=1, keepdim=True)).div_(self.temp).exp_())
        total = w[:, -1]
        thr = torch.minimum(self.draws.index_select(0, self.step) * total, torch.nextafter(total, torch.zeros_like(total)))
        pick = torch.searchsorted(w, thr[:, None], right=True).flatten().clamp_(max=w.shape[1] - 1)
        return torch.where(total.isfinite(), pick, -1)

    def replay(self):
        self.graph.replay()


class _Checkpoint:
    """Pinned host copies of the state later tokens overwrite: sliding ring buffers and conv states."""

    def __init__(self, cache: Inkling2Cache):
        self.ring = {s: torch.empty(cache.kv[s].shape, dtype=cache.kv[s].dtype, pin_memory=True)
                     for s in range(len(cache.kv)) if cache.is_sliding[s]}
        self.conv = [[torch.empty(c.shape, dtype=c.dtype, pin_memory=True) for c in slot] for slot in cache.conv_state]
        self.n = -1
        self.event = None
        self.stored = False  # already in the prompt cache as a full snapshot

    def save(self, cache: Inkling2Cache, n: int):
        for s, dst in self.ring.items():
            dst.copy_(cache.kv[s], non_blocking=True)
        for slot, dsts in zip(cache.conv_state, self.conv):
            for c, dst in zip(slot, dsts):
                dst.copy_(c, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream(cache.device))
        self.n, self.stored = n, False

    def load(self, cache: Inkling2Cache):
        self.event.synchronize()  # the copy out must finish before the host buffers are read
        for s, src in self.ring.items():
            cache.kv[s].copy_(src, non_blocking=True)
        for slot, srcs in zip(cache.conv_state, self.conv):
            for c, src in zip(slot, srcs):
                c.copy_(src, non_blocking=True)
        cache.seq_len = self.n
        cache.pos.fill_(self.n)


class CudaSession:
    def __init__(self, model: Inkling2LoopPacked, capacity: int, *, prompt_cache: Any = None, sync_every: int = 8,
                 checkpoints: int = 3):
        self.model, self.sync_every, self.prompt_cache = model, int(sync_every), prompt_cache
        cap = ((int(capacity) + self.sync_every + 255) // 256) * 256
        self.cache = model.make_cache(cap)
        self.mkey = f"{model.packed_dir.resolve()}|kv={self.cache.global_kv_mode}"
        self.graphs: dict[tuple, GreedyDecodeGraph | SampledDecodeGraph] = {}
        self.rows: list[int] = []  # tokens whose state the cache holds (once any pending repair is done)
        self.n_ckpt = max(1, int(checkpoints))  # pinned buffers (~0.34 GB each), allocated on first use
        self.free: list[_Checkpoint] = []
        self.ckpts: list[_Checkpoint] = []  # valid checkpoints, oldest first, each a prefix of `rows`
        self.backup = [[c.clone() for c in slot] for slot in self.cache.conv_state]  # conv states at a batch start
        self._conv_flat = [c for slot in self.cache.conv_state for c in slot]
        self._backup_flat = [b for slot in self.backup for b in slot]
        self._pending = None
        self.repair: tuple[int, list[int]] | None = None  # (batch start, tokens to replay from it) after an overrun

    @property
    def capacity(self) -> int:
        return self.cache.capacity

    def graph(self, blocked: frozenset, sampled: bool = False):
        key = (blocked, sampled)
        if key not in self.graphs:
            cls = SampledDecodeGraph if sampled else GreedyDecodeGraph
            # warm up on the session's own cache (no second full-size cache), then drop its state
            self.reset()
            self.graphs[key] = cls(self.model, self.cache, self.cache.capacity, blocked, scratch=self.cache)
            self.reset()
        return self.graphs[key]

    # ------------------------------------------------------------------ state

    def reset(self):
        self.cache.reset()
        self.rows, self.repair = [], None
        self.free += self.ckpts
        self.ckpts = []

    def _truncate(self, n: int):
        """Forget rows past `n` and the checkpoints that need them."""
        del self.rows[n:]
        keep = [c for c in self.ckpts if c.n <= n]
        self.free += [c for c in self.ckpts if c.n > n]
        self.ckpts = keep

    def _checkpoint(self, n: int):
        if self.free:
            ck = self.free.pop()
        elif len(self.ckpts) < self.n_ckpt:
            ck = _Checkpoint(self.cache)
        else:
            ck = self.ckpts.pop(0)
        ck.save(self.cache, n)
        self.ckpts.append(ck)

    def _repair(self):
        start, toks = self.repair
        self.repair = None
        for slot, saved in zip(self.cache.conv_state, self.backup):
            for c, b in zip(slot, saved):
                c.copy_(b)
        self.cache.pos.fill_(start)
        g = next(iter(self.graphs.values()))
        g.step.zero_()
        for t in toks:  # the same graph and inputs as the first time, so the same state
            g.tok.fill_(t)
            g.replay()
        self.cache.seq_len = start + len(toks)

    def _save_for_later(self):
        """Before another conversation overwrites the rows: the latest checkpoint as a full snapshot."""
        if self.prompt_cache is None or not self.ckpts:
            return
        ck = self.ckpts[-1]
        if ck.stored or ck.n < getattr(self.prompt_cache, "min_tokens", 0):
            return
        ck.stored = True
        live = self.cache.seq_len
        self.cache.seq_len = ck.n  # the global rows of the checkpoint's prefix are still the live ones
        snap = self.cache.snapshot_cpu()
        self.cache.seq_len = live
        ck.event.synchronize()
        for s, ring in ck.ring.items():
            snap["tensors"][f"kv.{s}"] = ring[:, :, : snap["tensors"][f"kv.{s}"].shape[2]].clone()
        for s, convs in enumerate(ck.conv):
            for i, c in enumerate(convs):
                snap["tensors"][f"conv.{s}.{i}"] = c.clone()
        self.prompt_cache.store(self.rows[: ck.n], snap, model_key=self.mkey)

    def _resume(self, ids: list[int], use_cache: bool) -> int:
        """Puts the cache at the longest reusable prefix of `ids[:-1]`; returns its length."""
        limit = len(ids) - 1
        if not use_cache:
            self.reset()
            return 0
        common = _common_prefix(self.rows, ids)
        live = len(self.rows) if common == len(self.rows) and len(self.rows) <= limit else -1
        ck = max((c for c in self.ckpts if c.n <= min(common, limit)), key=lambda c: c.n, default=None)
        best = max(live, ck.n if ck is not None else 0)
        # the prompt cache serves other conversations: look only when this request leaves the current one
        newest = self.ckpts[-1].n if self.ckpts else -1
        if self.prompt_cache is not None and best < limit and (best < newest or not self.ckpts):
            if self.ckpts and self.ckpts[-1].n > best + getattr(self.prompt_cache, "min_tokens", 0):
                self._save_for_later()  # this request will overwrite a longer checkpoint
            hit = self.prompt_cache.match(ids, model_key=self.mkey)
            if hit is not None and len(hit[0]) > best and len(hit[0]) <= limit:
                saved, snap = hit
                self.reset()
                self.cache.restore_from_snapshot(snap)
                self.rows = list(saved)
                return len(saved)
        if live >= 0 and live >= best:
            if self.repair is not None:
                self._repair()
            return live
        self.repair = None
        if ck is None:
            self.reset()
            return 0
        ck.load(self.cache)
        self._truncate(ck.n)
        return ck.n

    # ------------------------------------------------------------------ generation

    def stream(self, prompt_ids: Sequence[int], *, max_tokens: int, eos_ids: Sequence[int] = (),
               blocked: Sequence[int] = (), temperature: float = 0., seed: int | None = None,
               use_cache: bool = True, stats_out: dict | None = None) -> Iterator[int]:
        """Yield generated token ids (the first from the prompt's last token). Stop reading at any point: the
        tokens read so far, except the last, become rows a later request can extend."""
        ids = [int(t) for t in prompt_ids]
        if not ids or max_tokens < 1:
            raise ValueError("empty prompt or no tokens to generate")
        if len(ids) + max_tokens + self.sync_every > self.cache.capacity:
            raise ValueError(f"prompt + max_tokens needs {len(ids) + max_tokens} positions, "
                             f"the session holds {self.cache.capacity - self.sync_every}")
        eos = {int(t) for t in eos_ids}
        blocked = frozenset(int(t) for t in blocked)
        stats = {} if stats_out is None else stats_out
        dev = self.model.device
        greedy = not temperature
        with torch.cuda.device(dev):
            g = self.graph(blocked, sampled=not greedy)

        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        read: list[int] = []  # tokens handed to the caller
        fed_end = len(ids) - 1  # positions the cache has run (host mirror of cache.pos)
        step_ms: list[float] = []
        t1 = None
        started, clean = False, True
        self._pending = None
        try:
            with torch.cuda.device(dev):  # Triton and NVRTC launch on the current device's stream
                reused = self._resume(ids, use_cache)
                stats.update(prompt_tokens=len(ids), prompt_cached_tokens=reused)
                if reused < len(ids) - 1:
                    self.model.prefill(ids[reused:-1], self.cache)
                self.rows = ids[:-1]
                if use_cache:
                    self._checkpoint(len(ids) - 1)
                    if getattr(self.prompt_cache, "cache_dir", None) is not None:
                        self._save_for_later()  # persisted every turn, so a restart can resume it
                started = True
                if not greedy:
                    generator = torch.Generator(device=dev)
                    generator.seed() if seed is None else generator.manual_seed(seed)
                    g.draws[:max_tokens].copy_(torch.rand(max_tokens, generator=generator, device=dev))
                    g.temp.fill_(float(temperature))
                g.tok.fill_(ids[-1])
                g.step.zero_()
            last_in = ids[-1]
            done = False
            while not done and len(read) < max_tokens:
                # one step first, for the time to the first token; then `sync_every` between host reads
                n = 1 if not read else min(self.sync_every, max_tokens - len(read))
                start, batch_in = fed_end, last_in
                ts = time.perf_counter()
                with torch.cuda.device(dev):
                    torch._foreach_copy_(self._backup_flat, self._conv_flat)
                    for _ in range(n):
                        g.replay()
                    new = g.out[len(read): len(read) + n].tolist()  # syncs
                if not greedy and min(new) < 0:
                    raise FloatingPointError("Invalid logits for temperature sampling")
                fed_end += n
                if t1 is None:
                    t1 = time.perf_counter()
                else:
                    step_ms += [(time.perf_counter() - ts) * 1e3 / n] * n
                inputs = [batch_in] + new[:-1]
                for i, t in enumerate(new):
                    read.append(t)
                    # if the caller stops reading here, the cache should hold inputs[: i + 1] from `start`
                    self._pending = (start, inputs[: i + 1], n)
                    yield t
                    if t in eos:
                        done = True
                        break
                last_in = new[-1]
        except GeneratorExit:  # the caller stopped reading, at a yield: the bookkeeping is exact
            raise
        except BaseException:
            clean = False
            raise
        finally:
            if clean and started and use_cache:
                self._finish(ids, read)
            else:
                self.reset()
            if t1 is not None:
                t2 = time.perf_counter()
                stats.update(
                    mode="session-cuda-graph" if greedy else "session-cuda-graph-sampled",
                    generated_tokens=len(read),
                    prompt_s=round(t1 - t0, 4),
                    prefill_tok_s=round((len(ids) - reused) / max(t1 - t0, 1e-9), 1),
                    decode_tok_s=round((len(read) - 1) / max(t2 - t1, 1e-9), 2) if len(read) > 1 else 0.0,
                    median_step_ms=round(statistics.median(step_ms), 2) if step_ms else 0.0,
                    peak_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 3),
                )

    def _finish(self, ids: list[int], read: list[int]):
        """Record which tokens the cache holds after the caller stopped reading."""
        self.repair = None
        pend, self._pending = self._pending, None
        if pend is None:  # no token was read (only on an early close)
            self.reset()
            return
        start, inputs, n = pend
        if len(inputs) < n:  # the batch ran past the last token read
            self.repair = (start, inputs)
        self.rows = ids + read[:-1]
        self.cache.seq_len = start + len(inputs)
        assert self.cache.seq_len == len(self.rows), (self.cache.seq_len, len(self.rows))
