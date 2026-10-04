"""S1-HOST: a shared round's host-side tables built off the GPU's critical path (TF_MULTI_HOST_AHEAD, default on).

A replayed shared round leaves the GPU idle while the host builds the next launch's tables (DeltaNet tables, attention
step tables, the fold's pointer table) from Python lists, one tensor view and one pointer read at a time, and pins a
fresh host buffer for every table. Here the same tables come from numpy: every pointer is the state tensor's base
address plus its stride offset (what ``tensor[i].data_ptr()`` returns), the chains' GDN schedules are kept per window
shape, and the tables go to the GPU from double-buffered pinned host buffers (an event per buffer: a buffer is
rewritten only once the copy out of it has run). The bytes copied are the ones the list-based code copies
(tests/test_flashnext_hostprep.py checks every table both ways). TF_MULTI_HOST_AHEAD=0: the list-based code.

``Times`` measures, on the host, the stretches where a shared round's GPU waits for it (round start -> main launch,
draft pick read-back -> next MTP launch) for the done line's graph summary.
"""

from __future__ import annotations

import os
import time
import weakref
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch

ENV = "TF_MULTI_HOST_AHEAD"
ON = os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")

_TORCH_NP = {torch.int32: np.int32, torch.int64: np.int64}


# --- pinned staging: two host buffers a call site, each reused only after its copy ran -----------------------------

class _Slot:
    __slots__ = ("host", "event")

    def __init__(self) -> None:
        self.host: torch.Tensor | None = None
        self.event: torch.cuda.Event | None = None


class _Ring:
    __slots__ = ("slots", "at")

    def __init__(self) -> None:
        self.slots, self.at = (_Slot(), _Slot()), 0


_RINGS: dict = {}


def put(values: np.ndarray, dtype: torch.dtype, dev, out: torch.Tensor | None, site: str) -> torch.Tensor:
    """``values`` on the device: written into ``out`` (a graph's persistent table) or a new tensor, by one
    non-blocking copy from this site's next pinned buffer (the list-based path's ``torch.tensor(...).pin_memory()``
    copy with the same bytes)."""

    arr = np.ascontiguousarray(values, dtype=_TORCH_NP[dtype])
    n = arr.size
    dev = torch.device(dev)
    if dev.type != "cuda":                              # host tests: no pinning, no events
        t = torch.from_numpy(arr.copy())
        if out is None:
            return t
        out.copy_(t)
        return out
    ring = _RINGS.get((site, dtype, dev))
    if ring is None:
        ring = _RINGS[(site, dtype, dev)] = _Ring()
    slot = ring.slots[ring.at]
    ring.at ^= 1
    if slot.event is not None:
        slot.event.synchronize()                        # the copy out of this buffer two uses ago has run
    if slot.host is None or slot.host.numel() < n:
        slot.host = torch.empty((max(n, 256) * 2,), dtype=dtype, pin_memory=True)
    host = slot.host[:n]
    host.numpy()[:] = arr
    if out is None:
        out = torch.empty((n,), dtype=dtype, device=dev)
        out.copy_(host, non_blocking=True)
    else:
        out.copy_(host, non_blocking=True)
    if slot.event is None:
        slot.event = torch.cuda.Event()
    slot.event.record(torch.cuda.current_stream(dev))
    return out


# --- pointers: base address + stride offset (== the view's data_ptr) ---------------------------------------------

def _rows_ptrs(t: torch.Tensor, index: np.ndarray, dim: int = 0) -> np.ndarray:
    """``t.select(dim, i).data_ptr()`` for every ``i`` in ``index``."""

    return t.data_ptr() + index.astype(np.int64) * (t.stride(dim) * t.element_size())


def gdn_ptrs(segs: Sequence, lin: int) -> np.ndarray:
    """gdn_multi.Tables' pointer table [2, lin, n]: each stream's conv state and current recurrent state a layer."""

    if not ON:
        return gdn_ptrs_ref(segs, lin)
    n = len(segs)
    ptrs = np.empty((2, lin, n), dtype=np.int64)
    layers = np.arange(lin, dtype=np.int64)
    for s, (st, _, _) in enumerate(segs):
        rec = st.rec
        ptrs[0, :, s] = _rows_ptrs(st.conv, layers)
        cur = np.asarray(st.cur[:lin], dtype=np.int64)
        ptrs[1, :, s] = rec.data_ptr() + (cur * rec.stride(0) + layers * rec.stride(1)) * rec.element_size()
    return ptrs


def gdn_ptrs_ref(segs: Sequence, lin: int) -> np.ndarray:
    """The list-based table (v8.8): one view and one data_ptr a stream and layer."""

    n = len(segs)
    ptrs = np.empty((2, lin, n), dtype=np.int64)
    for s, (st, _, _) in enumerate(segs):
        for li in range(lin):
            ptrs[0, li, s] = st.conv[li].data_ptr()
            ptrs[1, li, s] = st.rec[st.cur[li], li].data_ptr()
    return ptrs


def gdn_rows(segs: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """gdn_multi.Tables' rows' stream ids [rows] and conv taps [rows, 4] (< 3: a conv state row, else a window row)."""

    rows = segs[-1][2]
    if not ON:
        taps = np.arange(4)[None, :]
        win = np.empty((rows, 4), dtype=np.int32)
        sid = np.empty((rows,), dtype=np.int32)
        for s, (_, a0, a1) in enumerate(segs):
            j = np.arange(a1 - a0)[:, None] + taps
            win[a0:a1] = np.where(j < 3, j, a0 + j)
            sid[a0:a1] = s
        return sid, win
    sid, first = _sid_first(segs)
    j = (np.arange(rows) - first)[:, None] + np.arange(4)[None, :]
    win = np.where(j < 3, j, first[:, None] + j).astype(np.int32)
    return sid, win


def _sid_first(segs: Sequence) -> tuple[np.ndarray, np.ndarray]:
    a0s = np.fromiter((a0 for _, a0, _ in segs), dtype=np.int64, count=len(segs))
    counts = np.fromiter((a1 - a0 for _, a0, a1 in segs), dtype=np.int64, count=len(segs))
    sid = np.repeat(np.arange(len(segs), dtype=np.int32), counts)
    return sid, a0s[sid]


_PLANS: OrderedDict = OrderedDict()


def plan_host(lengths: Sequence[int]):
    """kernels.gdn.plan_host for chains of these lengths (each row's parent the row before), kept per shape."""

    from tensorfold.cuda.kernels import gdn as shared

    key = tuple(lengths)
    if not ON:
        return shared.plan_host([list(range(-1, n - 1)) for n in key])
    got = _PLANS.get(key)
    if got is None:
        entries, starts, slots, most = shared.plan_host([list(range(-1, n - 1)) for n in key])
        got = _PLANS[key] = (np.asarray(entries + starts, dtype=np.int32), slots, most)
        if len(_PLANS) > 4096:
            _PLANS.popitem(last=False)
    else:
        _PLANS.move_to_end(key)
    return got


ATTN_FIELDS = ("k", "v", "ks", "vs")


def attn_ptrs(segs: Sequence, layers: Sequence, mtp: bool, width: int) -> np.ndarray:
    """attn_multi.Step's pointer table [layers, width, n]: keys, values, key/value scales, index keys, pooled."""

    n = len(segs)
    ptrs = np.empty((len(layers), width, n), np.int64)
    if not ON:
        for s, (st, _, _) in enumerate(segs):
            for i, layer in enumerate(layers):
                if mtp:
                    kc, ikc, pooled = st.mtp_kc, st.mtp_ikc, st.mtp_pooled
                else:
                    a = st.att_index[layer.index]
                    kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
                ptrs[i, :, s] = [kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(),
                                 ikc.data_ptr(), pooled.data_ptr()]
        return ptrs
    for s, (st, _, _) in enumerate(segs):
        ptrs[:, :, s] = _state_attn(st, layers, mtp)
    return ptrs


_ATTN: dict = {}


def _state_attn(st, layers: Sequence, mtp: bool) -> np.ndarray:
    """A state's [layers, 6] attention pointers, kept while its caches stay where they are (State.version counts
    reallocations; the key holds the cache objects' ids next to it, and the entry is dropped with the state)."""

    key = (id(st), mtp, tuple(layer.index for layer in layers))
    if mtp:
        objs = (st.mtp_kc, st.mtp_ikc, st.mtp_pooled)
    else:
        objs = (st.kc, st.ikc, st.pooled)
    version = getattr(st, "version", None)
    stamp = (version, tuple(id(o) for o in objs))
    got = _ATTN.get(key) if version is not None else None
    if got is not None and got[0] == stamp and got[2]() is st:
        return got[1]
    rows = []
    for layer in layers:
        if mtp:
            kc, ikc, pooled = st.mtp_kc, st.mtp_ikc, st.mtp_pooled
        else:
            a = st.att_index[layer.index]
            kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
        rows.append([kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(),
                     ikc.data_ptr(), pooled.data_ptr()])
    arr = np.asarray(rows, dtype=np.int64)
    if version is None:
        return arr
    try:
        ref = weakref.ref(st, lambda _, k=key: _ATTN.pop(k, None))
    except TypeError:                                   # not weak-referenceable: no caching
        return arr
    _ATTN[key] = (stamp, arr, ref)
    return arr


def fold_ptrs(sc, states: Sequence) -> list[int] | np.ndarray:
    """kernels.gdn.replay_table for multi_graphs.fold: k, v, g, beta of each layer (the scratch's other parity),
    then each state's current recurrent state a layer."""

    from tensorfold.cuda.kernels import gdn

    parity = 1 - sc.parity
    lin = sc.lin
    if not ON:
        k, v, g, beta = [[getattr(sc, name)[parity, li] for li in range(lin)] for name in ("k", "v", "g", "beta")]
        return gdn.replay_table(k, v, g, beta, [[st.rec[st.cur[li], li] for li in range(lin)] for st in states])
    layers = np.arange(lin, dtype=np.int64)
    head = np.empty((lin, 4), dtype=np.int64)
    for c, name in enumerate(("k", "v", "g", "beta")):
        t = getattr(sc, name)
        head[:, c] = t.data_ptr() + (parity * t.stride(0) + layers * t.stride(1)) * t.element_size()
    tail = np.empty((len(states), lin), dtype=np.int64)
    for s, st in enumerate(states):
        rec = st.rec
        if rec.dtype != torch.float32 or rec.dim() != 5:
            raise ValueError("every state is one fp32 (Hv, Dv, 128) tensor")
        cur = np.asarray(st.cur[:lin], dtype=np.int64)
        tail[s] = rec.data_ptr() + (cur * rec.stride(0) + layers * rec.stride(1)) * rec.element_size()
    return np.concatenate([head.ravel(), tail.ravel()])


# --- host-side GPU-wait stretches of shared rounds (done line) ---------------------------------------------------

class Times:
    """Host ms from a graphed shared round's start to its main launch (the GPU has nothing queued then), and from
    a draft pick's read-back to the next MTP launch, for rounds whose main graph replayed."""

    def __init__(self) -> None:
        self.rounds = self.steps = 0
        self.prep = self.draft = 0.0
        self.t0 = self.mark = None
        self.replay = False

    def start(self) -> None:
        self.t0, self.mark = time.perf_counter(), None

    def launched(self, replay: bool) -> None:
        """The main graph is about to launch (``replay``: its graph is kept)."""

        if self.t0 is None:
            return
        self.replay = replay
        if replay:
            self.prep += time.perf_counter() - self.t0
            self.rounds += 1
        self.t0 = None

    def picked(self) -> None:
        self.mark = time.perf_counter() if self.replay else None

    def mtp_launched(self, replay: bool = True) -> None:
        if self.mark is not None and replay:
            self.draft += time.perf_counter() - self.mark
            self.steps += 1
        self.mark = None

    def summary(self) -> str:
        """Cumulative since start: host ms before replayed main launches, host ms between draft read-backs and
        replayed MTP launches in those rounds, and their counts (a log reader diffs two done lines)."""

        return f"host_ms=prep:{self.prep * 1e3:.1f},draft:{self.draft * 1e3:.1f},rounds:{self.rounds},steps:{self.steps}"
