"""S1-HOST (TF_MULTI_HOST_AHEAD): a shared round's DeltaNet tables, attention step tables and fold pointer table
built from numpy (base address + stride offsets, kept GDN schedules, double-buffered pinned staging) hold the same
values as the list-based v8.8 code, for any streams, window lengths, recurrent-state parities and cache moves."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda.kernels import gdn as shared
from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi, hostprep, multi_graphs

LIN, ATT = 6, 3


@pytest.fixture(autouse=True)
def _host_copies(monkeypatch):
    """The list-based path's device copies, on the host (no pinning)."""

    monkeypatch.setattr(shared, "to_device", lambda values, dtype, device: torch.tensor(values, dtype=dtype))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(shared, "pointers", lambda tensors: [t.data_ptr() for t in tensors])   # CPU tensors here
    hostprep._ATTN.clear()
    hostprep._PLANS.clear()
    yield
    hostprep._ATTN.clear()


class _State:                                          # weak-referenceable, like State
    pass


def _kc():
    return SimpleNamespace(k=torch.empty(64), v=torch.empty(64), ks=torch.empty(8), vs=torch.empty(8))


def _w():
    layers = [SimpleNamespace(index=i, linear=(i % 3 != 2)) for i in range(LIN + ATT)]
    return SimpleNamespace(layers=layers, mtp=SimpleNamespace(layer=SimpleNamespace(index=LIN + ATT, linear=False)),
                           device=torch.device("cpu"), comm=None)


def _state(w, rng, pos, version=0, offset=0):
    att = [l for l in w.layers if not l.linear]
    st = _State()
    st.pos, st.mtp_len, st.version, st.image_positions = pos, pos + 1, version, None
    conv = torch.empty((LIN + offset, 3, 8), dtype=torch.bfloat16)
    st.conv = conv[offset:]                            # a view with a storage offset
    st.rec = torch.empty((2, LIN, 2, 2, 4), dtype=torch.float32)
    st.cur = [int(x) for x in rng.integers(0, 2, LIN)]
    st.kc, st.ikc = [_kc() for _ in att], [torch.empty(32) for _ in att]
    st.pooled = [torch.empty(16) for _ in att]
    st.att_index = {l.index: i for i, l in enumerate(att)}
    st.ple_tail = torch.empty(8)
    st.mtp_kc, st.mtp_ikc, st.mtp_pooled = _kc(), torch.empty(32), torch.empty(16)
    return st


def _segs(states, lengths):
    segs, a0 = [], 0
    for st, n in zip(states, lengths):
        segs.append((st, a0, a0 + n))
        a0 += n
    return segs


def _scratch(rows):
    sc = object.__new__(gdn_multi.Scratch)
    sc.lin, sc.parity, sc.q = LIN, 0, None
    sc.k = torch.empty((2, LIN, rows, 2, 4))
    sc.v = torch.empty((2, LIN, rows, 2, 2), dtype=torch.bfloat16)
    sc.g, sc.beta = torch.empty((2, LIN, rows, 2)), torch.empty((2, LIN, rows, 2))
    return sc


def _both(monkeypatch, fn):
    monkeypatch.setattr(hostprep, "ON", False)
    ref = fn()
    monkeypatch.setattr(hostprep, "ON", True)
    got = fn()
    again = fn()                                       # kept schedules / pointers: the same again
    return ref, got, again


LENGTHS = [[1], [4, 4], [4, 1, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1], [4, 2, 4, 4, 1, 3, 4, 2]]


@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("graph", [False, True])
def test_gdn_tables_same(monkeypatch, lengths, graph):
    w, rng = _w(), np.random.default_rng(len(lengths))
    states = [_state(w, rng, 100 * i, offset=i % 2) for i in range(len(lengths))]
    segs = _segs(states, lengths)
    sc = _scratch(segs[-1][2])
    n, rows = len(segs), segs[-1][2]
    pending = [[] for _ in segs] if graph else [list(range(int(rng.integers(0, 4)))) for _ in segs]

    def build():
        out = None
        if graph:
            ints = rows + 4 * rows + 3 * rows + (n + 1) + n * 5 + n
            out = {"i32": torch.full((ints,), -7, dtype=torch.int32), "i64": torch.zeros((2 * LIN * n,), dtype=torch.int64)}
        t = gdn_multi.Tables(w, sc, segs, pending, out=out, width=5 if graph else None)
        return (t.sid.clone(), t.win.clone(), t.plan.entries.clone(), t.plan.starts.clone(), t.plan.slots,
                t.plan.max_rows, t.held.clone(), t.held_counts.clone(), t.folds, t.conv.clone(), t.state.clone())

    ref, got, again = _both(monkeypatch, build)
    for a, b, c in zip(ref, got, again):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a, b) and torch.equal(a, c)
        else:
            assert a == b == c


@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_attn_step_same(monkeypatch, lengths, mtp, graph):
    w, rng = _w(), np.random.default_rng(7 + len(lengths))
    states = [_state(w, rng, 1000 + 37 * i) for i in range(len(lengths))]
    segs = _segs(states, [1] * len(lengths) if mtp else lengths)
    n, rows = len(segs), segs[-1][2]
    layers = 1 if mtp else ATT

    def build():
        out = None
        if graph:
            out = {"ints": torch.zeros((2 * rows + 2 * n,), dtype=torch.int32),
                   "ptrs": torch.zeros((layers * attn_multi.PTRS * n,), dtype=torch.int64),
                   "tails": torch.zeros((n,), dtype=torch.int64)}
        s = attn_multi.Step(w, segs, mtp=mtp, out=out, bucket=8192 if graph else None)
        return (s.posr.clone(), s.sid.clone(), s.first.clone(), s.counts.clone(), s.ptrs.clone(),
                None if s.tails is None else s.tails.clone(), s.ends, s.most, s.index)

    ref, got, again = _both(monkeypatch, build)
    for a, b, c in zip(ref, got, again):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a, b) and torch.equal(a, c)
        else:
            assert a == b == c


def test_attn_pointers_follow_a_resize(monkeypatch):
    """A kept state's pointers are read again once its caches move (State.version), and dropped with the state."""

    monkeypatch.setattr(hostprep, "ON", True)
    w, rng = _w(), np.random.default_rng(3)
    st = _state(w, rng, 50)
    segs = _segs([st], [4])
    first = attn_multi.Step(w, segs, mtp=False).ptrs.clone()
    st.kc[1] = _kc()                                   # State.resize: new caches, version + 1
    st.version += 1
    moved = attn_multi.Step(w, segs, mtp=False).ptrs.clone()
    monkeypatch.setattr(hostprep, "ON", False)
    ref = attn_multi.Step(w, segs, mtp=False).ptrs.clone()
    assert torch.equal(moved, ref) and not torch.equal(first, moved)
    assert hostprep._ATTN
    del st, segs
    import gc

    gc.collect()
    assert not hostprep._ATTN


@pytest.mark.parametrize("held", [[[0, 1, 2]], [[0], [], [3, 4, 5, 6]], [[1, 2]] * 8])
def test_fold_same(monkeypatch, held):
    w, rng = _w(), np.random.default_rng(len(held))
    states = [_state(w, rng, 10 * i) for i in range(len(held))]
    sc = _scratch(16)
    sc.parity = 1
    calls = []
    monkeypatch.setattr(shared, "replay", lambda table, layers, streams, rows, counts, k0, v0, in_place=False:
                        calls.append((table.clone(), layers, streams, rows.clone(), counts.clone(), k0.data_ptr(),
                                      v0.data_ptr(), in_place)))
    monkeypatch.setattr(hostprep, "ON", False)
    multi_graphs.fold(w, sc, states, held)
    monkeypatch.setattr(hostprep, "ON", True)
    multi_graphs.fold(w, sc, states, held)
    if not any(held):
        assert not calls
        return
    (a, *ra), (b, *rb) = calls
    assert torch.equal(a, b) and a.dtype == b.dtype
    for x, y in zip(ra, rb):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y


def test_put_host_and_kinds():
    vals = np.arange(10, dtype=np.int64) * 3
    t = hostprep.put(vals, torch.int32, "cpu", None, "t")
    assert t.dtype == torch.int32 and t.tolist() == vals.tolist()
    out = torch.zeros(10, dtype=torch.int64)
    assert hostprep.put(vals, torch.int64, "cpu", out, "t") is out and out.tolist() == vals.tolist()


def test_times_summary():
    t = hostprep.Times()
    t.start()
    t.launched(True)
    t.picked()
    t.mtp_launched(True)
    t.picked()
    t.mtp_launched(False)
    t.start()
    t.launched(False)
    t.picked()
    t.mtp_launched(True)
    assert t.rounds == 1 and t.steps == 1 and t.summary().startswith("host_ms=prep:")
