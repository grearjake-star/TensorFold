"""L (the fork's L measurements, "lock fix"): with TF_NGRAM_LOCK and --parallel, a stream's cache growth unpins n-gram table
runs before the memory gate makes it wait or end it; TF_NGRAM_LOCK=auto leaves the gate's floor free; released runs
return the same rows."""

from __future__ import annotations

import sys

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")
pytest.importorskip("torch")

from tensorfold.cuda.memory_gate import MemoryGate  # noqa: E402
from tensorfold.families.qwen4_exp import host_residency, host_table  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import multi as multi_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mlock/proc")

GIB = 1 << 30


def _table(tmp_path, counts=(3000, 500, 6400, 1900, 30, 4100)):
    rng = np.random.default_rng(0)
    tensors = {}
    for i, rows in enumerate(counts):
        tensors[f"emb.shard_{i}.weight"] = rng.integers(0, 2**32, (rows, 20), dtype=np.uint32)
        tensors[f"emb.shard_{i}.scales"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint16)
        tensors[f"emb.shard_{i}.biases"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint16)
    safetensors_numpy.save_file(tensors, str(tmp_path / "model-0.safetensors"))
    table = host_table.from_checkpoint(tmp_path, "emb", len(counts))
    return table, np.random.default_rng(1).integers(0, table.rows, 500)


def _vmlck() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024
    return 0


def test_release_unpins_the_last_locked_first_and_keeps_the_rows(tmp_path):
    table, ids = _table(tmp_path)
    before = table.gather(ids)
    base = _vmlck()
    full = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "all"}))
    order = list(table._locked)
    assert full == table.locked_bytes() > 0
    last = order[-1][1]
    freed = table.release(1)                       # one array, the last one locked
    assert freed == last and table._locked == order[:-1]
    assert table.locked_bytes() == full - last
    freed += table.release(1 << 62)                # everything left
    assert freed == full and table.locked_bytes() == 0 and table.release(1) == 0
    assert _vmlck() == base
    assert all(np.array_equal(x, y) for x, y in zip(before, table.gather(ids)))


class _State:
    """A slot's cache sizes as State reports them: 1 MiB a row over 2 layers."""

    def __init__(self, capacity=256, limit=65536):
        self.capacity, self.limit, self.layers = capacity, limit, 2

    def layer_bytes(self, rows=None):
        return (self.capacity if rows is None else rows) * (1 << 19)

    def cache_bytes(self, rows=None):
        return self.layers * self.layer_bytes(rows)

    def resize(self, rows):
        before = self.cache_bytes()
        self.capacity = rows
        return self.cache_bytes() - before


def _decoder(host, release):
    md = object.__new__(multi_mod.MultiDecoder)
    md.memory_gate = MemoryGate(1 << 50, reserve=2 * GIB, live=lambda: host[0])
    md.release, md.released = release, 0
    md.kept, md.streams, md.filling, md.free = [], {}, [], [_State() for _ in range(4)]
    md.solo, md.solo_on, md.planning = None, False, False          # 0.6.4: no lone-stream graph slot, not planning
    return md


def test_growth_unpins_table_runs_instead_of_refusing():
    host = [2 * GIB]                                # the host check: what MemAvailable leaves above the reserves
    pinned = [GIB] * 20
    calls = []

    def release(n):
        calls.append(n)
        freed = 0
        while pinned and freed < n:
            freed += pinned.pop()
        host[0] += freed                            # munlocked pages become reclaimable: MemAvailable rises
        return freed

    md = _decoder(host, release)
    st = _State()
    assert md._grow(st, 300)                        # 256 -> 8192 rows: 8 GiB + one layer's 4 GiB held twice
    assert st.capacity == 8192 and len(calls) == 1 and md.released == 12 * GIB      # 11.75 GiB short: 12 runs
    assert md.memory_gate.fits(0) and len(pinned) == 8


def test_without_pinned_pages_the_gate_still_refuses():
    host = [2 * GIB]
    md = _decoder(host, None)
    assert not md._grow(_State(), 300)
    md = _decoder(host, lambda n: 0)                # nothing left to unpin: the old behaviour (wait / newest ends)
    assert not md._grow(_State(), 300)


def test_gate_room_is_not_released_against():
    host = [100 * GIB]
    md = _decoder(host, lambda n: pytest.fail("the gate's own room refused, not the host: nothing to unpin"))
    md.memory_gate.room = 4 * GIB                   # held + extra past room - reserve
    assert not md._grow(_State(), 300)


def test_planned_growth_is_every_slots_first_step():
    md = _decoder([0], None)
    st = md.free[0]
    step = min(st.limit, multi_mod.STEP)
    want = 4 * (st.cache_bytes(step) - st.cache_bytes(multi_mod.FIRST)) + st.layer_bytes(step)
    assert md.planned_growth() == want


def test_auto_lock_under_parallel_leaves_the_gate_floor(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.cuda import ngram_residency

    multi = _decoder([0], None)
    monkeypatch.setattr(ngram_residency, "mem_total", lambda: 120 * GIB)
    floor = ngram_residency.gate_floor(multi)
    assert floor == capacity.reserve_bytes(120 * GIB) + 2 * GIB + multi.planned_growth()
    # the floor is what the gate's host check subtracts: at MemAvailable == floor, every slot's first step fits
    host = floor - capacity.reserve_bytes(120 * GIB)
    assert MemoryGate(1 << 50, reserve=2 * GIB, live=lambda: host).fits(multi.planned_growth())


# NB: TF_NGRAM_LOCK=<GiB>, a fixed pin budget. auto pins whatever MemAvailable leaves above its floor at start-up, so
# two loads of one build pinned 26-33 GiB and decoded at different speeds; a budget pins the same runs every load.

BIG = (30000, 20000, 64000, 25000, 21000, 41000)    # every array many pages: edge pages stay small beside it


def _words(table):
    """Every array's span in lock order (dense components first)."""
    return [host_residency._span(a)[1] for v in table.parts().values() for a in v]


def _same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_budget_counts_whole_runs_with_their_edge_pages(tmp_path):
    table, _ = _table(tmp_path, BIG)
    spans = _words(table)
    page = host_residency.PAGE
    # N runs fit a budget of their bytes less up to two pages each (page spans overrun a run's bytes by its edges)
    assert host_residency.planned_lock([table], sum(spans[:5])) == sum(spans[:5])
    assert host_residency.planned_lock([table], sum(spans[:5]) - 2 * page) == sum(spans[:5])
    assert host_residency.planned_lock([table], sum(spans[:5]) - 2 * 5 * page - 1) == sum(spans[:4])
    assert host_residency.planned_lock([table], 1 << 50) == sum(spans)


def test_budget_lock_pins_the_same_runs_whatever_memavailable(tmp_path, monkeypatch):
    table, ids = _table(tmp_path, BIG)
    before = table.gather(ids)
    spans = _words(table)
    budget = sum(spans[:5])
    want = host_residency.planned_lock([table], budget)
    assert want == budget
    policy = host_residency.residency_policy({"TF_NGRAM_LOCK": repr(budget / GIB)})
    got = []
    for avail in (40 * GIB, 60 * GIB, 10 * GIB + want):         # three "loads" with different MemAvailable
        monkeypatch.setattr(host_residency, "mem_available", lambda a=avail: a)
        n = host_residency.apply_lock(table, policy)
        got.append((n, list(table._locked)))
        table.unlock()
    assert got[0] == got[1] == got[2] and got[0][0] == want
    assert _same(before, table.gather(ids))


def test_budget_lock_is_clipped_only_below_the_floor(tmp_path, monkeypatch):
    table, _ = _table(tmp_path, BIG)
    spans = _words(table)
    policy = host_residency.residency_policy({"TF_NGRAM_LOCK": repr(sum(spans) / GIB)})     # the whole table
    monkeypatch.setattr(host_residency, "mem_available", lambda: 10 * GIB + sum(spans[:2]))  # floor 10 GiB
    assert host_residency.apply_lock(table, policy) == sum(spans[:2])
    table.unlock()


def test_budget_over_several_tables_and_the_startup_note(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ngram_residency

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _ = _table(tmp_path / "a", BIG)
    b, _ = _table(tmp_path / "b", (20000, 22000))
    first = sum(_words(a))
    budget = first + _words(b)[0]                               # all of a, the first array of b
    assert host_residency.planned_lock([a, b], budget) == budget
    res = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(budget / GIB)})
    monkeypatch.setattr(ngram_residency, "_unique", lambda w: {1: a, 2: b})
    monkeypatch.setattr(ngram_residency, "mem_available", lambda: 50 * GIB)
    monkeypatch.setattr(host_residency, "mem_available", lambda: 50 * GIB)
    note = res.start(None, False, None)
    assert res.pinned == budget == a.locked_bytes() + b.locked_bytes() and b.locked_bytes() == _words(b)[0]
    assert "(budget" in note and "met;" in note and "floor 10.0 GiB" in note
    a.unlock(), b.unlock()
    monkeypatch.setattr(host_residency, "mem_available", lambda: 10 * GIB + first // 2)
    note = res.start(None, False, None)
    assert res.pinned < budget and "CLIPPED" in note
    a.unlock(), b.unlock()


def test_budget_under_parallel_keeps_the_gate_floor_and_installs_the_release(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ngram_residency

    table, ids = _table(tmp_path, BIG)
    before = table.gather(ids)
    spans = _words(table)
    multi = _decoder([0], None)
    monkeypatch.setattr(ngram_residency, "mem_total", lambda: 120 * GIB)
    floor = ngram_residency.gate_floor(multi)
    budget = sum(spans[:4])
    assert host_residency.planned_lock([table], budget) == budget
    res = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(budget / GIB)})
    monkeypatch.setattr(ngram_residency, "_unique", lambda w: {1: table})
    monkeypatch.setattr(ngram_residency, "mem_available", lambda: floor + sum(spans))
    monkeypatch.setattr(host_residency, "mem_available", lambda: floor + sum(spans))
    note = res.start(None, False, multi)
    assert res.pinned == budget and "met;" in note and f"floor {floor / GIB:.1f} GiB" in note
    assert multi.release == res.release                         # elastic: growth still unpins, last run first
    assert res.release(1) == spans[3] and table.locked_bytes() == sum(spans[:3])
    assert _same(before, table.gather(ids))
    res.stop()                                                  # the TF_NGRAM_REPIN thread (test_flashnext_repin)
    table.unlock()
    # MemAvailable under the gate floor + budget: clipped down to the floor, never past it
    monkeypatch.setattr(host_residency, "mem_available", lambda: floor + sum(spans[:2]))
    res = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(budget / GIB)})
    note = res.start(None, False, _decoder([0], None))
    res.stop()
    assert "CLIPPED" in note and table.locked_bytes() == sum(spans[:2])
    table.unlock()


def test_budget_reads_the_unpinned_rest_back_once_unless_refresh_is_off(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ngram_residency

    table, ids = _table(tmp_path, BIG)
    before = table.gather(ids)
    spans = _words(table)
    calls = []
    monkeypatch.setattr(type(table), "refresh", lambda self, reserve, chunk=512: calls.append(reserve) or 7)
    monkeypatch.setattr(ngram_residency, "_unique", lambda w: {1: table})
    monkeypatch.setattr(host_residency, "mem_available", lambda: 50 * GIB)
    monkeypatch.setattr(ngram_residency, "mem_available", lambda: 50 * GIB)
    res = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(sum(spans[:3]) / GIB)})
    note = res.start(None, False, None)
    assert calls == [10 * GIB] and "the rest read back" in note
    res.after_request()                                         # once: no refresh thread after requests
    assert res._refreshing is None
    table.unlock()
    calls.clear()
    note = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(sum(spans[:3]) / GIB), "TF_NGRAM_REFRESH": "0"}).start(
        None, False, None)
    assert calls == [] and "read back" not in note
    table.unlock()
    note = ngram_residency.Residency({"TF_NGRAM_LOCK": "auto"}).start(None, False, None)
    assert calls == [] and "read back" not in note               # auto unchanged
    table.unlock()
    assert _same(before, table.gather(ids))
