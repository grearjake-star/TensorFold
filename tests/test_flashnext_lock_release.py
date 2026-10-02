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
