"""TF_NGRAM_REPIN (v8.8): table runs that growing stream caches unpinned are pinned back between requests, one run a
period, up to the startup lock's budget, never past the floor + margin, not right after a release and not while a
prompt prefills. Residency only: the gathered rows stay the same bytes."""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")
pytest.importorskip("torch")

from tensorfold.families.qwen4_exp import host_residency  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import ngram_residency  # noqa: E402

from .test_flashnext_lock_release import BIG, _decoder, _table, _words  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mlock/proc")

GIB = 1 << 30


class _Host:
    """MemAvailable as the host reports it: ``free`` less what the tables hold pinned."""

    def __init__(self, free, tables):
        self.free, self.tables = free, tables

    def __call__(self):
        return self.free - sum(t.locked_bytes() for t in self.tables)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def step(self, s):
        self.t += s
        return self.t


def _start(tmp_path, monkeypatch, env=None, runs=4, spare=None):
    table, ids = _table(tmp_path, BIG)
    spans = _words(table)
    multi = _decoder([0], None)
    monkeypatch.setattr(ngram_residency, "mem_total", lambda: 120 * GIB)
    floor = ngram_residency.gate_floor(multi)
    budget = sum(spans[:runs])
    host = _Host(floor + sum(spans) + (8 * GIB if spare is None else spare), [table])
    monkeypatch.setattr(ngram_residency, "_unique", lambda w: {1: table})
    monkeypatch.setattr(ngram_residency, "mem_available", host)
    monkeypatch.setattr(host_residency, "mem_available", host)
    res = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(budget / GIB), **(env or {})})
    res.PERIOD = 1e9                                   # the thread never steps during a test: the test drives it
    note = res.start(None, False, multi)
    res.PERIOD, res.COOLDOWN = 5.0, 30.0
    res.clock = clock = _Clock()
    assert "met;" in note and res.pinned == budget
    return res, table, ids, spans, multi, host, floor, clock


def test_repin_restores_the_released_runs_after_caches_shrink(tmp_path, monkeypatch, capsys):
    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch)
    before, order = table.gather(ids), list(table._locked)
    assert res.target == sum(spans[:4]) and res.floor == floor and res._repinning.is_alive()
    assert res.release(1) + res.release(1) == spans[3] + spans[2]       # two growths unpinned the last two runs
    assert res.repin_step(clock.step(29)) == 0                        # cooldown after a release
    assert res.repin_step(clock.step(2)) == spans[2]                  # one run: the first unpinned, in lock order
    assert res.repin_step(clock.step(1)) == 0                         # <= one run a period
    assert res.repin_step(clock.step(4.5)) == spans[3]
    assert res.repin_step(clock.step(20)) == 0                        # the budget is met: nothing more
    assert table._locked == order and table.locked_bytes() == res.target
    assert all(np.array_equal(x, y) for x, y in zip(before, table.gather(ids)))
    out = capsys.readouterr().out
    assert out.count("memory: re-pinned") == 2 and f"now {res.target / GIB:.2f} GiB locked" in out
    res.stop()
    table.unlock()


def test_repin_never_takes_memavailable_under_the_floor_plus_margin(tmp_path, monkeypatch):
    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch, spare=0)
    res.release(1 << 62)                                               # everything unpinned
    clock.step(res.COOLDOWN)
    # MemAvailable after one more run would sit just under floor + margin: no re-pin, however long it waits
    host.free = floor + res.MARGIN + spans[0] - 1
    assert [res.repin_step(clock.step(res.PERIOD)) for _ in range(5)] == [0] * 5
    assert table.locked_bytes() == 0
    # room for exactly two runs above the floor + margin: two runs, then it stops (the budget has room for more)
    host.free = floor + res.MARGIN + spans[0] + spans[1]
    got = [res.repin_step(clock.step(res.PERIOD)) for _ in range(6)]
    assert got == [spans[0], spans[1], 0, 0, 0, 0] and res.target > spans[0] + spans[1]
    assert host() >= floor + res.MARGIN
    res.stop()
    table.unlock()


def test_repin_respects_the_budget_and_waits_for_prefill(tmp_path, monkeypatch):
    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch, runs=2, spare=64 * GIB)
    res.release(1 << 62)
    clock.step(res.COOLDOWN)
    multi.filling = ["a prompt"]                                       # a prompt prefills: the re-pin waits
    assert res.repin_step(clock.step(res.PERIOD)) == 0
    multi.filling = []
    got = [res.repin_step(clock.step(res.PERIOD)) for _ in range(5)]
    assert got == [spans[0], spans[1], 0, 0, 0] and table.locked_bytes() == res.target == sum(spans[:2])
    res.stop()
    table.unlock()


def test_no_thrash_a_release_resets_the_cooldown(tmp_path, monkeypatch):
    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch)
    res.release(1 << 62)
    assert res.repin_step(clock.step(res.COOLDOWN)) == spans[0]
    res.release(1)                                                     # the next growth takes it straight back
    assert [res.repin_step(clock.step(s)) for s in (1, 9, 10, 9)] == [0, 0, 0, 0]    # inside the cooldown
    assert table.locked_bytes() == 0
    # MemAvailable inside the hysteresis band (above the floor, under floor + margin + a run): no pin either
    host.free = floor + res.MARGIN // 2 + spans[0]
    assert res.repin_step(clock.step(res.COOLDOWN)) == 0
    res.stop()
    table.unlock()


def test_repin_off_and_bad_values(tmp_path, monkeypatch):
    res, table, *_ = _start(tmp_path, monkeypatch, env={"TF_NGRAM_REPIN": "0"})
    assert res._repinning is None and res.target == 0
    res.release(1 << 62)
    assert res.repin_step(1e12) == 0 and table.locked_bytes() == 0
    with pytest.raises(ValueError, match="TF_NGRAM_REPIN"):
        ngram_residency.Residency({"TF_NGRAM_REPIN": "yes"})
    # without --parallel (no stream caches to release for) there is no thread either
    assert ngram_residency.Residency({"TF_NGRAM_LOCK": "1"})._repinning is None


def test_repin_thread_runs_and_stops(tmp_path, monkeypatch):
    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch)
    res.stop()
    res._repinning.join(timeout=5)
    assert not res._repinning.is_alive()
    res2 = ngram_residency.Residency({"TF_NGRAM_LOCK": repr(sum(spans[:2]) / GIB)})
    res2.PERIOD, res2.COOLDOWN = 0.01, 0.0
    table.unlock()
    res2.start(None, False, _decoder([0], None))
    res2.release(1 << 62)
    deadline = time.monotonic() + 5
    while table.locked_bytes() < res2.target and time.monotonic() < deadline:
        time.sleep(0.02)
    assert table.locked_bytes() == res2.target
    res2.stop()
    table.unlock()


def test_lock_next_pins_in_lock_order(tmp_path):
    table, ids = _table(tmp_path, BIG)
    order = []
    names = tuple(table.parts())
    while True:
        got = table.lock_next(names, 1 << 50)
        if not got:
            break
        order.append(table._locked[-1])
    assert order == list(map(host_residency._span, [a for v in table.parts().values() for a in v]))
    assert table.lock_next(names, 1 << 50) == 0
    table.unlock()
    assert table.lock_next(table.DENSE, 0) == 0                        # a budget below the first run: none
