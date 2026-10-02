"""The n-gram residency hooks (TF_NGRAM_LOCK, TF_NGRAM_REFRESH) change which table pages stay in memory, never the
rows a lookup returns."""

from __future__ import annotations

import sys

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from tensorfold.families.qwen4_exp import host_residency, host_table  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mlock/mincore/proc")

GIB = 1 << 30


def _table(tmp_path, counts=(3000, 500, 6400, 1900, 30, 4100)):
    rng = np.random.default_rng(0)
    tensors = {}
    for i, rows in enumerate(counts):
        tensors[f"emb.shard_{i}.weight"] = rng.integers(0, 2**32, (rows, 20), dtype=np.uint32)
        tensors[f"emb.shard_{i}.scales"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint16)
        tensors[f"emb.shard_{i}.biases"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint16)
    half = len(counts) // 2
    for f, part in enumerate((range(half), range(half, len(counts)))):
        safetensors_numpy.save_file({k: v for k, v in tensors.items() if int(k.split(".")[1][6:]) in part},
                                    str(tmp_path / f"model-{f}.safetensors"))
    table = host_table.from_checkpoint(tmp_path, "emb", len(counts))
    ids = np.random.default_rng(1).integers(0, table.rows, 500)
    return table, ids


def _vmlck() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024
    return 0


def _same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_dense_and_full_locks_keep_the_rows_and_unlock(tmp_path):
    table, ids = _table(tmp_path)
    before = table.gather(ids)
    base = _vmlck()
    dense = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "dense"}))
    spans = {k: sum(host_residency._span(a)[1] for a in v) for k, v in table.parts().items()}
    assert dense == spans["scales"] + spans["biases"]
    assert _vmlck() - base >= dense - 4 * host_residency.PAGE * len(table.scales)     # neighbours share edge pages
    assert _same(before, table.gather(ids))
    full = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "all"}))
    assert full == sum(spans.values())
    assert _same(before, table.gather(ids))
    table.unlock()
    assert _vmlck() == base
    assert _same(before, table.gather(ids))


def test_auto_lock_stops_at_the_reserve(tmp_path, monkeypatch):
    table, ids = _table(tmp_path)
    before = table.gather(ids)
    spans = {k: [host_residency._span(a)[1] for a in v] for k, v in table.parts().items()}
    dense = sum(spans["scales"]) + sum(spans["biases"])
    room = dense + spans["words"][0] + spans["words"][1] // 2           # dense, one words shard, half another
    monkeypatch.setattr(host_residency, "mem_available", lambda: 3 * GIB + room)
    got = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "auto:3"}))
    assert got == dense + spans["words"][0]                              # whole arrays only, never past the room
    monkeypatch.setattr(host_residency, "mem_available", lambda: 2 * GIB)
    assert host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "auto"})) == got   # no room: kept
    assert _same(before, table.gather(ids))
    table.unlock()


def test_residency_is_read_only_and_counts_every_page(tmp_path):
    table, ids = _table(tmp_path)
    before = table.gather(ids)
    res = table.residency()
    assert set(res) == {"scales", "biases", "words"}
    for name, (have, total) in res.items():
        assert total == sum(host_residency._span(a)[1] for a in table.parts()[name])
        assert 0 <= have <= total
    assert _same(before, table.gather(ids))


def test_refresh_asks_for_missing_runs_within_the_reserve(tmp_path, monkeypatch):
    table, ids = _table(tmp_path)
    before = table.gather(ids)
    page = host_residency.PAGE
    def fake(at, size):                        # pretend pages 1-2 and 4 of every array were evicted
        bits = np.ones(size // page, dtype=np.uint8)
        bits[1:3] = 0
        bits[4:5] = 0
        return bits

    asked = []

    class Libc:
        def madvise(self, at, n, advice):
            asked.append((at, n, advice))
            return 0

    monkeypatch.setattr(host_residency, "_mincore", fake)
    monkeypatch.setattr(host_residency, "_libc", lambda: Libc())
    monkeypatch.setattr(host_residency, "mem_available", lambda: 100 * GIB)
    got = table.refresh(10 * GIB, chunk=4)
    import mmap

    assert all(a == mmap.MADV_WILLNEED for _, _, a in asked)
    pages = [host_residency._span(a)[1] // page for v in table.parts().values() for a in v]
    missing = sum(min(n, 3) - 1 + (n > 4) for n in pages)                 # pages 1-2 and 4 where they exist
    assert got == missing * page
    assert len(asked) == sum((n > 1) + (n > 4) for n in pages)          # chunk [0, 4) and chunk [4, 8)
    assert all(n <= 4 * page and at % page == 0 for at, n, _ in asked)
    asked.clear()
    monkeypatch.setattr(host_residency, "mem_available", lambda: 10 * GIB + 3 * page)
    assert table.refresh(10 * GIB, chunk=4) == 3 * page and len(asked) == 2   # stops before the reserve
    asked.clear()
    monkeypatch.setattr(host_residency, "mem_available", lambda: 9 * GIB)
    assert table.refresh(10 * GIB) == 0 and not asked
    monkeypatch.undo()
    assert _same(before, table.gather(ids))


def test_policy_values_and_refusals():
    p = host_residency.residency_policy({})
    assert p == {"lock": None, "lock_reserve": 10 * GIB, "refresh": False, "reserve": 10 * GIB, "startup": True}
    p = host_residency.residency_policy({"TF_NGRAM_LOCK": "auto:6.5", "TF_NGRAM_REFRESH": "1", "TF_NGRAM_RESERVE_GIB": "12"})
    assert p == {"lock": "auto", "lock_reserve": int(6.5 * GIB), "refresh": True, "reserve": 12 * GIB, "startup": True}
    assert host_residency.residency_policy({"TF_NGRAM_REFRESH": "0"})["startup"] is False
    for bad in ({"TF_NGRAM_LOCK": "yes"}, {"TF_NGRAM_LOCK": "auto:x"}, {"TF_NGRAM_REFRESH": "on"},
                {"TF_NGRAM_RESERVE_GIB": "-1"}, {"TF_NGRAM_RESERVE_GIB": "ten"}):
        with pytest.raises(ValueError, match="TF_NGRAM"):
            host_residency.residency_policy(bad)


def test_bf16_and_nvfp4_tables_take_the_same_hooks(tmp_path):
    """0.5.0's other table layouts (the published NVFP4 checkpoint's bf16 table; hibrid48's NVFP4 table): the same
    lock, residency and refresh, their dense components first; rows unchanged."""

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    rng = np.random.default_rng(3)
    save_file({f"emb.shard_{i}.weight": torch.from_numpy(rng.standard_normal((n, 64)).astype(np.float32))
               .to(torch.bfloat16) for i, n in enumerate((900, 3000))}, str(tmp_path / "b.safetensors"))
    b16 = host_table.open_table(tmp_path, [("b.safetensors", f"emb.shard_{i}") for i in range(2)], lambda _: 1.0)
    tensors = {}
    for i, n in enumerate((700, 2500)):
        tensors[f"q.shard_{i}.weight"] = torch.from_numpy(rng.integers(0, 256, (n, 32), dtype=np.uint8))
        tensors[f"q.shard_{i}.weight_scale"] = torch.from_numpy(
            rng.integers(0x30, 0x40, (n, 4), dtype=np.uint8)).view(torch.float8_e4m3fn)
    save_file(tensors, str(tmp_path / "q.safetensors"))
    fp4 = host_table.open_table(tmp_path, [("q.safetensors", f"q.shard_{i}") for i in range(2)], lambda _: 1.0)
    assert isinstance(fp4, host_table.NVFP4Table) and isinstance(b16, host_table.BF16Table)
    base = _vmlck()
    for table, dense in ((b16, ()), (fp4, ("scales",))):
        ids = np.random.default_rng(4).integers(0, table.rows, 300)
        before = table.gather(ids)
        assert table.DENSE == dense and set(table.residency()) == set(table.parts())
        got = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "dense"}))
        assert got == sum(host_residency._span(a)[1] for name in dense for a in table.parts()[name])
        full = host_residency.apply_lock(table, host_residency.residency_policy({"TF_NGRAM_LOCK": "all"}))
        assert full == sum(host_residency._span(a)[1] for v in table.parts().values() for a in v)
        assert table.refresh(0) >= 0
        assert np.array_equal(table.gather(ids), before)
        table.unlock()
    assert _vmlck() == base
