"""TF_NGRAM_ADVICE (UP-REPORT, the coordinator's MADV_RANDOM item), re-derived for 0.6.0: the maps that n-gram
lookups read can be advised random-access (no read-ahead on a page fault) or normal after the startup read.
0.6.0 opens the per-shard maps and the MLX table's whole-file gather maps MADV_RANDOM; "normal" gives the maps
``gather`` really reads the kernel's read-around back (B measured random worse under memory pressure). The advice
never changes the rows a lookup returns."""

from __future__ import annotations

import sys

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from tensorfold.families.qwen4_exp import host_table  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/smaps")


def _checkpoint(tmp_path, counts):
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
    return tensors


def _flags_at(address: int) -> set[str]:
    """VmFlags of the mapping of this process that holds ``address``."""

    inside = False
    for line in open("/proc/self/smaps"):
        head = line.split()[0]
        if line[0] in "0123456789abcdef" and "-" in head:
            lo, hi = (int(x, 16) for x in head.split("-"))
            inside = lo <= address < hi
        elif inside and line.startswith("VmFlags:"):
            return set(line.split()[1:])
    raise AssertionError(f"no mapping holds {address:#x}")


def _addr(array) -> int:
    return array.ctypes.data


def test_random_advice_keeps_the_rows_and_marks_the_gather_maps(tmp_path):
    counts = [37, 5, 64, 19, 3, 41]
    tensors = _checkpoint(tmp_path, counts)
    table = host_table.from_checkpoint(tmp_path, "emb", len(counts))
    ids = np.random.default_rng(1).integers(0, table.rows, 200)
    before = table.gather(ids)
    assert "rr" in _flags_at(_addr(table.words[0]))                    # the shard maps are random at open
    assert all("rr" in _flags_at(_addr(m)) for m in table.files)       # 0.6.0: the gather maps too
    advised = table.advise("normal")
    assert advised == sum((tmp_path / f"model-{f}.safetensors").stat().st_size for f in (0, 1))
    assert all("rr" not in _flags_at(_addr(m)) for m in table.files)   # read-around back on lookups' faults
    assert np.array_equal(table.gather(ids)[0], before[0])
    table.advise("random")
    assert all("rr" in _flags_at(_addr(m)) for m in table.files)       # VM_RAND_READ: no read-ahead
    after = table.gather(ids)
    for a, b in zip(before, after):
        assert np.array_equal(a, b)
    starts = np.cumsum([0] + counts)
    s = int(np.searchsorted(starts, ids[0], side="right") - 1)
    assert np.array_equal(after[0][0], tensors[f"emb.shard_{s}.weight"][ids[0] - starts[s]])
    table.advise("normal")
    assert all("rr" not in _flags_at(_addr(m)) for m in table.files)
    with pytest.raises(ValueError, match="random or normal"):
        table.advise("sequential")


def test_bf16_table_advice_is_on_its_values(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    rng = np.random.default_rng(2)
    save_file({f"emb.shard_{i}.weight": torch.from_numpy(rng.standard_normal((n, 8)).astype(np.float32))
               .to(torch.bfloat16) for i, n in enumerate((9, 30))}, str(tmp_path / "t.safetensors"))
    table = host_table.open_table(tmp_path, [("t.safetensors", f"emb.shard_{i}") for i in range(2)], lambda n: 1.0)
    ids = rng.integers(0, table.rows, 50)
    before = table.gather(ids)
    assert all("rr" in _flags_at(_addr(v)) for v in table.values)
    assert table.advise("normal") > 0
    assert all("rr" not in _flags_at(_addr(v)) for v in table.values)
    assert np.array_equal(table.gather(ids), before)
    table.advise("random")
    assert all("rr" in _flags_at(_addr(v)) for v in table.values)
