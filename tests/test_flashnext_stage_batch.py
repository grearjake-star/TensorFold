"""An EXL3 pack's ``stage`` with several windows: one n-gram gather, host write and copy per layer (TF_STAGE_BATCH,
the default) stages the same bytes in the same rows as one of each per window (TF_STAGE_BATCH=0)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import forward

HEADS, WORDS, TABLE_ROWS = 4, 6, 997


class _Ngram:
    heads = HEADS

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [L, heads] that depend on the history and every token (as the real hashing does)."""

        seq = np.concatenate([np.asarray(history, dtype=np.int64), np.asarray(tokens, dtype=np.int64)])
        tail = seq[-len(tokens):]
        mix = (tail[:, None] * 131 + np.arange(HEADS)[None] * 7919 + int(seq.sum()) * 17) % TABLE_ROWS
        return mix.astype(np.int64)


class _Table:
    def __init__(self, seed: int) -> None:
        self.data = np.random.default_rng(seed).integers(-32768, 32767, (TABLE_ROWS, WORDS), dtype=np.int16)
        self.calls = 0

    def gather(self, ids: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.data[np.asarray(ids, dtype=np.int64).reshape(-1)]


class _Event:
    def synchronize(self) -> None:
        pass

    def record(self) -> None:
        pass


def _setup(rows: int, layers: int):
    tables = [_Table(seed) for seed in range(layers)]
    w = SimpleNamespace(
        x3=SimpleNamespace(ple_dev=torch.zeros((rows * HEADS, WORDS), dtype=torch.int16),
                           ple_emb=torch.zeros((rows, 1))),
        layers=[SimpleNamespace(ple=SimpleNamespace(ngram=_Ngram(), table=t)) for t in tables]
        + [SimpleNamespace(ple=None)])
    b = SimpleNamespace(rows=rows, staged=_Event(), ids_host=torch.zeros(rows, dtype=torch.int32),
                        ids=torch.zeros(rows, dtype=torch.int32))
    b.x3_ple = forward._X3Ple(rows * HEADS, WORDS, torch.device("cpu"))
    b.x3_ple.ple_host.fill_(-1)                     # rows nobody stages keep the sentinel in both modes
    b.x3_ple.ple_dev.fill_(-1)
    return w, b, tables


def _windows(lengths, seed: int):
    rng = np.random.default_rng(100 + seed)
    out = []
    for n in lengths:
        st = SimpleNamespace(pos=int(rng.integers(0, 50)), capacity=10_000,
                             ple_history=rng.integers(0, 5000, 3).astype(np.int64))
        out.append((st, [int(t) for t in rng.integers(0, 5000, n)]))
    return out


@pytest.mark.parametrize("lengths", [(1, 1), (3, 1, 2), (1, 4, 1, 1, 6, 2, 1, 3), (11,), (2, 11, 11, 1)])
def test_batched_staging_writes_the_same_bytes(lengths, monkeypatch) -> None:
    rows = sum(lengths) + 3
    got = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("TF_STAGE_BATCH", flag)
        w, b, tables = _setup(rows, layers=1)
        windows = _windows(lengths, seed=len(lengths))
        segs = forward.stage(w, b, windows)
        got[flag] = (b.x3_ple.ple_host.clone(), b.x3_ple.ple_dev.clone(), b.ids_host.clone(),
                     [(a0, a1) for _, a0, a1 in segs], [(st.ple_last[0].tolist(), st.ple_last[1].tolist())
                                                        for st, _ in windows], tables[0].calls)
    off, on = got["0"], got["1"]
    assert torch.equal(off[0], on[0]) and torch.equal(off[1], on[1]) and torch.equal(off[2], on[2])
    assert off[3] == on[3] and off[4] == on[4]
    assert off[5] == len(lengths) and on[5] == 1                # one gather a layer instead of one a window
    staged = sum(lengths) * HEADS
    assert torch.all(on[1][staged:] == -1)                       # nothing past the windows' rows
    w, _, tables = _setup(rows, layers=1)
    want = np.concatenate([tables[0].data[_Ngram().ids(st.ple_history, np.asarray(t))].reshape(-1, WORDS)
                           for st, t in _windows(lengths, seed=len(lengths))])
    assert np.array_equal(on[1][:staged].numpy(), want)


def test_batched_staging_never_mixes_layers_tables(monkeypatch) -> None:
    """Two PLE layers (the real pack has one; the staging is shared): each layer's single lookup reads its own table,
    and the last layer's rows are what stays staged, exactly as per-window staging leaves them."""

    lengths = (2, 3, 1)
    got = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("TF_STAGE_BATCH", flag)
        w, b, tables = _setup(sum(lengths), layers=2)
        forward.stage(w, b, _windows(lengths, seed=7))
        got[flag] = (b.x3_ple.ple_dev.clone(), [t.calls for t in tables])
    assert torch.equal(got["0"][0], got["1"][0])
    assert got["0"][1] == [3, 3] and got["1"][1] == [1, 1]
