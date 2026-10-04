"""CUDA graphs for --parallel rounds of several streams: the verify forward and each MTP draft step replay a graph.

A shared round's launches read every per-stream address (caches, DeltaNet states, n-gram tails, pooled index keys)
and every row's stream and position from device tables, so one graph serves any streams in any slots at any cache
size: the tables are persistent buffers per (streams, rows), rewritten by one host-to-device copy before a replay.
What stays host-side is fixed by the key: the window's stream count and total rows, the DeltaNet scratch parity
and a power-of-two context bucket that bounds the attention launches
(chunks and blocks past a row's keys write nothing, so a bound never changes bits; attn_multi._qsa_multi).

Last round's kept DeltaNet rows are folded into the states before the forward (``fold``), so a graph round's forward
writes only what it writes again with the same bits. A key runs its graph-mode forward eagerly the first ``after - 1``
times it is seen (compiling its kernels; with ``after`` 1, one discarded eager run first), then is captured and
replayed; at most ``limit`` graphs are kept (least recently used dropped). TF_MULTI_GRAPHS=0: eager.
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict

import numpy as np
import torch

from .attn_multi import PTRS
from .hostprep import Times

LIMIT = int(os.environ.get("TF_MULTI_GRAPHS_MAX", "160"))       # graphs kept (main + MTP), least recently used out
AFTER = int(os.environ.get("TF_MULTI_GRAPHS_AFTER", "2"))       # the sighting of a key that captures it


def fold(w, sc, states, held) -> None:
    """Fold each stream's held DeltaNet rows (last shared round's kept rows, in the scratch's other parity) into its
    recurrent states in place: multi_solo._flush's replay for every stream in one launch. Rounds on graphs fold
    here instead of in the trees (gdn_multi ``pending``), so replaying or re-running a graph never folds twice."""

    todo = [(st, rows) for st, rows in zip(states, held) if rows]
    if not todo or not sc.lin:
        return
    from tensorfold.cuda.kernels import gdn

    from . import hostprep

    ptrs = hostprep.fold_ptrs(sc, [st for st, _ in todo])
    width = max(len(rows) for _, rows in todo)
    kept = [r for _, rows in todo for r in list(rows) + [0] * (width - len(rows))]
    k0, v0 = sc.k[1 - sc.parity, 0], sc.v[1 - sc.parity, 0]
    if hostprep.ON:                              # one int32 table (rows then counts) and the pointers, pinned twice
        table = hostprep.put(ptrs, torch.int64, w.device, None, "fold_i64")
        ints = hostprep.put(np.asarray(kept + [len(rows) for _, rows in todo], dtype=np.int32), torch.int32, w.device,
                            None, "fold_i32")
        rows_dev, counts = ints[:len(kept)].view(len(todo), width), ints[len(kept):]
    else:
        table = gdn.to_device(ptrs, torch.int64, w.device)
        rows_dev = gdn.to_device(kept, torch.int32, w.device).view(len(todo), width)
        counts = gdn.to_device([len(rows) for _, rows in todo], torch.int32, w.device)
    gdn.replay(table, sc.lin, len(todo), rows_dev, counts, k0, v0, in_place=True)


def bucket(end: int) -> int:
    """The context bound a graph is captured at: a power of two from 8,192 (decode's Graphs._bucket, uncapped)."""

    return max(8192, 1 << (max(1, end) - 1).bit_length())


class RoundGraphs:
    def __init__(self, w, depth: int, *, limit: int = LIMIT, after: int = AFTER) -> None:
        self.w, self.width = w, depth + 1
        self.limit, self.after = max(1, limit), max(1, after)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = OrderedDict()
        self.seen: dict[tuple, int] = {}
        self.tables: dict[tuple, dict] = {}
        self.attn_layers = sum(1 for layer in w.layers if not layer.linear)
        self.lin = sum(1 for layer in w.layers if layer.linear)
        self.host = Times()                      # S1-HOST: host ms the GPU waits for in replayed rounds
        self.captures = self.replays = self.eager = self.dropped = 0
        self.kinds: dict[str, list[int]] = {}    # key[0] ("main" / "mtp") -> [replays, eager, captures, dropped]

    def attn_out(self, mtp: bool, n: int, rows: int) -> dict:
        """Persistent attn_multi.Step tables for windows of ``n`` streams and ``rows`` rows."""

        key = ("attn", mtp, n, rows)
        got = self.tables.get(key)
        if got is None:
            dev = self.w.device
            layers = 1 if mtp else self.attn_layers
            got = self.tables[key] = {
                "ints": torch.zeros((2 * rows + 2 * n,), dtype=torch.int32, device=dev),
                "ptrs": torch.zeros((layers * PTRS * n,), dtype=torch.int64, device=dev),
                "tails": torch.zeros((n,), dtype=torch.int64, device=dev)}
        return got

    def gdn_out(self, n: int, rows: int) -> dict:
        """Persistent gdn_multi.Tables tables (pending rows padded to ``width`` = depth + 1)."""

        key = ("gdn", n, rows)
        got = self.tables.get(key)
        if got is None:
            dev = self.w.device
            ints = rows + 4 * rows + 3 * rows + (n + 1) + n * self.width + n
            got = self.tables[key] = {"i32": torch.zeros((ints,), dtype=torch.int32, device=dev),
                                      "i64": torch.zeros((2 * self.lin * n,), dtype=torch.int64, device=dev)}
        return got

    def _capture(self, fn):
        torch.cuda.synchronize()                 # no gc.collect(): a full collection of a server's heap is a
        g = torch.cuda.CUDAGraph()               # large share of a mid-round capture; gc stays off while capturing
        enabled = gc.isenabled()                 # collecting old graphs mid-capture invalidates it (graphs.py)
        gc.disable()
        try:
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                out = fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        return g, out

    def run(self, key: tuple, fn):
        """``fn``'s result for this round: its graph replayed, or ``fn`` eagerly until the key is captured."""

        count = self.kinds.setdefault(str(key[0]), [0, 0, 0, 0])
        hit = self.graphs.get(key)
        if hit is not None:
            self.graphs.move_to_end(key)
            hit[0].replay()
            self.replays += 1
            count[0] += 1
            return hit[1]
        seen = self.seen[key] = self.seen.get(key, 0) + 1
        if seen < self.after:
            self.eager += 1
            count[1] += 1
            return fn()
        if seen == 1:
            fn()                                 # compiles its launches; a graph round's forward is idempotent
        while len(self.graphs) >= self.limit:
            old, _ = self.graphs.popitem(last=False)
            self.dropped += 1
            self.kinds.setdefault(str(old[0]), [0, 0, 0, 0])[3] += 1
        g, out = self._capture(fn)              # capture runs nothing: the replay is this round's forward
        count[2] += 1                            # a capture round: its replay is not counted as a replay
        self.graphs[key] = (g, out)
        g.replay()
        self.replays += 1
        return out

    def summary(self) -> str:
        """Cumulative counts since start, per kind: ``main=replays/eager/captures/dropped`` (a capture round is one
        capture, not a replay), and the graphs kept."""

        parts = [f"{k}={'/'.join(str(n) for n in v)}" for k, v in sorted(self.kinds.items())]
        host = getattr(self, "host", None)
        host = [host.summary()] if host is not None and host.rounds else []
        return " ".join(parts + [f"kept={len(self.graphs)}"] + host)
