"""The expected-time draft stop: a later MTP draft is verified only while its chain's probability repays its row."""

from __future__ import annotations

import os

# Verify windows of 1, 2, ... rows at an 8K context on DGX Spark, ms (0.3.6.2 kernels, real reply tokens, CUDA graphs;
# a greedy diagnostic run, 2026-09-28), and one chained MTP draft step: the cost rule's table.
VERIFY_MS = (24.5, 28.4, 32.2, 35.0, 38.1, 42.0, 44.4, 46.2, 49.2, 51.8, 54.2, 57.0, 58.6)
DRAFT_MS = 1.2


def timing_table(env=os.environ) -> tuple[tuple[float, ...], float]:
    """The cost rule's verify-window table and draft-step ms: ``TF_VERIFY_MS`` (comma-separated ms of windows of 1, 2, ...
    rows, e.g. a checkpoint's own measurement) and ``TF_DRAFT_MS`` override the MLX defaults above. Speed only."""

    ms, dms = VERIFY_MS, DRAFT_MS
    if env.get("TF_VERIFY_MS", "").strip():
        ms = tuple(float(x) for x in env["TF_VERIFY_MS"].split(","))
        if len(ms) < 2 or any(b < a for a, b in zip(ms, ms[1:])) or ms[0] <= 0:
            raise ValueError("TF_VERIFY_MS: at least two positive, non-decreasing window times (ms)")
    if env.get("TF_DRAFT_MS", "").strip():
        dms = float(env["TF_DRAFT_MS"])
        if dms < 0:
            raise ValueError("TF_DRAFT_MS: 0 or more")
    return ms, dms


def cost_bars(cost: float, count: int, table: tuple[tuple[float, ...], float] | None = None) -> list[float]:
    """The running-product bar drafts 1..count + 1 must reach under the cost rule: ``cost`` (tokens per ms) times
    what verifying the draft adds to the window plus its MTP step (``table``: (verify ms, draft ms), default the MLX
    table)."""

    verify, dms = table if table is not None else (VERIFY_MS, DRAFT_MS)
    ms = list(verify)
    while len(ms) < count + 3:                          # past the table: its last step again
        ms.append(ms[-1] + (ms[-1] - ms[-2]))
    return [cost * (ms[j + 1] - ms[j] + dms) for j in range(count + 1)]
