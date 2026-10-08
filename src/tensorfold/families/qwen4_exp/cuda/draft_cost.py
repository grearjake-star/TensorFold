"""The expected-time draft stop: a later MTP draft is verified only while its chain's probability repays its row."""

from __future__ import annotations

import math
import os

# Verify windows of 1, 2, ... rows at an 8K context on DGX Spark, ms (0.3.6.2 kernels, real reply tokens, CUDA graphs;
# a greedy diagnostic run, 2026-09-28), and one chained MTP draft step: the cost rule's table.
VERIFY_MS = (24.5, 28.4, 32.2, 35.0, 38.1, 42.0, 44.4, 46.2, 49.2, 51.8, 54.2, 57.0, 58.6)
DRAFT_MS = 1.2


def cost_setting(default: float, env=os.environ) -> float:
    """``TF_DRAFT_COST``: the expected-time stop's tokens per ms (unset: ``default``; empty, "0" or "off": no cost
    stop). A finite number 0 or more: "nan" would turn the stop off silently, "inf" would stop every chain at its
    first draft, and both break the two ranks' settings check."""

    raw = env.get("TF_DRAFT_COST")
    if raw is None:
        return float(default)
    value = raw.strip().lower()
    if value in ("", "off"):
        return 0.0
    try:
        cost = float(value)
    except ValueError:
        cost = math.nan
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"TF_DRAFT_COST: tokens per ms, a finite number 0 or more (0 or off: no cost stop), "
                         f"not {raw!r}")
    return cost


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
    ms = _extended(verify, count + 3)
    return [cost * (ms[j + 1] - ms[j] + dms) for j in range(count + 1)]


def _extended(verify, n: int) -> list[float]:
    """The table's first ``n`` windows (1..n rows); past its end, its last step again."""

    ms = list(verify)
    while len(ms) < n:
        ms.append(ms[-1] + (ms[-1] - ms[-2]))
    return ms


def window_ms(verify, rows: int) -> float:
    """A verify window of ``rows`` rows (1 or more), ms, by the table (extended past its end by its last step)."""

    return _extended(verify, rows)[rows - 1]


# House: joint draft pricing for --parallel's shared rounds (TF_JOINT_PRICE=1). Every live stream's window is one
# forward, so a stream's next draft costs the round's marginal row at the round's total rows, and repays at the
# round's aggregate rate (tokens a ms over all streams), not at one stream's. Speed only: drafts never change output.
JOINT_ROWS = 4          # a stream's typical window in a shared round (the pending row and ~3 drafts): the rate's guess


def joint_settings(env=os.environ) -> dict | None:
    """``TF_JOINT_PRICE=1`` turns joint pricing on (default off); ``TF_JOINT_RATE`` (tokens per ms over all streams)
    fixes the aggregate rate instead of the table's estimate; ``TF_JOINT_VERIFY_MS`` (comma-separated ms of shared
    windows of 1, 2, ... total rows) replaces the one-stream table for the round's row cost. None when off."""

    flag = env.get("TF_JOINT_PRICE", "").strip()
    if flag not in ("", "0", "1"):
        raise ValueError(f"TF_JOINT_PRICE: 0 or 1, not {flag!r}")
    if flag != "1":
        return None
    rate = None
    if env.get("TF_JOINT_RATE", "").strip():
        try:
            rate = float(env["TF_JOINT_RATE"])
        except ValueError:
            rate = math.nan
        if not math.isfinite(rate) or rate <= 0:              # nan / inf priced every draft as free or never
            raise ValueError(f"TF_JOINT_RATE: tokens per ms over all streams, more than 0, not {env['TF_JOINT_RATE']!r}")
    verify = None
    if env.get("TF_JOINT_VERIFY_MS", "").strip():
        try:
            verify = tuple(float(x) for x in env["TF_JOINT_VERIFY_MS"].split(","))
        except ValueError:
            verify = ()
        if (len(verify) < 2 or not all(math.isfinite(x) for x in verify)
                or any(b < a for a, b in zip(verify, verify[1:])) or verify[0] <= 0):
            raise ValueError("TF_JOINT_VERIFY_MS: at least two positive, finite, non-decreasing window times (ms)")
    return {"rate": rate, "verify": verify}


def joint_rate(cost: float, streams: int, verify, shared=None) -> float:
    """The aggregate rate a shared round of ``streams`` streams is priced at, from the one-stream rate ``cost``: n
    streams emit about n times one stream's tokens in a window of n x JOINT_ROWS rows instead of JOINT_ROWS, so
    cost x n x V(JOINT_ROWS) / V'(n x JOINT_ROWS), V' the shared table (default V). One stream: ``cost`` itself."""

    n = max(1, int(streams))
    if n == 1:
        return cost
    return cost * n * window_ms(verify, JOINT_ROWS) / window_ms(shared or verify, n * JOINT_ROWS)


class JointBars:
    """A shared round's draft bars: the running product a stream's next draft needs is ``rate`` x (the round's
    marginal verify row at its current total rows + one MTP step). ``rows`` counts the next round's window: each
    live stream's pending row plus every draft kept so far; ``take`` adds a kept draft. With one stream, rate =
    cost and rows = 1 + drafts, this is ``cost_bars`` exactly."""

    def __init__(self, rate: float, verify, dms: float, rows: int) -> None:
        self.rate, self.dms, self.rows = rate, dms, int(rows)
        self.verify = tuple(verify)

    def bar(self, ahead: int = 0) -> float:
        """The bar of a draft added at ``rows + ahead`` rows."""

        r = max(1, self.rows + ahead)
        return self.rate * (window_ms(self.verify, r + 1) - window_ms(self.verify, r) + self.dms)

    def take(self) -> None:
        self.rows += 1


def joint_bars(settings: dict, cost: float, timing, streams: int, rows: int) -> JointBars:
    """The round's bars from ``joint_settings``, the one-stream ``cost`` and table ``timing`` ((verify, draft ms))."""

    verify, dms = timing if timing is not None else (VERIFY_MS, DRAFT_MS)
    shared = settings.get("verify") or verify
    rate = settings.get("rate") or joint_rate(cost, streams, verify, settings.get("verify"))
    return JointBars(rate, shared, dms, rows)
