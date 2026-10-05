"""--mtp-cost: price a later MTP draft by its calibrated chance of being kept against the ms it adds to the round;
--mtp-lookahead: draft toward the depth that gives the most expected tokens per ms of the whole round;
--mtp-live-cost: re-price either stop from the rounds' own times as decoding goes;
--mtp-calibration depth-confidence: calibrate a sampled draft's chance per depth and per its own probability's bucket."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

RATE = 0.05          # EMA weight of one round's observation
PRIOR = 0.01         # weight the startup table keeps under --mtp-live-cost (the live rounds' weight grows to 1)
BUCKETS = (0.5, 0.7, 0.9)   # --mtp-calibration depth-confidence: a draft's own probability in [0, .5), [.5, .7), ...
SHRINK = 20.0        # rounds a bucket needs before its ratio outweighs its depth's (a bucket with n rounds: n / (n + 20))
CALIBRATIONS = ("depth", "depth-confidence")


def bucket(p: float) -> int:
    """The confidence bucket of a draft whose own (temperature-1) probability is ``p``."""

    return sum(p >= b for b in BUCKETS)


class DraftPrice:
    """The measured round costs and the head's chain product calibrated per depth by the kept/drafted counts of live
    rounds (greedy and sampled apart). ``lookahead`` replaces the marginal bar at ``cost`` by the round's best depth.
    ``live`` corrects the startup prices by a line in the round's draft count, fitted to the rounds' measured times.
    ``calibration="depth-confidence"`` keeps the same kept/said ratio per depth and per bucket of the draft's own
    probability, and uses it for drafts already sampled, shrunk toward the depth's ratio while a bucket has few rounds."""

    def __init__(self, cost: float, verify_ms: Sequence[float], draft_ms: float, depth: int,
                 lookahead: bool = False, live: bool = False, calibration: str = "depth") -> None:
        if calibration not in CALIBRATIONS:
            raise ValueError(f"MTP calibration: one of {', '.join(CALIBRATIONS)}, not {calibration!r}")
        ms = list(verify_ms)
        while len(ms) < depth + 3:                  # windows past the measured table repeat its last step
            ms.append(ms[-1] + (ms[-1] - ms[-2]) if len(ms) > 1 else ms[-1])
        self.cost, self.verify, self.draft_ms, self.depth = float(cost), ms, float(draft_ms), int(depth)
        self.kept = {False: [None] * depth, True: [None] * depth}   # EMA of "drafts 1..j all kept"
        self.said = {False: [None] * depth, True: [None] * depth}   # EMA of the head's product for those drafts
        self.sampled, self.lookahead, self.limit = False, bool(lookahead), int(depth)
        self.products: list[float] = []
        self.table, self.live = tuple(ms), bool(live)   # the startup prices; ``verify`` follows the live line
        self.sums = [0.0] * 5                       # EMA of 1, x, x^2, y, xy over timed rounds: x drafts, y live - table
        self.rounds = 0
        self.calibration, self.confs = calibration, []    # the drafts' own probabilities, this chain
        n = len(BUCKETS) + 1
        self.bkept = {m: [[None] * n for _ in range(depth)] for m in (False, True)}   # as kept/said, per bucket
        self.bsaid = {m: [[None] * n for _ in range(depth)] for m in (False, True)}
        self.bseen = {m: [[0] * n for _ in range(depth)] for m in (False, True)}      # rounds in each bucket

    def table_ms(self, drafts: int) -> float:
        """The startup table's ms for a round of ``drafts`` verified drafts: its window plus their MTP steps."""

        return self.table[drafts] + drafts * self.draft_ms

    def line(self) -> tuple[float, float]:
        """The live correction, ms = a + b x drafts over the table: least squares on the EMA of timed rounds, with the
        table itself (no correction at 0 .. depth drafts) kept at weight ``PRIOR`` so one depth cannot tilt it alone."""

        n, sx, sxx, sy, sxy = self.sums
        xs = range(self.depth + 1)
        n += PRIOR
        sx += PRIOR * sum(xs) / len(xs)
        sxx += PRIOR * sum(x * x for x in xs) / len(xs)
        det = n * sxx - sx * sx
        b = (n * sxy - sx * sy) / det if det > 1e-12 else 0.0
        b = max(b, -0.5 * (self.table_ms(self.depth) - self.table_ms(0)) / max(1, self.depth))  # drafts never free
        return (sy - b * sx) / n, b

    def timed(self, drafts: int, ms: float) -> None:
        """One round that verified ``drafts`` drafts took ``ms`` (host clock, between the reads that end two rounds:
        its MTP steps, verify window, sampling and commit); with ``live`` the prices move toward such rounds."""

        if not self.live or drafts >= len(self.table):
            return
        table = self.table_ms(drafts)
        y = min(max(ms - table, -0.5 * table), 2 * table)   # a stall (a paused host) moves the line a bounded step
        for i, v in enumerate((1.0, drafts, drafts * drafts, y, drafts * y)):
            self.sums[i] += RATE * (v - self.sums[i])
        self.rounds += 1
        a, b = self.line()
        self.verify = [v + a + b * d for d, v in enumerate(self.table)]

    def adds(self, j: int) -> float:
        """ms that verifying draft j (0-based) adds to the round: one more verify row and its MTP step."""

        return self.verify[j + 1] - self.verify[j] + self.draft_ms

    def ratio(self, j: int) -> float:
        """Depth j's calibration: kept over the head's product, 1 before rounds reach it."""

        kept, said = self.kept[self.sampled][j], self.said[self.sampled][j]
        return 1.0 if kept is None or said <= 0 else kept / said

    def multiplier(self, j: int, conf: float | None = None) -> float:
        """The factor on the head's chain product for draft j; with depth-confidence and the draft's own probability
        ``conf``, its bucket's ratio shrunk toward the depth's: (n x bucket + SHRINK x depth) / (n + SHRINK)."""

        depth = self.ratio(j)
        if self.calibration == "depth" or conf is None:
            return depth
        m, b = self.sampled, bucket(conf)
        kept, said, n = self.bkept[m][j][b], self.bsaid[m][j][b], self.bseen[m][j][b]
        if kept is None or said <= 0:
            return depth
        return (n * kept / said + SHRINK * depth) / (n + SHRINK)

    def chance(self, j: int, product: float) -> float:
        conf = self.confs[j] if j < len(self.confs) else None     # sampled drafts carry their own probability
        return min(1.0, product * self.multiplier(j, conf))

    def pays(self, j: int, product: float) -> bool:
        """Whether draft j, at head product ``product``, repays the ms it adds at ``cost`` tokens per ms."""

        return self.chance(j, product) >= self.cost * self.adds(j)

    def best(self, products: Sequence[float], lo: int) -> int:
        """The draft count from ``lo`` up with the most expected tokens per ms of the round, given the head's chain
        products of the drafts sampled so far; deeper drafts survive at the depth's calibrated rate of live rounds
        (kept 1..j+1 over kept 1..j), or at the last draft's own rate before rounds reach that depth."""

        k = self.kept[self.sampled]
        survival, tokens, last, best, score = 1.0, 1.0, 1.0, lo, -1.0
        for d in range(1, self.limit + 1):
            if d <= len(products):
                step = min(1.0, self.chance(d - 1, products[d - 1]) / survival) if survival > 0 else 0.0
                last = step
            elif d > 1 and k[d - 1] is not None and k[d - 2] is not None and k[d - 2] > 0:
                step = min(1.0, k[d - 1] / k[d - 2])
            else:
                step = last
            survival *= step
            tokens += survival
            ms = self.verify[d] + max(d, len(products)) * self.draft_ms   # drafts already sampled are spent
            if d >= lo and tokens / ms > score:
                best, score = d, tokens / ms
        return best

    def keeps(self, j: int, product: float) -> bool:
        """Whether draft j (0-based), just sampled at chain product ``product``, is worth its verify row."""

        if not self.lookahead:
            return self.pays(j, product)
        return j == 0 or self.best([*self.products[:j], product], j) > j

    def more(self, j: int, product: float) -> bool:
        """Whether to spend an MTP step on draft j + 1 after keeping draft j at chain product ``product``."""

        if not self.lookahead:
            return self.pays(j + 1, product)
        return self.best([*self.products[:j], product], j + 1) > j + 1

    def begin(self, sampled: bool, limit: int | None = None) -> None:
        """A new chain of at most ``limit`` drafts (the depth by default)."""

        self.sampled, self.products, self.confs = sampled, [], []
        self.limit = self.depth if limit is None else min(int(limit), self.depth)

    def observe(self, verified: int, kept: int) -> None:
        """One verified round: of its ``verified`` drafts the first ``kept`` were kept."""

        m = self.sampled
        k, s = self.kept[m], self.said[m]
        for j, p in enumerate(self.products[:verified]):
            if j < len(self.confs):                 # each bucket's ratio, seeded at its depth's
                b = bucket(self.confs[j])
                bk, bs = self.bkept[m][j], self.bsaid[m][j]
                if bk[b] is None:
                    bk[b], bs[b] = p * self.ratio(j), p
                bk[b] += RATE * (float(j < kept) - bk[b])
                bs[b] += RATE * (p - bs[b])
                self.bseen[m][j][b] += 1
            if k[j] is None:                        # seeded by the head's product: scale 1 until rounds say more
                k[j] = s[j] = p
            k[j] += RATE * (float(j < kept) - k[j])
            s[j] += RATE * (p - s[j])

    def describe(self) -> str:
        v = self.verify
        ahead = "look-ahead over every depth; " if self.lookahead else ""
        if self.calibration != "depth":
            ahead += f"chances calibrated per depth and draft probability ({', '.join(map(str, BUCKETS))}); "
        if self.live:
            ahead += "re-priced by live round times from: "
        return (f"{ahead}verify {v[0]:.1f}-{v[self.depth]:.1f} ms for 1-{self.depth + 1} rows, a draft step "
                f"{self.draft_ms:.2f} ms")


    def fits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(intercept, ms a draft) of the startup table's rounds and of the current prices, over 0 .. depth drafts."""

        xs = np.arange(self.depth + 1)
        table = [self.table_ms(int(d)) for d in xs]
        now = [self.verify[int(d)] + d * self.draft_ms for d in xs]
        (bt, at), (bn, an) = np.polyfit(xs, table, 1), np.polyfit(xs, now, 1)
        return (float(at), float(bt)), (float(an), float(bn))


@torch.no_grad()
def measure_round_costs(e, rows: int, reps: int = 5) -> tuple[tuple[float, ...], float]:
    """This engine's verify windows of 1 .. ``rows`` rows and one MTP draft step, ms (medians, after a short prompt),
    on distinct tokens (a window's cost grows with the distinct experts its rows route to); leaves the state empty."""

    from .decode import prefill

    gen = torch.Generator().manual_seed(0)
    lo = min(1000, e.w.cfg.vocab // 4)
    ids = torch.randint(lo, max(lo + 1, e.w.cfg.vocab // 2), (64 + rows,), generator=gen).tolist()
    prefill(e, ids[:64], None)
    tokens = ids[64:]

    def timed(fn) -> float:
        times = []
        for _ in range(reps + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times[1:]))

    verify = [timed(lambda r=r: e.forward(tokens[:r])) for r in range(1, rows + 1)]
    for r in range(1, rows):                       # a wider window never costs less: noise must not reorder bars
        verify[r] = max(verify[r], verify[r - 1])
    step = timed(lambda: e.mtp_forward(tokens[:1], e.last_streams)) if e.mbuf is not None else 0.0
    e.kept = None
    e.reset()
    return tuple(round(v, 3) for v in verify), round(step, 3)
