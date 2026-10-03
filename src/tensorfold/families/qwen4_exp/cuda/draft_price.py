"""--mtp-cost: price a later MTP draft by its calibrated chance of being kept against the ms it adds to the round;
--mtp-lookahead: draft toward the depth that gives the most expected tokens per ms of the whole round."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

RATE = 0.05          # EMA weight of one round's observation


class DraftPrice:
    """The measured round costs and the head's chain product calibrated per depth by the kept/drafted counts of live
    rounds (greedy and sampled apart). ``lookahead`` replaces the marginal bar at ``cost`` by the round's best depth."""

    def __init__(self, cost: float, verify_ms: Sequence[float], draft_ms: float, depth: int,
                 lookahead: bool = False) -> None:
        ms = list(verify_ms)
        while len(ms) < depth + 3:                  # windows past the measured table repeat its last step
            ms.append(ms[-1] + (ms[-1] - ms[-2]) if len(ms) > 1 else ms[-1])
        self.cost, self.verify, self.draft_ms, self.depth = float(cost), ms, float(draft_ms), int(depth)
        self.kept = {False: [None] * depth, True: [None] * depth}   # EMA of "drafts 1..j all kept"
        self.said = {False: [None] * depth, True: [None] * depth}   # EMA of the head's product for those drafts
        self.sampled, self.lookahead, self.limit = False, bool(lookahead), int(depth)
        self.products: list[float] = []

    def adds(self, j: int) -> float:
        """ms that verifying draft j (0-based) adds to the round: one more verify row and its MTP step."""

        return self.verify[j + 1] - self.verify[j] + self.draft_ms

    def chance(self, j: int, product: float) -> float:
        kept, said = self.kept[self.sampled][j], self.said[self.sampled][j]
        if kept is None or said <= 0:
            return product
        return min(1.0, product * kept / said)

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

        self.sampled, self.products = sampled, []
        self.limit = self.depth if limit is None else min(int(limit), self.depth)

    def observe(self, verified: int, kept: int) -> None:
        """One verified round: of its ``verified`` drafts the first ``kept`` were kept."""

        k, s = self.kept[self.sampled], self.said[self.sampled]
        for j, p in enumerate(self.products[:verified]):
            if k[j] is None:                        # seeded by the head's product: scale 1 until rounds say more
                k[j] = s[j] = p
            k[j] += RATE * (float(j < kept) - k[j])
            s[j] += RATE * (p - s[j])

    def describe(self) -> str:
        v = self.verify
        ahead = "look-ahead over every depth; " if self.lookahead else ""
        return (f"{ahead}verify {v[0]:.1f}-{v[self.depth]:.1f} ms for 1-{self.depth + 1} rows, a draft step "
                f"{self.draft_ms:.2f} ms")


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
