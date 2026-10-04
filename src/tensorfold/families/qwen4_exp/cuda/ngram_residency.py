"""The n-gram tables' page residency after start-up (TF_NGRAM_ADVICE, TF_NGRAM_LOCK, TF_NGRAM_REFRESH): same bytes."""

from __future__ import annotations

import threading
import time

from ..host_residency import GIB, apply_lock, mem_available, mem_total, planned_lock, residency_policy


class Residency:
    """The engine's residency hooks, read before the load (bad values refuse to start); all off by default.

    TF_NGRAM_REPIN=1 (default; 0 off): with TF_NGRAM_LOCK under --parallel, growing stream caches unpin 1 GiB table
    runs (``release``); a background thread pins them back, one run every ``PERIOD`` s at most, while no prompt is
    prefilling, no release happened in the last ``COOLDOWN`` s and MemAvailable stays ``MARGIN`` above the startup
    floor after the run, up to the startup lock's budget. Residency only: no gathered byte changes."""

    PERIOD = 5.0             # s between re-pinned runs (<= 1 GiB per 5 s)
    COOLDOWN = 30.0          # s after a release before re-pinning again
    MARGIN = 2 * GIB         # hysteresis: release fires at the floor, re-pin stops MARGIN above it
    clock = staticmethod(time.monotonic)

    def __init__(self, env) -> None:
        self.policy = residency_policy(env)
        self.advice = env.get("TF_NGRAM_ADVICE", "")
        if self.advice not in ("", "random", "normal"):
            raise ValueError(f"TF_NGRAM_ADVICE: random or normal, not {self.advice!r}")
        self.tables: list = []
        self._refreshing = None
        repin = env.get("TF_NGRAM_REPIN", "") or "1"
        if repin not in ("0", "1"):
            raise ValueError(f"TF_NGRAM_REPIN: 0 or 1, not {repin!r}")
        self.repin = repin == "1"
        self.target = 0              # bytes re-pinning restores up to (0: no re-pin)
        self.floor = 0               # the startup lock's MemAvailable floor
        self.busy = None             # () -> True while a prompt is prefilling (re-pin waits)
        self.repinned = 0
        self._guard = threading.Lock()
        self._last_release = float("-inf")
        self._last_repin = float("-inf")
        self._stop = threading.Event()
        self._repinning = None

    @property
    def owns_startup(self) -> bool:
        """A lock or refresh policy (or TF_NGRAM_REFRESH=0) replaces the default read-back after warm-up."""

        return bool(self.policy["lock"] or self.policy["refresh"]) or not self.policy["startup"]

    def advise(self, w, ple_on_ssd: bool) -> str:
        """TF_NGRAM_ADVICE on the maps lookups read; the startup line's note."""

        if not self.advice or ple_on_ssd:
            return ""
        for table in _unique(w).values():
            if hasattr(table, "advise"):
                table.advise(self.advice)
        return f", {self.advice} access"

    def start(self, w, ple_on_ssd: bool, multi) -> str:
        """After the warm-up's allocations: pin (TF_NGRAM_LOCK) and read back (TF_NGRAM_REFRESH); the startup note."""

        self.tables = [] if ple_on_ssd else [t for t in _unique(w).values() if hasattr(t, "lock_parts")]
        policy = self.policy
        if not self.tables or not (policy["lock"] or policy["refresh"]):
            return ""
        if multi is not None and policy["lock"] in ("auto", "budget"):
            policy = dict(policy, lock_reserve=max(policy["lock_reserve"], gate_floor(multi)))
        before = mem_available()
        if policy["lock"] == "budget":           # one budget across the tables, in the same order every load
            pinned = 0
            for t in self.tables:
                pinned += apply_lock(t, policy, policy["budget"] - pinned)
        else:
            pinned = sum(apply_lock(t, policy) for t in self.tables)
        self.pinned = pinned
        if multi is not None and policy["lock"]:
            multi.release = self.release          # a growing stream cache unpins table runs before the gate refuses
            if self.repin:                        # ... and caches that shrink give them back (TF_NGRAM_REPIN)
                self.names = policy["lock"]
                self.target = planned_lock(self.tables, policy["budget"]) if policy["lock"] == "budget" else pinned
                self.floor = policy["lock_reserve"]
                self.busy = lambda: bool(getattr(multi, "filling", None))
                self._repinning = threading.Thread(target=self._repin_loop, name="ngram-repin", daemon=True)
                self._repinning.start()
        # a budget also reads the unpinned rest back once (unless TF_NGRAM_REFRESH=0): what survived the load's reads
        # in the page cache varied 29.6-36.4 GiB between loads, and decode with it (59.8 vs 66.1 tok/s at 27 pinned)
        once = policy["lock"] == "budget" and policy["startup"] and not policy["refresh"]
        asked = sum(t.refresh(policy["reserve"]) for t in self.tables) if policy["refresh"] or once else 0
        return (_lock_note(policy, pinned, before, self.tables) if policy["lock"] else "") + (
            f", {asked / 2**30:.1f} GiB re-read, refreshed after each request" if policy["refresh"] else
            f", the rest read back ({asked / 2**30:.1f} GiB asked)" if once else "")

    def release(self, nbytes: int) -> int:
        """Unpin table runs until ``nbytes`` are released (or none is left); bytes released."""

        freed = 0
        with self._guard:
            for t in self.tables:
                if freed >= nbytes:
                    break
                if hasattr(t, "release"):
                    freed += t.release(nbytes - freed)
            self._last_release = self.clock()
        return freed

    def locked(self) -> int:
        with self._guard:
            return sum(t.locked_bytes() for t in self.tables if hasattr(t, "locked_bytes"))

    def repin_step(self, now: float | None = None) -> int:
        """Pin back one released run if every condition holds (see the class note); bytes re-pinned."""

        now = self.clock() if now is None else now
        if not self.target or now - self._last_release < self.COOLDOWN or now - self._last_repin < self.PERIOD:
            return 0
        if self.busy is not None and self.busy():
            return 0
        held = self.locked()
        if held >= self.target:
            return 0
        afford = mem_available() - self.floor - self.MARGIN          # what one run may take from MemAvailable
        for t in self.tables:
            if not hasattr(t, "lock_next"):
                continue
            names = t.DENSE if self.names == "dense" else tuple(t.parts())
            mine = t.locked_bytes()
            got = t.lock_next(names, mine + min(self.target - held, afford), self._guard)
            if got:
                self._last_repin = now
                self.repinned += got
                print(f"[tensorfold] memory: re-pinned {got / GIB:.2f} GiB of the n-gram table "
                      f"(now {self.locked() / GIB:.2f} GiB locked, {self.repinned / GIB:.2f} GiB re-pinned so far)",
                      flush=True)
                return got
        self._last_repin = now                   # nothing fit (memory or budget): look again a period later
        return 0

    def _repin_loop(self) -> None:
        while not self._stop.wait(self.PERIOD):
            try:
                self.repin_step()
            except Exception as exc:              # residency only: a failure must never reach a request
                print(f"[tensorfold] memory: re-pin stopped: {exc}", flush=True)
                return

    def stop(self) -> None:
        """Stop the re-pin thread (tests start several engines)."""

        self._stop.set()

    def after_request(self) -> None:
        """TF_NGRAM_REFRESH=1: between requests, on a thread, ask for evicted pages back (``_Residency.refresh``)."""

        if not self.tables or not self.policy["refresh"]:
            return
        if self._refreshing is not None and self._refreshing.is_alive():
            return
        reserve = self.policy["reserve"]
        self._refreshing = threading.Thread(target=lambda: [t.refresh(reserve) for t in self.tables],
                                            name="ngram-refresh", daemon=True)
        self._refreshing.start()


def _lock_note(policy: dict, pinned: int, before: int, tables) -> str:
    """The startup line's lock note. A budget says whether it was met (exactly the runs the budget alone pins); where
    MemAvailable less the floor was too small (or mlock refused) it says CLIPPED: that load's speed follows the host
    again and is not comparable with others."""

    note = f", {pinned / GIB:.1f} GiB locked ({policy['lock']}"
    if policy["lock"] != "budget":
        return note + ")"
    met = "met" if pinned == planned_lock(tables, policy["budget"]) else "CLIPPED"
    return (note + f" {policy['budget'] / GIB:g} GiB, {met}; MemAvailable {before / GIB:.1f} GiB before it, "
            f"floor {policy['lock_reserve'] / GIB:.1f} GiB)")


def gate_floor(multi) -> int:
    """The MemAvailable --parallel's gate needs for every slot's first growth (TF_NGRAM_LOCK=auto pins down to it)."""

    from tensorfold.cuda.capacity import reserve_bytes

    return reserve_bytes(mem_total()) + multi.memory_gate.reserve + multi.planned_growth()


def _unique(w) -> dict:
    return {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
