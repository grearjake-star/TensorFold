"""The n-gram tables' page residency after start-up (TF_NGRAM_ADVICE, TF_NGRAM_LOCK, TF_NGRAM_REFRESH): same bytes."""

from __future__ import annotations

import threading

from ..host_residency import apply_lock, mem_total, residency_policy


class Residency:
    """The engine's residency hooks, read before the load (bad values refuse to start); all off by default."""

    def __init__(self, env) -> None:
        self.policy = residency_policy(env)
        self.advice = env.get("TF_NGRAM_ADVICE", "")
        if self.advice not in ("", "random", "normal"):
            raise ValueError(f"TF_NGRAM_ADVICE: random or normal, not {self.advice!r}")
        self.tables: list = []
        self._refreshing = None

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
        if multi is not None and policy["lock"] == "auto":
            policy = dict(policy, lock_reserve=max(policy["lock_reserve"], gate_floor(multi)))
        pinned = sum(apply_lock(t, policy) for t in self.tables)
        if multi is not None and policy["lock"]:
            multi.release = self.release          # a growing stream cache unpins table runs before the gate refuses
        asked = sum(t.refresh(policy["reserve"]) for t in self.tables) if policy["refresh"] else 0
        return (f", {pinned / 2**30:.1f} GiB locked ({policy['lock']})" if policy["lock"] else "") + (
            f", {asked / 2**30:.1f} GiB re-read, refreshed after each request" if policy["refresh"] else "")

    def release(self, nbytes: int) -> int:
        """Unpin table runs until ``nbytes`` are released (or none is left); bytes released."""

        freed = 0
        for t in self.tables:
            if freed >= nbytes:
                break
            if hasattr(t, "release"):
                freed += t.release(nbytes - freed)
        return freed

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


def gate_floor(multi) -> int:
    """The MemAvailable --parallel's gate needs for every slot's first growth (TF_NGRAM_LOCK=auto pins down to it)."""

    from tensorfold.cuda.capacity import reserve_bytes

    return reserve_bytes(mem_total()) + multi.memory_gate.reserve + multi.planned_growth()


def _unique(w) -> dict:
    return {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
