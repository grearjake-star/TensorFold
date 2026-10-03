"""Page residency of host n-gram tables: partial mlock, mincore, WILLNEED refresh and the TF_NGRAM_* policy."""

from __future__ import annotations

import os

import numpy as np

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
GIB = 1 << 30


def residency_policy(env) -> dict:
    """The n-gram residency hooks (all off by default; none changes a gathered byte):
    TF_NGRAM_LOCK=dense       mlock the dense components (MLX: scales and biases, 6 GiB of the 29.8 and two thirds
                              of the pages a lookup reads; NVFP4: the block scales)
    TF_NGRAM_LOCK=auto[:GiB]  dense first, then whole shards of the rest while MemAvailable stays above the reserve
    TF_NGRAM_LOCK=all         every table page, whatever the admission's room (a memory-budget decision)
    TF_NGRAM_LOCK=<GiB>       a fixed budget: the same leading runs (dense first) on every load, so speed does not
                              follow MemAvailable at start-up; clipped (and the startup line says so) only where
                              pinning it would leave MemAvailable under the floor auto stops at
    TF_NGRAM_REFRESH=1        after start-up and after each request, read evicted pages back (MADV_WILLNEED) while
                              MemAvailable stays above the reserve
    (unset)                   the engine's own read-back after warm-up (and its run pins within the startup room)
    TF_NGRAM_REFRESH=0        no read-back at all
    TF_NGRAM_RESERVE_GIB=10   that reserve (the host's MemAvailable floor; auto:GiB overrides it for the lock)."""

    lock = env.get("TF_NGRAM_LOCK", "") or None
    refresh = env.get("TF_NGRAM_REFRESH", "") or None
    try:
        reserve = float(env.get("TF_NGRAM_RESERVE_GIB", "") or 10)
    except ValueError:
        raise ValueError(f"TF_NGRAM_RESERVE_GIB: GiB, not {env.get('TF_NGRAM_RESERVE_GIB')!r}") from None
    lock_reserve, budget = reserve, None
    if lock is not None and lock not in ("dense", "auto", "all") and not lock.startswith("auto:"):
        try:
            budget = float(lock)
        except ValueError:
            raise ValueError(f"TF_NGRAM_LOCK: dense, auto, auto:<GiB>, all or <GiB>, not {lock!r}") from None
        if not budget > 0 or budget != budget or budget == float("inf"):
            raise ValueError(f"TF_NGRAM_LOCK: a budget is a positive number of GiB, not {lock!r}")
        lock = "budget"
    if lock is not None and lock.startswith("auto:"):
        try:
            lock_reserve = float(lock[5:])
        except ValueError:
            raise ValueError(f"TF_NGRAM_LOCK: dense, auto, auto:<GiB>, all or <GiB>, not {lock!r}") from None
        lock = "auto"
    if lock not in (None, "dense", "auto", "all", "budget"):
        raise ValueError(f"TF_NGRAM_LOCK: dense, auto, auto:<GiB>, all or <GiB>, not {lock!r}")
    if refresh not in (None, "0", "1"):
        raise ValueError(f"TF_NGRAM_REFRESH: 0 or 1, not {refresh!r}")
    if reserve < 0 or lock_reserve < 0:
        raise ValueError("TF_NGRAM_RESERVE_GIB: not negative")
    return {"lock": lock, "lock_reserve": int(lock_reserve * GIB), "budget": None if budget is None else int(budget * GIB),
            "refresh": refresh == "1",
            "reserve": int(reserve * GIB), "startup": refresh != "0"}


def apply_lock(table: _Residency, policy: dict, budget: int | None = None) -> int:
    """TF_NGRAM_LOCK on one table: bytes locked. A budget lock pins ``budget`` bytes (default: the policy's) of whole
    runs in ``parts()`` order, the same runs on every load, unless MemAvailable less ``lock_reserve`` is smaller."""

    mode = policy["lock"]
    if mode == "dense":
        return table.lock_parts(table.DENSE)
    if mode == "all":
        return table.lock_parts(tuple(table.parts()))
    if mode == "auto":
        already = sum(n for _, n in getattr(table, "_locked", []))
        return table.lock_parts(tuple(table.parts()), already + max(0, mem_available() - policy["lock_reserve"]))
    if mode == "budget":
        already = sum(n for _, n in getattr(table, "_locked", []))
        want = planned_lock([table], policy["budget"] if budget is None else budget)
        return table.lock_parts(tuple(table.parts()), min(already + want,
                                                          already + max(0, mem_available() - policy["lock_reserve"])))
    return 0


def planned_lock(tables, budget: int) -> int:
    """Bytes a ``budget`` lock pins on ``tables`` with room to spare: whole runs in ``parts()`` order, stopping at the
    first that does not fit; a fixed quantity of the checkpoint and the budget. Each run counted may overrun by its
    two edge pages (a page span is a little over the run's bytes), so N GiB pins N of EXL3's 1 GiB runs, not N - 1."""

    total = runs = 0
    for table in tables:
        for name in table.parts():
            for arr in table.parts()[name]:
                size = _span(arr)[1]
                runs += 1
                if total + size > budget + 2 * PAGE * runs:
                    return total
                total += size
    return total


def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    return libc


def _span(arr: np.ndarray) -> tuple[int, int]:
    """An array's pages: (page-aligned address, bytes) covering it (inside its memory map)."""

    at = arr.ctypes.data
    start = at - at % PAGE
    end = at + arr.nbytes
    return start, -(-(end - start) // PAGE) * PAGE


def _mincore(at: int, size: int) -> np.ndarray:
    """One byte a page, bit 0 set where the page is in memory (read-only, touches nothing)."""

    import ctypes

    vec = np.zeros(size // PAGE, dtype=np.uint8)
    if _libc().mincore(at, size, vec.ctypes.data_as(ctypes.c_void_p)) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return vec


def mem_available() -> int:
    """MemAvailable in bytes (0 where /proc/meminfo is missing)."""

    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def mem_total() -> int:
    """MemTotal in bytes (0 where /proc/meminfo is missing)."""

    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0



class _Residency:
    """Page residency of a table's shard arrays (B: the fork's B measurements): partial mlock, mincore, WILLNEED refresh.
    None of these changes a byte that ``gather`` returns. ``parts()`` lists the arrays by component, densest first
    (the components a lookup reads one short row of, so the most lookups a page); ``DENSE`` names those."""

    DENSE: tuple[str, ...] = ()

    def parts(self) -> dict[str, list[np.ndarray]]:
        raise NotImplementedError

    def lock_parts(self, names: tuple[str, ...], budget: int | None = None) -> int:
        """Best effort: mlock whole shard arrays of ``names`` in that order while ``budget`` bytes last (None: no
        bound); an array the memory-lock limit refuses is skipped. Bytes locked; ``unlock`` undoes it."""

        libc = _libc()
        locked = getattr(self, "_locked", [])
        held = {a for a, _ in locked}
        total = sum(n for _, n in locked)
        for name in names:
            for arr in self.parts()[name]:
                at, size = _span(arr)
                if at in held:
                    continue
                if budget is not None and total + size > budget:
                    self._locked = locked
                    return total
                if libc.mlock(at, size) == 0:
                    locked.append((at, size))
                    held.add(at)
                    total += size
        self._locked = locked
        return total

    def locked_bytes(self) -> int:
        """Bytes ``lock_parts`` holds pinned now."""

        return sum(n for _, n in getattr(self, "_locked", []))

    def release(self, nbytes: int) -> int:
        """munlock locked arrays, the last locked first (``lock_parts`` pins the most useful first), until at least
        ``nbytes`` are released or none is left; bytes released. The pages stay in the page cache, now reclaimable:
        gathers return the same bytes, at worst a later lookup faults one back from disk."""

        libc = _libc()
        locked = getattr(self, "_locked", [])
        freed = 0
        while locked and freed < nbytes:
            at, size = locked.pop()
            libc.munlock(at, size)
            freed += size
        self._locked = locked
        return freed

    def unlock(self) -> None:
        """munlock what ``lock_parts`` locked."""

        libc = _libc()
        for at, size in getattr(self, "_locked", []):
            libc.munlock(at, size)
        self._locked = []

    def residency(self) -> dict[str, list[int]]:
        """Read-only (mincore): [resident bytes, total bytes] per component."""

        out = {}
        for name, arrays in self.parts().items():
            have = total = 0
            for arr in arrays:
                bits = _mincore(*_span(arr))
                have += int(np.count_nonzero(bits & 1)) * PAGE
                total += len(bits) * PAGE
            out[name] = [have, total]
        return out

    def refresh(self, reserve: int, chunk: int = 512) -> int:
        """Ask the kernel to read evicted table pages back (MADV_WILLNEED, read asynchronously) on every ``chunk``
        pages (2 MiB) in which mincore finds a page missing, only while MemAvailable stays above ``reserve`` bytes
        afterwards: evicted pages come back after a passing squeeze without pushing the host below its reserve.
        Chunks, not single-page runs: eviction leaves ~1M scattered holes, and a call per hole would hold the GIL
        against the next request's decode for seconds. Bytes requested (missing pages only)."""

        import mmap

        libc = _libc()
        room = mem_available() - reserve
        asked = 0
        for arrays in self.parts().values():
            for arr in arrays:
                at, size = _span(arr)
                missing = (_mincore(at, size) & 1) == 0
                if not missing.any():
                    continue
                pages = len(missing)
                per = np.add.reduceat(missing.astype(np.int64), np.arange(0, pages, chunk))
                for c in np.flatnonzero(per):
                    need = int(per[c]) * PAGE
                    if asked + need > room:
                        return asked
                    first = int(c) * chunk
                    libc.madvise(at + first * PAGE, min(chunk, pages - first) * PAGE, mmap.MADV_WILLNEED)
                    asked += need
        return asked


def _advise(arrays: list[np.ndarray], advice: str) -> int:
    """madvise the maps under ``arrays`` (bytes advised): "random" turns off read-ahead on their page faults, so a
    lookup whose rows were evicted reads its pages alone rather than a read-around window each (fewer bytes read,
    less page cache displaced); "normal" restores the kernel's default read-around."""

    import mmap as _mmap

    flags = {"random": _mmap.MADV_RANDOM, "normal": _mmap.MADV_NORMAL}
    if advice not in flags:
        raise ValueError(f"TF_NGRAM_ADVICE: random or normal, not {advice!r}")
    advised = 0
    for array in arrays:
        array._mmap.madvise(flags[advice])              # type: ignore[attr-defined]
        advised += len(array._mmap)                     # type: ignore[attr-defined]
    return advised
