"""TF_PIN_REWARM (H2c; on with TF_PIN_BASE, 0 turns only this off): prefill the pinned base again once it was evicted.

H2b kept the pinned family's base (its block end and its checkpoint inside the head the family's sessions share) last
under memory pressure, but when it is the only idle slot a growing cache frees it, and every fresh session after that
starts cold. Here, once the scheduler has been idle (no live, filling or waiting request) for ``IDLE_S`` seconds after
a busy period, the decoder checks whether a fresh session of the family still finds a kept state of at least the
family's checkpoint; if not, and a free slot can grow to the base's rows within the memory gate (with ``MARGIN``
spare; nothing kept is evicted and the lone-stream allowance is not used), it admits the base as a warm-start replay
does: the base block plus a user turn's opening, one greedy token, drafts on, through the ordinary prefill, alone. So
the kept states are those a first chat on an idle server would keep (exact by construction, ``warm_starts.replay``).

A request never waits behind it longer than the prompt pass in flight (an idle pass: up to TENSORFOLD_PREFILL_ROWS,
4096 rows in the house, ~2 s; a pass cannot be cut short): the fill loop gives way between passes as soon as a
foreground request is queued, and admitting any other request first takes the re-warm out (its slot keeps the
states it reached, all of them prefilled alone, so still exact; nothing else of it is kept). One re-warm at most per
idle period (a request arms the next); one log line each. Cost: a cold prefill of the base (~8.4 s for a 16.9K-token
Hermes block), only after memory pressure evicted it and the server has been idle ``IDLE_S``. Its block is not noted
in the warm-start file (a re-warm is not a use: a stale base must not outrank a new family by being re-warmed)."""

from __future__ import annotations

import os
import time

GIB = 1 << 30
ENV, IDLE_ENV = "TF_PIN_REWARM", "TF_PIN_REWARM_IDLE_S"
IDLE_S = 30.0            # idle seconds after a busy period before the base is checked (agent tool gaps are shorter)
POLL_S = 10.0            # then, while it is missing but memory has no room, a check every this many seconds
MARGIN = GIB             # room the gate keeps beyond the re-warm's growth (so the next growth need not evict it)


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def idle_seconds() -> float:
    raw = os.environ.get(IDLE_ENV, "").strip()
    if not raw:
        return IDLE_S
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{IDLE_ENV}: seconds (at least 1), not {raw!r}") from None
    if not value >= 1:
        raise ValueError(f"{IDLE_ENV}: seconds (at least 1), not {raw!r}")
    return value


class Rewarm:
    """The decoder's re-warm state: armed by a request, checked when the scheduler has been idle."""

    clock = staticmethod(time.perf_counter)

    def __init__(self, warm, sampling, idle_s: float = IDLE_S) -> None:
        self.warm, self.sampling, self.idle_s = warm, sampling, idle_s
        self.state = None                # None: nothing to do until a request; "idle": first check; "poll": recheck
        self.stream = None               # the re-warm being prefilled
        self.started = 0.0
        self.done = self.yielded = 0

    def arm(self) -> None:
        self.state = "idle"

    def wait(self) -> float | None:
        """How long the idle scheduler waits for a request before ``check`` (None: wait for one)."""

        return None if self.state is None else self.idle_s if self.state == "idle" else POLL_S

    def missing(self, owner) -> list[int] | None:
        """The base's replay prompt when a fresh session of the family finds no kept state of at least its checkpoint
        (None: present, no family yet, or no checkpoint)."""

        prompt = self.warm.base_replay()
        head = self.warm.shared_head()
        cp = getattr(owner.points, "checkpoint", None) if owner.points is not None else None
        at = cp(prompt) if prompt is not None and cp is not None else None
        if at is None or head is None:
            return None
        for ids, _, _, _ in owner.kept:
            if at <= len(ids) <= head and self.warm.family(ids):
                return None
        return prompt

    @staticmethod
    def need(owner, rows: int) -> int | None:
        """Bytes a free slot's caches take to hold ``rows`` (the most any free slot needs); None: no free slot."""

        from .multi import STEP

        need = None
        for st in owner.free:
            if rows <= st.capacity:
                grow = 0
            else:
                size = min(st.limit, -(-rows // STEP) * STEP)
                grow = st.cache_bytes(size) - st.cache_bytes() + st.layer_bytes(size)
            need = grow if need is None else max(need, grow)
        return need


def check(owner) -> bool:
    """The idle scheduler's hook: admit the base again when it is missing and memory has room; True if admitted."""

    from tensorfold.cuda.memory_gate import NoRoom
    from tensorfold.cuda.streams import Stream

    rw = owner.rewarm
    if rw is None or rw.state is None or rw.stream is not None or owner.live() or owner.arrived():
        return False
    try:
        prompt = rw.missing(owner)
        need = Rewarm.need(owner, len(prompt) + owner.depth + 2) if prompt is not None else None
    except Exception as exc:                         # noqa: BLE001  (an optimization only: off until a request)
        rw.state = None
        print(f"[tensorfold] pin re-warm: check failed ({type(exc).__name__}: {exc})", flush=True)
        return False
    if prompt is None:
        rw.state = None                              # present (or nothing to pin): until the next busy period
        return False
    if need is None or not owner.memory_gate.fits(need + MARGIN):
        rw.state = "poll"                            # no free slot, or no room without evicting: look again later
        return False
    s = Stream(list(prompt), 1, rw.sampling, draft=True)
    rw.stream, rw.started, rw.state = s, rw.clock(), None
    owner.quiet_notes = owner.no_evict = True
    try:
        owner.admit(s)
    except NoRoom:                                   # the room went between the check and the growth
        rw.stream, rw.state = None, "poll"
        return False
    except Exception as exc:                         # noqa: BLE001  (an optimization only)
        rw.stream = None
        print(f"[tensorfold] pin re-warm: not started ({type(exc).__name__}: {exc})", flush=True)
        return False
    finally:
        owner.no_evict = False
        if rw.stream is None:
            owner.quiet_notes = False
    return True


def before_admit(owner, s) -> bool:
    """``MultiDecoder.admit``'s first step: True for the re-warm itself; any other request takes a re-warm in progress
    out first (it is admitted between passes, so the re-warm's last pass is all it waits for) and arms the next."""

    rw = owner.rewarm
    if s is rw.stream:
        return True
    if rw.stream is not None:
        give_way(owner)
    rw.arm()
    return False


def give_way(owner) -> None:
    """Another request is being admitted: take the re-warm out (between passes; its kept states stay)."""

    rw = owner.rewarm
    s = rw.stream
    rw.stream = None
    owner.quiet_notes = False
    fill = owner.fills.pop(s.sid, None)
    if s in owner.filling:
        owner.filling.remove(s)
    owner.passed.pop(s.sid, None)
    owner.streams.pop(s.sid, None)
    owner.held.pop(s.sid, None)
    s.done = True
    reached = fill[2] if fill is not None else len(s.prompt)
    kept = [len(k[0]) for k in owner.kept if k[1] is s.st]
    if not kept and all(f is not s.st for f in owner.free):
        owner._shrink(s.st)
        owner.free.append(s.st)
    rw.yielded += 1
    print(f"[tensorfold] pin re-warm: gave way to a request after {reached} of {len(s.prompt)} rows "
          f"({rw.clock() - rw.started:.1f}s; kept {max(kept) if kept else 'nothing'})", flush=True)


def finished(owner, s) -> None:
    """The re-warm's stream ended (its one token sampled, or failed): one line."""

    rw = owner.rewarm
    rw.stream = None
    owner.quiet_notes = False
    if s.error is not None:
        print(f"[tensorfold] pin re-warm: failed ({type(s.error).__name__}: {s.error})", flush=True)
        return
    rw.done += 1
    cp = getattr(owner.points, "checkpoint", None)
    at = cp(s.prompt) if cp is not None else None
    present = rw.missing(owner) is None
    print(f"[tensorfold] pin re-warm: base prefilled again ({len(s.prompt) - len(rw.warm.user)}-token block, "
          f"checkpoint {at}, cached {s.cached}) in {rw.clock() - rw.started:.1f}s while idle"
          f"{'' if present else '; its checkpoint is still missing'}", flush=True)
