"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""


def _best(kept, prompt, busy=()):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def _starts(owner) -> set:
    """The ids of kept message-start states (a shared system block), as tuples; created on first use."""

    starts = getattr(owner, "starts", None)
    if starts is None:
        starts = owner.starts = set()
    return starts


def _is_start(owner, entry) -> bool:
    starts = _starts(owner)
    return bool(starts) and any(len(t) == len(entry[0]) for t in starts) and tuple(entry[0]) in starts


CHECKPOINTS = 4          # system-block checkpoints kept beside ``keep`` (oldest dropped past it)


def _checkpoints(owner) -> set:
    """The ids of kept system-block checkpoints (``remember(checkpoint=True)``), as tuples; created on first use."""

    got = getattr(owner, "checkpoints", None)
    if got is None:
        got = owner.checkpoints = set()
    return got


def _is_checkpoint(owner, entry) -> bool:
    cps = getattr(owner, "checkpoints", None)
    return bool(cps) and tuple(entry[0]) in cps


def _touch(owner, entry) -> None:
    """A resumed message-start state becomes the newest, so a shared system block outlives the conversations
    that fork from it. Prompt ends keep their place: a resend must not age the ends nobody has resent yet."""

    if _is_start(owner, entry):
        owner.kept = [k for k in owner.kept if k is not entry] + [entry]
        if not _is_checkpoint(owner, entry):             # a checkpoint is part of a block, not a block to record
            _note(owner, entry[0])


def _note(owner, ids) -> None:
    """TF_WARM_STARTS: a kept or resumed message-start state's system block is recorded for the next start."""

    warm = getattr(owner, "warm_starts", None)
    if warm is not None:
        warm.note(ids)


def _middle(kept, entry) -> bool:
    """A conversation's superseded turn: its slot also keeps a shorter and a longer entry. A slot's entries share
    its attention rows, so they are prefixes of one another (``slot_for`` drops the rest): lengths suffice."""

    sizes = [len(k[0]) for k in kept if k[1] is entry[1] and k is not entry]
    n = len(entry[0])
    return any(m < n for m in sizes) and any(m > n for m in sizes)


def _victim(owner) -> int:
    """The entry to drop past ``keep`` (never the newest), oldest first within the first class that has one:
    a superseded middle turn, a prompt end its own slot has continued, any prompt end, then any entry. Message-start
    states (the shared system block) go last; every prompt's own newest end outlasts the ends it superseded."""

    kept = owner.kept
    older = [(i, k, _is_start(owner, k)) for i, k in enumerate(kept[:-1]) if not _is_checkpoint(owner, k)]
    for rule in (lambda k, start: not start and _middle(kept, k),
                 lambda k, start: not start and _longer(kept, k),
                 lambda k, start: not start,
                 lambda k, start: True):
        i = next((i for i, k, start in older if rule(k, start)), None)
        if i is not None:
            return i
    return 0


def _claim(owner, best, busy):
    """The idle slot whose kept chains are shortest, when losing them costs less than resuming in place would (or the
    source is busy). Among those, a slot keeping no message-start state goes first: a shared system block stays."""

    longest, slots, starts = {}, {}, set()
    for k in owner.kept:
        key = id(k[1])
        longest[key] = max(longest.get(key, 0), len(k[0]))
        slots[key] = k[1]
        if _is_start(owner, k):
            starts.add(key)
    cut = None if id(best[1]) in busy else longest[id(best[1])] - len(best[0])
    idle = [(key in starts, n, i, key) for i, (key, n) in enumerate(longest.items())
            if key not in busy and key != id(best[1]) and (cut is None or n < cut)]
    return slots[min(idle)[3]] if idle else None


def slot_for(owner, prompt: list[int], reuse: bool):
    """Copy a fork into spare capacity, or into the idle slot that loses the least; otherwise keep the old behavior."""

    busy = owner._busy()
    best = _best(owner.kept, prompt) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        spare, before = (owner.free.pop(), None) if owner.free else (None, None)
        if spare is None and (victim := _claim(owner, best, busy)) is not None:
            spare, before = victim, list(owner.kept)
            owner._drop_kept(victim)
        if spare is not None:
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    _touch(owner, best)
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            if before is None:
                owner.free.append(spare)
            else:                                    # refused: the victim's chains stay, in their old places
                left = {id(k) for k in owner.kept}
                owner.kept = [k for k in before if k[1] is spare or id(k) in left] + \
                             [k for k in owner.kept if all(k is not b for b in before)]
        best = _best(owner.kept, prompt, busy) if reuse else None
        if best is not None and owner.free and _longer(owner.kept, best):
            best = None
    if best is not None:
        n = len(best[0])
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        _touch(owner, best)
        return best[1], {"state": best[2], "tail": best[3]}, n
    if not owner.free:
        idle = _cheapest_idle(owner, busy)
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


def fresh_slack() -> int | None:
    """TF_FRESH_SLACK=<tokens> (opt-in, unset or "off": the house rule): a fresh request with no free slot does not
    evict an idle slot whose longest kept chain is more than this many tokens beyond the cheapest idle slot's.
    The decoder reads it once at start-up (``owner.fresh_slack``), so a bad value is refused before serving."""

    import os

    value = os.environ.get("TF_FRESH_SLACK", "").strip().lower()
    if value in ("", "off", "0", "false", "no"):
        return None
    try:
        return max(0, int(value))
    except ValueError:
        raise ValueError(f"TF_FRESH_SLACK={value!r}: a token count, or off") from None


def _cheapest_idle(owner, busy):
    """A fresh request's slot when none is free: the oldest idle slot (``kept`` order), a slot keeping a message-start
    state (a shared system block) last. With ``TF_FRESH_SLACK`` set (#315: an unrelated task arriving between two long
    conversations' turns), only idle slots whose longest kept chain is within that many tokens of the cheapest idle
    slot's are eligible, so a long conversation is not evicted while a much cheaper idle slot exists."""

    longest, order = {}, []
    for k in owner.kept:
        key = id(k[1])
        if key in busy:
            continue
        if key not in longest:
            order.append(k[1])
        longest[key] = max(longest.get(key, 0), len(k[0]))
    if not order:
        raise RuntimeError("no free stream slot")
    held = {id(k[1]) for k in owner.kept if _is_start(owner, k)}
    pool = [st for st in order if id(st) not in held] or order
    slack = owner.fresh_slack if hasattr(owner, "fresh_slack") else fresh_slack()
    if slack is None:
        return pool[0]
    floor = min(longest[id(st)] for st in pool)
    return next(st for st in pool if longest[id(st)] <= floor + slack)


def remember(owner, ids, st, snap, tail, start: bool = False, checkpoint: bool = False) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list. ``start``: a message-start
    state (the end of a shared system block), which eviction keeps longest. ``checkpoint`` (a start too): a
    system-block checkpoint (markers ``checkpoint``) a block that differs near its end resumes from; at most
    ``CHECKPOINTS`` are kept beside ``keep`` (they never count against it, and are never recorded as warm starts)."""

    starts = _starts(owner)
    cps = _checkpoints(owner)
    if checkpoint:
        cps.add(tuple(ids))
        start = True
    else:
        cps.discard(tuple(ids))
    if start:
        starts.add(tuple(ids))
        if not checkpoint:
            _note(owner, ids)
    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    while sum(not _is_checkpoint(owner, k) for k in owner.kept) > owner.keep:
        gone.append(owner.kept.pop(_victim(owner))[1])
    held = [i for i, k in enumerate(owner.kept) if _is_checkpoint(owner, k)]
    for i in reversed(held[:max(0, len(held) - CHECKPOINTS)]):     # the oldest checkpoints past the limit
        gone.append(owner.kept.pop(i)[1])
    if cps:
        owner.checkpoints = {t for t in cps if any(tuple(k[0]) == t for k in owner.kept)}
    if starts:
        lengths = {len(k[0]) for k in owner.kept}
        owner.starts = {t for t in starts if len(t) in lengths and any(len(k[0]) == len(t) and tuple(k[0]) == t
                                                                       for k in owner.kept)}
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
