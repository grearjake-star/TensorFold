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


def _touch(owner, entry) -> None:
    """A resumed message-start state becomes the newest, so a shared system block outlives the conversations
    that fork from it. Prompt ends keep their place: a resend must not age the ends nobody has resent yet."""

    if _is_start(owner, entry):
        owner.kept = [k for k in owner.kept if k is not entry] + [entry]
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
    older = [(i, k, _is_start(owner, k)) for i, k in enumerate(kept[:-1])]
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
        idle = [k[1] for k in owner.kept if id(k[1]) not in busy]
        if not idle:
            raise RuntimeError("no free stream slot")
        held = {id(k[1]) for k in owner.kept if _is_start(owner, k)}
        idle = next((st for st in idle if id(st) not in held), idle[0])   # a kept system block's slot goes last
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


def remember(owner, ids, st, snap, tail, start: bool = False) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list. ``start``: a message-start
    state (the end of a shared system block), which eviction keeps longest."""

    starts = _starts(owner)
    if start:
        starts.add(tuple(ids))
        _note(owner, ids)
    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    while len(owner.kept) > owner.keep:
        gone.append(owner.kept.pop(_victim(owner))[1])
    if starts:
        lengths = {len(k[0]) for k in owner.kept}
        owner.starts = {t for t in starts if len(t) in lengths and any(len(k[0]) == len(t) and tuple(k[0]) == t
                                                                       for k in owner.kept)}
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
