"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""


def _best(kept, prompt, busy=()):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def _touch(owner, entry) -> None:
    """A resumed entry becomes the newest, so a shared system block outlives the conversations that fork from it."""

    owner.kept = [k for k in owner.kept if k is not entry] + [entry]


def _middle(kept, entry) -> bool:
    """A conversation's superseded turn: its slot also keeps a shorter and a longer entry. A slot's entries share
    its attention rows, so they are prefixes of one another (``slot_for`` drops the rest): lengths suffice."""

    sizes = [len(k[0]) for k in kept if k[1] is entry[1] and k is not entry]
    n = len(entry[0])
    return any(m < n for m in sizes) and any(m > n for m in sizes)


def _victim(kept) -> int:
    """The entry to drop past ``keep``: the oldest middle turn (never the newest entry), else the oldest. A
    slot's chain root (a message-start state: the shared system block) and its newest end stay longest."""

    return next((i for i, k in enumerate(kept[:-1]) if _middle(kept, k)), 0)


def slot_for(owner, prompt: list[int], reuse: bool):
    """Copy a fork into spare capacity; otherwise retain the released idle-slot and memory-pressure behavior."""

    busy = owner._busy()
    best = _best(owner.kept, prompt) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        if owner.free:
            spare = owner.free.pop()
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    _touch(owner, best)
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            owner.free.append(spare)
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
        idle = next((k[1] for k in owner.kept if id(k[1]) not in busy), None)
        if idle is None:
            raise RuntimeError("no free stream slot")
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


def remember(owner, ids, st, snap, tail) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list."""

    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    while len(owner.kept) > owner.keep:
        gone.append(owner.kept.pop(_victim(owner.kept))[1])
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
