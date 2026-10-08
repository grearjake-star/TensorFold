"""Kept-state bookkeeping does not copy or hash long conversations' ids on the decode thread.

``remember`` and the eviction victim run for every kept point and admission while the decode thread holds the GIL.
They asked ``tuple(ids) in checkpoints`` of every kept entry: a 100K-token conversation became a 100K tuple, built
and hashed, several times per call (tens of ms per admission with long conversations). Checkpoints are compared by
length first now, so only entries as long as a checkpoint are turned into tuples. Same decisions, bit for bit."""

import random

from tensorfold.families.qwen4_exp.cuda import prefixes
from tests.test_flashnext_prefix_victim import Owner, Slot, entry


class Watched(list):
    """A kept entry's ids that count how often they are iterated (tuple() of a list subclass iterates it)."""

    iterated = 0

    def __iter__(self):
        Watched.iterated += 1
        return super().__iter__()


SYSTEM = list(range(4000))
CHECKPOINT = SYSTEM[:2048]


def pool(conv_len=30000, n=10):
    slots = [Slot(str(i)) for i in range(n + 1)]
    kept = [entry(CHECKPOINT, slots[0], "cp"), entry(SYSTEM, slots[0], "system")]
    for i in range(n):
        kept.append(entry(Watched(SYSTEM + list(range(100000 * (i + 1), 100000 * (i + 1) + conv_len))),
                          slots[i + 1], f"c{i}"))
    owner = Owner(kept)
    owner.keep = n
    owner.starts, owner.checkpoints = {tuple(SYSTEM), tuple(CHECKPOINT)}, {tuple(CHECKPOINT)}
    return owner, slots


def test_remember_and_victim_never_iterate_long_entries():
    owner, slots = pool()
    Watched.iterated = 0
    prefixes._victim(owner)
    prefixes.remember(owner, SYSTEM + list(range(7, 900)), slots[3], {"mtp_len": 0}, None)
    prefixes.remember(owner, SYSTEM + list(range(9, 700)), slots[4], {"mtp_len": 0}, None)
    assert Watched.iterated == 0
    assert tuple(CHECKPOINT) in owner.checkpoints and any(k[3] == "cp" for k in owner.kept)


def _old_is_checkpoint(owner, entry):
    cps = getattr(owner, "checkpoints", None)
    return bool(cps) and tuple(entry[0]) in cps


def test_same_decisions_as_the_unguarded_lookup(monkeypatch):
    for trial in range(200):
        runs = []
        for guarded in (True, False):
            r = random.Random(trial)
            slots = [Slot(str(i)) for i in range(6)]
            sys_len = r.choice([300, 600, 2048])
            system = list(range(sys_len))
            kept = [entry(system[:256], slots[0], "cp"), entry(system, slots[0], "system")]
            owner = Owner(kept)
            owner.keep = r.randint(1, 5)
            owner.starts, owner.checkpoints = {tuple(system)}, {tuple(system[:256])}
            if not guarded:
                monkeypatch.setattr(prefixes, "_is_checkpoint", _old_is_checkpoint)
            for step in range(12):
                ids = system + [r.randint(1000, 1003) for _ in range(r.randint(0, 6))]
                cp = r.random() < 0.1
                prefixes.remember(owner, system[:256] if cp else ids, slots[r.randint(0, 5)], {"mtp_len": 0}, None,
                                  start=r.random() < 0.2, checkpoint=cp)
            monkeypatch.undo()
            runs.append(([(k[0], k[1].name) for k in owner.kept], sorted(owner.checkpoints),
                         sorted(owner.starts), [s.name for s in owner.free]))
        assert runs[0] == runs[1], trial
