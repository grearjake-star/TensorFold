"""System-block checkpoints (TF_SYS_CHECKPOINT): a kept state at the last prompt-pass boundary before a system
block's end, so a block that differs only near its end (a date line, a memory section) resumes there. They never
count against ``keep``, are capped, are never recorded as warm starts, and resume like any kept prefix."""

import pytest

from tensorfold.cuda import markers
from tensorfold.cuda.markers import MIN_GAP, snapshot_points
from tensorfold.families.qwen4_exp.cuda import prefixes

OPEN, ASSISTANT = 900, 901


def _prompt(system: int, user: int, tail: int = 0) -> list[int]:
    return [OPEN] + [5] * system + [9] * tail + [OPEN] + [6] * user + [OPEN, ASSISTANT, 8]


def test_the_checkpoint_is_the_last_granule_at_least_min_gap_before_the_block_end(monkeypatch):
    monkeypatch.delenv("TF_SYS_CHECKPOINT", raising=False)
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    ids = _prompt(13123, 600)                        # the second message starts at 13124 (Hermes-sized)
    assert points.checkpoint(ids) == 12288
    assert points(ids) == [12288, 13124, len(ids) - 3]
    near = _prompt(12288 + MIN_GAP - 2, 600)         # block end 12288 + 255: too close, the granule before it
    assert points.checkpoint(near) == 10240
    assert points.checkpoint(_prompt(2000, 600)) is None         # shorter than a granule + MIN_GAP: none
    assert points.checkpoint([5] * 30000) is None                # no second message
    monkeypatch.setenv("TF_SYS_CHECKPOINT", "0")
    off = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    assert off.checkpoint(ids) is None and off(ids) == [13124, len(ids) - 3]
    monkeypatch.setenv("TF_SYS_CHECKPOINT", "4096")
    assert snapshot_points((OPEN,), (OPEN, ASSISTANT)).checkpoint(ids) == 12288
    monkeypatch.setenv("TF_SYS_CHECKPOINT", "100")
    with pytest.raises(ValueError):
        markers.checkpoint_rows()


def test_two_blocks_differing_only_near_the_end_share_the_checkpoint(monkeypatch):
    monkeypatch.delenv("TF_SYS_CHECKPOINT", raising=False)
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    a = _prompt(12600, 300, tail=500)
    b = [OPEN] + [5] * 12600 + [7] * 500 + [OPEN] + [6] * 300 + [OPEN, ASSISTANT, 8]
    assert points.checkpoint(a) == points.checkpoint(b) == 12800 // 2048 * 2048
    cp = points.checkpoint(a)
    assert a[:cp] == b[:cp] and a[:points(a)[1]] != b[:points(b)[1]]


class Slot:
    def __init__(self, name):
        self.name, self.copied = name, None

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)


class Warm:
    def __init__(self):
        self.noted = []

    def note(self, ids):
        self.noted.append(len(ids))


class Owner:
    def __init__(self, keep, free=()):
        self.kept, self.free, self.keep, self.depth = [], list(free), keep, 3
        self.warm_starts = Warm()

    def _busy(self):
        return set()

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, **kwargs):
        return True

    def _shrink(self, st, **kwargs):
        pass


def snap(ids):
    return {"pos": len(ids), "mtp_len": len(ids) - 1}


BLOCK = list(range(13124))
CP = BLOCK[:12288]


def test_a_checkpoint_is_kept_beside_keep_and_never_recorded_as_a_warm_start():
    owner, a = Owner(keep=2), Slot("a")
    prefixes.remember(owner, CP, a, snap(CP), "cp", checkpoint=True)
    prefixes.remember(owner, BLOCK, a, snap(BLOCK), "block", start=True)
    assert owner.warm_starts.noted == [len(BLOCK)]                      # the block, not its checkpoint
    for i in range(3):                                                  # conversation ends on other slots
        ids = list(range(50000 + 1000 * i, 50000 + 1000 * i + 500))
        prefixes.remember(owner, ids, Slot(f"c{i}"), snap(ids), f"conv{i}")
    tags = [k[3] for k in owner.kept]
    assert "cp" in tags and sum(t != "cp" for t in tags) == 2           # keep 2 counts the others only
    assert tuple(CP) in owner.checkpoints and tuple(CP) in owner.starts


def test_checkpoints_past_the_cap_drop_the_oldest():
    owner = Owner(keep=16)
    for i in range(prefixes.CHECKPOINTS + 2):
        ids = [i] + list(range(1, 2048))
        prefixes.remember(owner, ids, Slot(f"s{i}"), snap(ids), f"cp{i}", checkpoint=True)
    assert [k[3] for k in owner.kept] == [f"cp{i}" for i in range(2, prefixes.CHECKPOINTS + 2)]
    assert len(owner.checkpoints) == prefixes.CHECKPOINTS
    assert {s.name for s in owner.free} == {"s0", "s1"}                  # their slots come back


def test_a_block_with_a_new_date_resumes_from_the_checkpoint_on_a_copy():
    owner, a, spare = Owner(keep=16), Slot("a"), Slot("spare")
    prefixes.remember(owner, CP, a, snap(CP), "cp", checkpoint=True)
    prefixes.remember(owner, BLOCK, a, snap(BLOCK), "block", start=True)
    owner.free = [spare]
    new_block = CP + [7] * (len(BLOCK) - len(CP) + 6)                   # same up to the checkpoint, new tail
    st, resume, cached = prefixes.slot_for(owner, new_block + [1, 2, 3], True)
    assert cached == len(CP) and resume["tail"] == "cp" and st is spare and spare.copied == (a, len(CP), len(CP) - 1)
    assert owner.warm_starts.noted == [len(BLOCK)]                      # resuming the checkpoint records nothing
    assert [k[3] for k in owner.kept][-1] == "cp"                       # touched: the newest, kept longest


def test_a_checkpoint_is_never_the_victim_while_prompt_ends_exceed_keep():
    owner, a = Owner(keep=3), Slot("a")
    prefixes.remember(owner, CP, a, snap(CP), "cp", checkpoint=True)
    for i in range(6):
        ids = list(range(90000 + 1000 * i, 90000 + 1000 * i + 400))
        prefixes.remember(owner, ids, Slot(f"c{i}"), snap(ids), f"conv{i}")
    assert [k[3] for k in owner.kept] == ["cp", "conv3", "conv4", "conv5"]
