"""An admission refused for memory (NoRoom) puts its resumed kept state back as it was.

A request that resumes from the system-block checkpoint (TF_SYS_CHECKPOINT) and then finds no memory to grow its
slot re-keeps that state. It used to go back as a plain prompt end: remember() dropped it from the checkpoints, so
it counted against keep and could push out a prompt end (or itself), and lost its place beside keep."""

import types
from types import SimpleNamespace

import pytest

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tests.test_flashnext_prefix_victim import Owner, Slot, entry

CHECKPOINT = list(range(2048))


class Decoder(Owner):
    """MultiDecoder.admit's single-GPU path up to its memory check, on host fakes."""

    def __init__(self, kept, keep):
        super().__init__(kept, room=False)
        self.keep, self.capacity, self.depth = keep, 1 << 20, 3
        self.streams, self.filling, self.w = {}, [], SimpleNamespace(comm=None)
        self._remember = types.MethodType(MultiDecoder._remember, self)

    def _slot_for(self, prompt, reuse):
        return prefixes.slot_for(self, prompt, reuse)

    def _grow(self, st, rows, **kwargs):
        return False                                     # no memory for the prompt's rows


def request(prompt):
    return SimpleNamespace(prompt=prompt, count=8, draft=True, vision=None, constraint=None, probabilities=None)


def test_a_refused_admission_keeps_the_checkpoint_a_checkpoint():
    a, b = Slot("a"), Slot("b")
    other = list(range(50000, 53000))
    dec = Decoder([entry(CHECKPOINT, a, "checkpoint"), entry(other, b, "other")], keep=1)
    dec.checkpoints, dec.starts = {tuple(CHECKPOINT)}, {tuple(CHECKPOINT)}
    with pytest.raises(NoRoom):
        MultiDecoder.admit(dec, request(CHECKPOINT + list(range(9000, 9500))))
    assert tuple(CHECKPOINT) in dec.checkpoints and tuple(CHECKPOINT) in dec.starts
    assert {k[3] for k in dec.kept} == {"checkpoint", "other"}       # keep=1 counts only the prompt end


def test_a_refused_admission_from_a_prompt_end_is_unchanged():
    a = Slot("a")
    end = list(range(3000))
    dec = Decoder([entry(end, a, "end")], keep=4)
    with pytest.raises(NoRoom):
        MultiDecoder.admit(dec, request(end + [1, 2, 3]))
    assert [k[3] for k in dec.kept] == ["end"] and not getattr(dec, "checkpoints", None)
