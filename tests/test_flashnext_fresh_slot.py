"""A fresh request (no kept prefix to resume) with no free slot takes the idle slot that loses the fewest kept tokens,
not the first idle slot in ``kept`` order (scottleimroth's #315 report: an unrelated task arriving between two long
conversations' turns evicted one of them while a cheaper idle slot existed). Slot choice only: replies are unchanged."""

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes
from tests.test_flashnext_prefix_victim import Owner, Slot, entry

SYSTEM = list(range(300))                                   # the shared system block (tools + instructions)


def long_conversations(short_len=40):
    """Two long conversations on one system block (slots a, b) and a short earlier task's chain (slot c)."""

    a, b, c = Slot("a"), Slot("b"), Slot("c")
    one = SYSTEM + list(range(1000, 31000))                  # ~30K-token conversation
    two = SYSTEM + list(range(40000, 60000))                 # ~20K-token conversation
    short = list(range(90000, 90000 + short_len))
    kept = [entry(SYSTEM, a, "system"), entry(one, a, "one"), entry(two, b, "two"), entry(short, c, "short")]
    owner = Owner(kept)
    owner.starts = {tuple(SYSTEM)}
    return owner, a, b, c


def test_an_unrelated_task_takes_the_cheap_idle_slot_and_both_conversations_stay():
    owner, a, b, c = long_conversations()
    st, resume, cached = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is c and resume is None and cached == 0
    tags = {k[3] for k in owner.kept}
    assert {"system", "one", "two"} <= tags and "short" not in tags


def test_the_next_turns_of_both_conversations_still_resume_from_their_kept_ends():
    owner, a, b, c = long_conversations()
    prefixes.slot_for(owner, list(range(70000, 70100)), True)          # the unrelated task, between turns
    owner.free.append(c)                                                  # it finished; its slot is free again
    for slot, tag in ((a, "one"), (b, "two")):
        chain = next(k[0] for k in owner.kept if k[3] == tag)
        st, resume, cached = prefixes.slot_for(owner, chain + [7, 8], True)
        assert st is slot and cached == len(chain) and resume["tail"] == tag


def test_with_only_long_conversations_idle_the_shorter_one_goes():
    a, b = Slot("a"), Slot("b")
    one, two = SYSTEM + list(range(1000, 31000)), SYSTEM + list(range(40000, 45000))
    owner = Owner([entry(SYSTEM, a, "system"), entry(one, a, "one"), entry(two, b, "two")])
    owner.starts = {tuple(SYSTEM)}
    st, _, cached = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b and cached == 0 and {k[3] for k in owner.kept} == {"system", "one"}


def test_the_system_blocks_slot_goes_last_even_when_it_keeps_fewer_tokens():
    a, b = Slot("a"), Slot("b")
    owner = Owner([entry(SYSTEM, a, "system"), entry(list(range(5000, 6000)), b, "other")])
    owner.starts = {tuple(SYSTEM)}
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b and [k[3] for k in owner.kept] == ["system"]


def test_ties_keep_the_kept_order_and_busy_slots_are_never_taken():
    a, b, c = Slot("a"), Slot("b"), Slot("c")
    owner = Owner([entry(list(range(10, 60)), a, "a"), entry(list(range(100, 150)), b, "b"),
                   entry(list(range(200, 210)), c, "c")], busy=[c])
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is a and {k[3] for k in owner.kept} == {"b", "c"}


def test_no_idle_slot_still_raises():
    a = Slot("a")
    owner = Owner([entry(list(range(10, 60)), a, "a")], busy=[a])
    with pytest.raises(RuntimeError, match="no free stream slot"):
        prefixes.slot_for(owner, list(range(70000, 70100)), True)
