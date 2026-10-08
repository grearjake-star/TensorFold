"""A fresh request (no kept prefix to resume) with no free slot takes the idle slot that loses the fewest kept tokens,
not the first idle slot in ``kept`` order (scottleimroth's #315 report: an unrelated task arriving between two long
conversations' turns evicted one of them while a cheaper idle slot existed). Slot choice only: replies are unchanged."""

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes
from tests.test_flashnext_prefix_victim import Owner, Slot, entry


@pytest.fixture(autouse=True)
def fresh_slack(monkeypatch):
    """These cases test the opt-in #315 rule (TF_FRESH_SLACK); the default is the house rule (test at the end)."""

    monkeypatch.setenv("TF_FRESH_SLACK", "2048")

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


def test_within_the_slack_the_oldest_idle_slot_goes_and_busy_slots_are_never_taken():
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


def test_a_conversation_that_just_started_keeps_its_end_beside_older_short_ones():
    """The full-pool pattern of tests/cuda/test_flashnext_L7.py: older one-shot prompts' ends and a conversation's
    first, shorter end. The next fresh request takes the oldest end, not the conversation's (kept order decides
    among slots within FRESH_SLACK tokens of each other)."""

    a, b, c = Slot("a"), Slot("b"), Slot("c")
    owner = Owner([entry(list(range(100, 124)), a, "old-1"), entry(list(range(200, 225)), b, "old-2"),
                   entry(list(range(300, 316)), c, "conversation")])
    st, _, cached = prefixes.slot_for(owner, list(range(70000, 70006)), True)
    assert st is a and cached == 0 and {k[3] for k in owner.kept} == {"old-2", "conversation"}


@pytest.mark.parametrize("value", ["", "off", "0"])
def test_without_tf_fresh_slack_the_oldest_idle_slot_goes_as_in_the_house(monkeypatch, value):
    """Default: the house rule, the oldest idle slot in kept order (a system block's slot last), even a long chain."""

    monkeypatch.setenv("TF_FRESH_SLACK", value)
    owner, a, b, c = long_conversations()
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b and {k[3] for k in owner.kept} == {"system", "one", "short"}


def test_a_bad_tf_fresh_slack_is_refused(monkeypatch):
    monkeypatch.setenv("TF_FRESH_SLACK", "lots")
    owner, *_ = long_conversations()
    with pytest.raises(ValueError, match="TF_FRESH_SLACK"):
        prefixes.slot_for(owner, list(range(70000, 70100)), True)


def test_a_bad_tf_fresh_slack_is_refused_when_the_decoder_starts(monkeypatch):
    """Read once at start-up like the other knobs: a bad value stops the server before it serves, not an admission."""

    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    monkeypatch.setenv("TF_FRESH_SLACK", "lots")
    with pytest.raises(ValueError, match="TF_FRESH_SLACK"):
        MultiDecoder(object(), slots=2, capacity=64)


def test_the_decoder_uses_the_value_read_at_start_up(monkeypatch):
    owner, a, b, c = long_conversations()
    owner.fresh_slack = None                             # read at start-up: the house rule
    monkeypatch.setenv("TF_FRESH_SLACK", "lots")         # changed later: not read again
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b
