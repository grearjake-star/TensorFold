"""House: a fresh request on a full pool takes a cheap or the least recently used idle slot, not a conversation's.

``_cheapest_idle`` used to rank idle slots by their OLDEST kept entry. A Hermes agent conversation keeps its first
prompt end for its whole life (resumes cut nothing shorter), so after a few minutes its slot looked older than any
other, and each short helper request Hermes sends between an agent's steps (a 400-500-token summary, a title) took
it. On 2026-10-08 20:00-20:03 that re-prefilled Jake's 84K, 86K and 94K-token follow-ups from the 16,607-token system
block (TTFT 39 s, 37 s, 55 s) while a 275-token title's slot sat idle. Six of the eight slots held system blocks
(warm starts and the session's block), so two slots carried every conversation and helper request.

Now: an idle slot whose longest kept chain is at most CHEAP_ROWS tokens goes first, else the least recently used by
its newest entry; a start-holding slot still goes last, and TF_FRESH_SLACK keeps its opt-in rule. Slot choice only:
no kept state, resume or kernel changes, so replies keep their bits.

The replay test drives MultiDecoder's real slot code (slot_for, _fresh_slot, _hand_over, remember, _move_to_solo,
finish) on stand-in slots with the house's arrival pattern of that evening (19:02-20:08, warm starts 4951, 26709,
15100, 2535): request times, prompt and reply lengths, and which request each one continues, as synthetic token ids
in the house template's marker shape (data/flashnext_cachemiss_20261008.json; no text). On the old rule the three
long follow-ups resume at the system block's end (16607 observed); here at their previous prompt's end.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.markers import MIN_GAP, snapshot_points
from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.cuda import graphs as graphs_module
from tensorfold.families.qwen4_exp.cuda import multi as multi_module
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.decode import entry_end
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder
from tests.test_flashnext_prefix_victim import Owner, Slot, entry

OPEN, ASSISTANT = 1, 2
DEPTH = 3
DATA = Path(__file__).parent / "data" / "flashnext_cachemiss_20261008.json"
SYSTEM = list(range(300))


@pytest.fixture(autouse=True)
def house_rule(monkeypatch):
    monkeypatch.delenv("TF_FRESH_SLACK", raising=False)


# ------------------------------------------------------------------------------------------------------ the rule
def test_a_fresh_request_takes_a_titles_slot_not_a_conversation_that_started_earlier():
    conv, title, block = Slot("conversation"), Slot("title"), Slot("block")
    first = SYSTEM + list(range(1000, 1300))                    # the conversation's first prompt end (kept for life)
    last = first + list(range(2000, 80000))                      # its newest end, 78K tokens on
    owner = Owner([entry(SYSTEM, block, "system"), entry(first, conv, "first"),
                   entry(list(range(5000, 5275)), title, "title"), entry(last, conv, "last")])
    owner.starts = {tuple(SYSTEM)}
    st, resume, cached = prefixes.slot_for(owner, list(range(9000, 9437)), True)
    assert st is title and resume is None and cached == 0
    assert {k[3] for k in owner.kept} == {"system", "first", "last"}


def test_without_a_cheap_slot_the_least_recently_used_conversation_goes():
    a, b, block = Slot("a"), Slot("b"), Slot("block")
    a_first, b_only = SYSTEM + list(range(1000, 5000)), SYSTEM + list(range(10000, 30000))
    a_last = a_first + list(range(40000, 60000))
    owner = Owner([entry(SYSTEM, block, "system"), entry(a_first, a, "a-first"), entry(b_only, b, "b"),
                   entry(a_last, a, "a-last")])                  # a started first but ran last
    owner.starts = {tuple(SYSTEM)}
    st, _, cached = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b and cached == 0 and {k[3] for k in owner.kept} == {"system", "a-first", "a-last"}


def test_a_cheap_slot_goes_before_an_older_long_conversation():
    a, c = Slot("a"), Slot("c")
    owner = Owner([entry(SYSTEM + list(range(1000, 31000)), a, "long"), entry(list(range(90000, 90040)), c, "probe")])
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is c and [k[3] for k in owner.kept] == ["long"]


def test_among_cheap_slots_the_least_recently_used_goes():
    a, b, c = Slot("a"), Slot("b"), Slot("c")
    owner = Owner([entry(list(range(100, 400)), a, "old"), entry(list(range(500, 900)), b, "newer"),
                   entry(SYSTEM + list(range(1000, 9000)), c, "long")])
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is a and {k[3] for k in owner.kept} == {"newer", "long"}


def test_the_system_blocks_slot_still_goes_last():
    a, b = Slot("a"), Slot("b")
    owner = Owner([entry(SYSTEM, a, "system"), entry(SYSTEM + list(range(1000, 40000)), b, "conversation")])
    owner.starts = {tuple(SYSTEM)}
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b and [k[3] for k in owner.kept] == ["system"]


def test_busy_slots_are_never_taken_and_no_idle_slot_still_raises():
    a, b = Slot("a"), Slot("b")
    owner = Owner([entry(list(range(10, 60)), a, "a"), entry(list(range(100, 150)), b, "b")], busy=[a])
    st, _, _ = prefixes.slot_for(owner, list(range(70000, 70100)), True)
    assert st is b
    owner = Owner([entry(list(range(10, 60)), a, "a")], busy=[a])
    with pytest.raises(RuntimeError, match="no free stream slot"):
        prefixes.slot_for(owner, list(range(70000, 70100)), True)


# ------------------------------------------------------------------------------------- the 2026-10-08 evening replay
class Graphs:
    def __init__(self, e, *, max_rows=8):
        self.e = e

    def warm(self, rows=None, every_bucket=False):
        return 0


class Rows:
    """A stand-in slot that counts rows (the replay checks slot choices, not tensors)."""

    def __init__(self):
        self.capacity, self.limit, self.pos, self.mtp_len, self.cur = FIRST, 262144, 0, 0, [0]
        self.image_positions, self.kv_dtype = None, "bf16"

    def cache_bytes(self, rows=None):
        return rows or self.capacity

    def layer_bytes(self, rows=None):
        return 0

    def resize(self, rows):
        before, self.capacity = self.capacity, rows
        return rows - before

    def reset(self, w):
        self.pos = self.mtp_len = 0

    def set_pos(self, n):
        self.pos = n

    def set_mtp_len(self, n):
        self.mtp_len = n

    def copy_from(self, other):
        self.pos, self.mtp_len = other.pos, other.mtp_len

    def copy_prefix(self, source, pos, mtp_len):
        assert pos <= min(self.capacity, source.pos)


def _decoder(monkeypatch, slots, keep):
    monkeypatch.setattr(graphs_module, "Graphs", Graphs)
    monkeypatch.setattr(multi_module, "_tensors", lambda st: iter(()))
    dec = object.__new__(MultiDecoder)
    dec.w, dec.planning, dec.link, dec.depth, dec.keep = SimpleNamespace(comm=None), False, None, DEPTH, keep
    dec.free = [Rows() for _ in range(slots)]
    dec.slots = list(dec.free)
    dec.streams, dec.filling, dec.fills, dec.held, dec.kept = {}, [], {}, {}, []
    dec.memory_gate, dec.next_id, dec.solo_on = MemoryGate(1 << 60, 0), 0, True
    dec._release_pinned = lambda need: False
    dec.solo = SimpleNamespace(st=dec.free[0], rows=DEPTH + 1)
    dec.solo.graphs = Graphs(dec.solo)
    dec.fresh_slack = None
    dec.points = snapshot_points([OPEN], [OPEN, ASSISTANT], checkpoint=2048)
    return dec


def _admit(dec, prompt, tokens):
    """MultiDecoder.admit's slot choice, then the kept points multi_fill._joined keeps (flags as it sets them)."""

    s = SimpleNamespace(prompt=list(prompt), sid=dec.next_id, draft=True, vision=None, done=False, waiting=False)
    dec.next_id += 1
    st, _, cached = dec._slot_for(list(prompt), True)
    dec._grow(st, len(prompt) + DEPTH + 2, alone=not dec.streams)
    s.st, s.cached = st, cached
    end = entry_end(prompt)
    cp = dec.points.checkpoint(prompt)
    for pos in sorted({p for p in dec.points(prompt) if cached + MIN_GAP <= p < end}) + [end]:
        if pos <= cached:
            continue
        start = pos != end and dec.points.block_end(prompt[:pos + 1]) == pos
        dec._remember(list(prompt[:pos]), st, {"pos": pos, "mtp_len": pos - 1}, None, start=start,
                      checkpoint=cp is not None and cp == pos)
    st.pos = st.mtp_len = len(prompt) + tokens
    dec._grow(st, len(prompt) + tokens + DEPTH + 2, alone=len(dec.streams) == 0)
    dec.streams[s.sid] = s
    return s


def _ids(runs):
    return [t for first, n in runs for t in range(first, first + n)]


def _lone_to_solo(dec):
    live = list(dec.streams.values())
    if len(live) == 1 and live[0].st is not dec.solo.st:
        dec._move_to_solo(live[0])


def _replay(monkeypatch):
    data = json.loads(DATA.read_text())
    dec = _decoder(monkeypatch, data["slots"], data["keep"])
    for runs in data["warm"]:                                   # TF_WARM_STARTS, alone, before serving
        dec.finish([_admit(dec, _ids(runs), 1)])
    reqs = data["requests"]
    events = sorted([(r["start"], 1, i) for i, r in enumerate(reqs)] + [(r["end"], 0, i) for i, r in enumerate(reqs)])
    live, cached = {}, [None] * len(reqs)
    for _, kind, i in events:
        if kind == 0:
            if i in live:
                dec.finish([live.pop(i)])
                _lone_to_solo(dec)
            continue
        s = _admit(dec, _ids(reqs[i]["ids"]), reqs[i]["tokens"])
        live[i] = s
        cached[i] = s.cached
        _lone_to_solo(dec)
    return reqs, cached


def test_the_evenings_long_follow_ups_resume_from_their_previous_prompt(monkeypatch):
    reqs, cached = _replay(monkeypatch)
    long = [(r, c) for r, c in zip(reqs, cached) if r["parent_end"] and len(_ids(r["ids"])) >= 80000]
    assert [r["cached_obs"] for r, _ in long] == [69823, 16607, 16607, 16607]      # what the house did
    assert [c for _, c in long] == [r["parent_end"] for r, _ in long] == [69823, 81898, 84314, 86135]


def test_no_follow_up_in_jakes_sessions_loses_its_previous_prompt(monkeypatch):
    reqs, cached = _replay(monkeypatch)
    follow = [(r, c) for r, c in zip(reqs, cached) if r["jake"] and r["parent_end"]]
    assert len(follow) == 45
    assert [c for _, c in follow] == [r["parent_end"] for r, _ in follow]
    lost_before = sum(r["parent_end"] - r["cached_obs"] for r, _ in follow)
    assert lost_before == 203956                                 # tokens the house prefilled again


def test_the_replay_matches_the_simulation_request_for_request(monkeypatch):
    reqs, cached = _replay(monkeypatch)
    assert cached == [r["cached_sim"] for r in reqs]
