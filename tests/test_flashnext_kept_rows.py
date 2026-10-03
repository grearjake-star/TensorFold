"""House: kept prompt ends leave a big graph slot by their rows, not the slot's capacity (host, stand-in slots)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.cuda import graphs as graphs_module
from tensorfold.families.qwen4_exp.cuda import multi as multi_module
from tensorfold.families.qwen4_exp.cuda.graphs import Graphs as RealGraphs
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder

DEPTH, ROWS = 3, 4


class Graphs:
    made = 0

    def __init__(self, e, *, max_rows=8):
        self.e, self.keys = e, set()
        Graphs.made += 1

    _bucket = RealGraphs._bucket

    def round(self, R):
        st = self.e.st
        self.keys.add((R, st.cur[0], self._bucket(st.pos + R)))
        self.keys.add(("mtp", R, self._bucket(st.mtp_len + R)))

    def warm(self, rows=None, every_bucket=False):
        st = self.e.st
        for b in [self._bucket(1)]:
            for R in range(1, ROWS + 1):
                self.keys |= {(R, 0, b), (R, 1, b), ("mtp", R, b)}


class Slot:
    def __init__(self):
        self.capacity, self.limit, self.pos, self.mtp_len, self.cur = FIRST, 262144, 0, 0, [0]
        self.image_positions, self.kv_dtype, self.resizes, self.copied = None, "bf16", 0, []

    def cache_bytes(self, rows=None):
        return rows or self.capacity

    def layer_bytes(self, rows=None):
        return 0

    def resize(self, rows):
        before, self.capacity, self.resizes = self.capacity, rows, self.resizes + 1
        return rows - before

    def reset(self, w):
        self.pos = self.mtp_len = 0

    def set_pos(self, n):
        self.pos = n

    def set_mtp_len(self, n):
        self.mtp_len = n

    def copy_from(self, other):
        self.copied.append(("all", other.capacity))
        self.pos, self.mtp_len = other.pos, other.mtp_len

    def copy_prefix(self, source, pos, mtp_len):
        assert pos <= min(self.capacity, source.pos) and mtp_len <= min(self.capacity, source.mtp_len)
        self.copied.append(("rows", pos))


@pytest.fixture
def decoder(monkeypatch):
    monkeypatch.setattr(graphs_module, "Graphs", Graphs)
    monkeypatch.setattr(multi_module, "_tensors", lambda st: iter(()))

    def make(slots):
        dec = object.__new__(MultiDecoder)
        dec.w, dec.planning, dec.link, dec.depth, dec.keep = SimpleNamespace(comm=None), False, None, DEPTH, 8
        dec.free = [Slot() for _ in range(slots)]
        dec.slots = list(dec.free)
        dec.streams, dec.filling, dec.fills, dec.held, dec.kept = {}, [], {}, {}, []
        dec.memory_gate, dec.next_id, dec.solo_on = MemoryGate(1 << 40, 0), 0, True
        dec._release_pinned = lambda need: False
        dec.solo = SimpleNamespace(st=dec.free[0], rows=ROWS)
        dec.solo.graphs = Graphs(dec.solo)
        dec.solo.graphs.warm(ROWS)
        return dec
    return make


def _request(dec, prompt, tokens=64):
    """Admission, the kept prompt end, then lone rounds as _make_room grows the slot, and the finish."""

    s = SimpleNamespace(prompt=list(prompt), sid=dec.next_id, draft=True, vision=None)
    dec.next_id += 1
    s.st, _, s.cached = dec._slot_for(list(prompt), True)
    dec._grow(s.st, len(prompt) + DEPTH + 2, alone=True)
    s.st.pos = s.st.mtp_len = len(prompt)
    dec._remember(list(prompt[:-1]), s.st, {"pos": len(prompt) - 1, "mtp_len": len(prompt) - 2}, None)
    dec.streams[s.sid] = s
    keys = None
    for r in range(tokens // 2):
        dec._grow(s.st, s.st.pos + DEPTH + 2 + ROWS, alone=True)
        if s.st is not dec.solo.st:
            dec._move_to_solo(s)
        keys = set(dec.solo.graphs.keys) if keys is None else keys
        dec.solo.graphs.round(1 + r % ROWS)
        s.st.cur = [1 - s.st.cur[0]]
        s.st.pos += 2
        s.st.mtp_len += 2
    new = dec.solo.graphs.keys - keys
    dec.finish([s])
    return s, new


def _text(seed, n):
    return [(seed * 7919 + 13 * i) % 150000 + 1 for i in range(n)]



def test_a_fresh_prompt_moves_a_big_graph_slots_kept_end_by_its_rows(decoder):
    dec = decoder(2)
    slot = dec.solo.st
    _request(dec, _text(1, 3000))
    slot.resize(65536)                                # the graph slot grew with an earlier long conversation
    second, _ = _request(dec, _text(2, 500))
    other = next(st for st in dec.slots if st is not slot)
    assert second.st is slot and other.copied == [("rows", 2999)] and other.capacity == 8192
    assert [k[1] is other for k in dec.kept] == [True, False]


def test_a_relocated_kept_end_takes_only_its_rows(decoder):
    dec = decoder(3)
    slot = dec.solo.st
    _request(dec, _text(1, 3000))
    slot.resize(65536)
    spare = dec.free[0]
    assert dec._relocate_kept(slot, None) and spare.copied == [("rows", 2999)] and spare.capacity == 8192
