"""D4 (TF_DECODE_AHEAD): decode rounds ask for their next window's n-gram table pages row by row while the draft chain
runs. The rows asked for are exactly the ones the next round's ``stage`` gathers, each once, and drafts (so tokens)
are the same with and without it. Residency only."""

from types import SimpleNamespace

import numpy as np
import torch

from tensorfold.families.qwen4_exp.cuda import decode, decode_ahead, hostprep, multi
from tensorfold.families.qwen4_exp.cuda.ngram import NGram


class Table:
    rows = 1 << 40

    def __init__(self):
        self.asked = []

    def willneed(self, ids):
        self.asked.append(np.asarray(ids).copy())
        return 0


def _ngram():
    return NGram(vocab=1000, ngram_size=3, heads_per_ngram=2, vocab_base=5000, divisor=128, shards=1, seed=0,
                 eos=999, embed_dim=64)


def _w(table, ngram):
    ple = SimpleNamespace(table=table, ngram=ngram)
    return SimpleNamespace(x3=object(), comm=None, layers=[SimpleNamespace(ple=None), SimpleNamespace(ple=ple)])


def _now(monkeypatch):
    monkeypatch.setattr(decode_ahead, "_submit", lambda job: job())


def test_rows_asked_are_the_windows_stage_ids_each_once(monkeypatch):
    _now(monkeypatch)
    ngram, table = _ngram(), Table()
    ahead = decode_ahead.rows(_w(table, ngram))
    rng = np.random.default_rng(1)
    hist = {k: rng.integers(0, 998, 2) for k in "abc"}
    hist["c"] = np.array([5, 999])                    # an EOS in the history resets the n-grams
    wins = {"a": [7, 8, 9, 10], "b": [999, 3], "c": [4]}
    for k in wins:
        ahead.start(k, hist[k], wins[k][0])
    ahead.send()
    for j in range(1, 4):
        for k in wins:
            if j < len(wins[k]):
                ahead.add(k, wins[k][j])
        ahead.send()
    ahead.send()                                      # nothing new: no job
    got = np.concatenate([a.reshape(-1) for a in table.asked])
    want = []
    for j in range(4):                                # send order: depth by depth, streams in start order
        for k in wins:
            if j < len(wins[k]):
                want.append(ngram.ids(hist[k], np.asarray(wins[k]))[j].reshape(-1))
    assert len(table.asked) == 4 and np.array_equal(got, np.concatenate(want))
    assert ahead.sent == sum(len(v) for v in wins.values())


def test_off_or_no_host_table_asks_nothing(monkeypatch):
    table = Table()
    w = _w(table, _ngram())
    assert decode_ahead.rows(SimpleNamespace(x3=None, layers=w.layers)) is None
    assert decode_ahead.rows(SimpleNamespace(x3=object(), layers=[SimpleNamespace(ple=None)])) is None
    monkeypatch.setattr(decode_ahead, "ON", False)
    assert decode_ahead.rows(w) is None
    decode_ahead.start(None, [("a", SimpleNamespace(ple_history=np.zeros(2)), 1)])     # a no-op
    r = decode_ahead.Rows(SimpleNamespace(table=table, ngram=_ngram()))
    decode_ahead.start(r, [("a", SimpleNamespace(ple_history=None), 1)])             # no n-gram layers: skipped
    assert r.win == {}


# --- the shared rounds' draft chain (MultiDecoder._draft_all) -------------------------------------------------------

class St:
    def __init__(self, pos):
        self.pos, self.mtp_len, self.mtp_drafted = pos, pos, 0
        self.ple_history = np.array([11, 12])

    def set_mtp_len(self, n):
        self.mtp_len = n


def _decoder(w, depth, script):
    """A MultiDecoder whose MTP steps and picks are scripted: ``script[j][sid]`` is (draft, probability) at depth j."""

    m = object.__new__(multi.MultiDecoder)
    m.w, m.depth, m.confidence, m.cost, m.timing, m.joint, m.rounds = w, depth, 0.5, 0.0, None, None, None
    m.mbuf = SimpleNamespace()
    m.buf = SimpleNamespace(streams=torch.zeros(16, 1))
    log = []
    state = {"j": 0, "rows": []}

    def mtp_windows(windows):
        log.append(("mtp", len(windows)))
        state["rows"] = [st for st, _, _ in windows]
        return None, [None] * len(windows)

    def picks(logits, positions, samplings):
        j = state["j"]
        state["j"] += 1
        log.append(("pick", j))
        return [script[j][st.sid] for st in state["rows"]]

    m._mtp_windows, m._picks = mtp_windows, picks
    return m, log


def _streams():
    out = []
    for sid in range(3):
        st = St(100 + sid)
        st.sid = sid
        out.append(SimpleNamespace(sid=sid, st=st, count=1000, out=[1], drafts=[], sampling=None))
    return out


SCRIPT = [{0: (21, 0.9), 1: (31, 0.9), 2: (41, 0.2)},       # stream 2 stops after its first draft (low)
          {0: (22, 0.9), 1: (32, 0.3)},                     # stream 1's second draft is low: dropped
          {0: (23, 0.9)}]


def test_draft_all_asks_each_draft_once_and_drafts_are_unchanged(monkeypatch):
    _now(monkeypatch)
    ngram, table = _ngram(), Table()
    w = _w(table, ngram)
    plain, _ = _decoder(w, 3, SCRIPT)
    ss = _streams()
    plain._draft_all([(s, 0, [5]) for s in ss])
    want = [list(s.drafts) for s in ss]
    assert want == [[21, 22, 23], [31], [41]]

    m, log = _decoder(w, 3, SCRIPT)
    ss = _streams()
    ahead = decode_ahead.rows(w)
    decode_ahead.start(ahead, [(s.sid, s.st, 50 + s.sid) for s in ss])
    m._draft_all([(s, 0, [5]) for s in ss], ahead=ahead)
    assert [list(s.drafts) for s in ss] == want
    for s in ss:                                      # the window the next round stages: [end] + drafts
        assert ahead.win[s.sid][1] == [50 + s.sid] + want[s.sid] and ahead.win[s.sid][2] == len(want[s.sid]) + 1
    got = np.concatenate([a.reshape(-1) for a in table.asked])
    rows = sum(ngram.ids(np.array([11, 12]), np.asarray([50 + s] + want[s])).shape[0] for s in range(3))
    assert got.size == rows * ngram.heads           # every row once
    assert len(table.asked) == 4                      # rows 0, then one job per depth with new drafts


def test_draft_all_without_drafting_streams_still_asks_rows_0(monkeypatch):
    _now(monkeypatch)
    table = Table()
    w = _w(table, _ngram())
    m, _ = _decoder(w, 3, SCRIPT)
    m.mbuf = None                                     # no MTP head: nothing drafts
    ss = _streams()
    ahead = decode_ahead.rows(w)
    decode_ahead.start(ahead, [(s.sid, s.st, 7) for s in ss])
    m._draft_all([(s, 0, [5]) for s in ss], ahead=ahead)
    assert len(table.asked) == 1 and ahead.sent == 3


# --- a lone stream's chain (decode.draft) ---------------------------------------------------------------------------

class Eng:
    def __init__(self, script):
        self.st = St(10)
        self.script, self.j = script, 0
        self.mbuf = SimpleNamespace(streams=torch.zeros(8, 1))
        self.timing = None

    def sample_draft(self, logits, position, sampling):
        got = self.script[self.j]
        self.j += 1
        return got

    def mtp_forward(self, tokens, streams):
        return None


def test_lone_draft_chain_same_drafts_and_rows(monkeypatch):
    _now(monkeypatch)
    monkeypatch.setattr(decode, "absorb", lambda e, streams, next_tokens: None)
    script = [(5, 0.9), (6, 0.8), (7, 0.1)]
    plain = decode.draft(Eng(script), None, [1], 11, 4, None, confidence=0.5)
    ngram, table = _ngram(), Table()
    ahead = decode_ahead.rows(_w(table, ngram))
    e = Eng(script)
    decode_ahead.start(ahead, [(0, e.st, 4)])
    got = decode.draft(e, None, [1], 11, 4, None, confidence=0.5, ahead=ahead)
    assert got == plain == [5, 6]
    assert ahead.win[0][1] == [4, 5, 6] and ahead.sent == 3
    asked = np.concatenate([a.reshape(-1) for a in table.asked])
    assert np.array_equal(asked, ngram.ids(np.array([11, 12]), np.array([4, 5, 6])).reshape(-1))


# --- done line: stage ms and major faults in replayed rounds --------------------------------------------------------

def test_times_charge_stage_only_in_replayed_rounds():
    t = hostprep.Times()
    t.start()
    t.staged(0.002, 3)
    t.launched(True)
    t.start()
    t.staged(0.005, 7)
    t.launched(False)                                 # eager / capture round: not charged
    t.start()
    t.launched(True)                                  # no stage timing: nothing charged
    assert t.rounds == 2 and t.majflt == 3 and abs(t.stage - 0.002) < 1e-12
    s = t.summary()
    assert s.startswith("host_ms=prep:") and s.endswith(",stage:2.0,majflt:3")
    assert decode_ahead.faults() >= 0
