"""House: two MTP heads picked per conversation by prompt length (TF_HEAD_SWITCH_ROWS), host only (stand-ins).

Either head is exact (drafts are verified against serial's keyed samples); what must hold is that a sequence's MTP
cache is written by one head, that a launch reads one head, and that graphs never replay one head for the other.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import decode as decode_module
from tensorfold.families.qwen4_exp.cuda import graphs as graphs_module
from tensorfold.families.qwen4_exp.cuda import heads
from tensorfold.families.qwen4_exp.cuda import multi as multi_module
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.state import State


def _w(rows=16384, two=True):
    return SimpleNamespace(mtp="primary", mtp_heads=["primary", "long"] if two else None,
                           meta={"head_switch_rows": rows} if two else {})


def test_settings():
    assert heads.switch_rows({}) is None
    assert heads.switch_rows({"TF_HEAD_SWITCH_ROWS": "0"}) is None
    assert heads.switch_rows({"TF_HEAD_SWITCH_ROWS": "16384"}) == 16384
    for bad in ("x", "-1"):
        with pytest.raises(ValueError):
            heads.switch_rows({"TF_HEAD_SWITCH_ROWS": bad})
    assert heads.long_head({}) is None and heads.long_head({"TF_MTP_HEAD_LONG": "own"}) is None
    assert heads.long_head({"TF_MTP_HEAD_LONG": "/x/v4.safetensors"}) == "/x/v4.safetensors"


def test_pick_by_prompt_length_and_off_is_head_0():
    w = _w(16384)
    assert heads.pick(w, 100) == 0 and heads.pick(w, 16383) == 0
    assert heads.pick(w, 16384) == 1 and heads.pick(w, 40000) == 1
    assert heads.pick(_w(two=False), 10**6) == 0
    assert heads.pick(SimpleNamespace(mtp="m"), 10**6) == 0          # stand-in weights without the field


def test_a_launch_reads_its_states_one_head():
    w = _w()
    a, b = SimpleNamespace(mtp_head=0), SimpleNamespace(mtp_head=1)
    assert heads.of(w, [(a, 0, 1)]) == "primary"
    assert heads.of(w, [(b, 0, 1), (SimpleNamespace(mtp_head=1), 1, 2)]) == "long"
    with pytest.raises(ValueError):
        heads.of(w, [(a, 0, 1), (b, 1, 2)])
    assert heads.of(SimpleNamespace(mtp="m"), [(b, 0, 1)]) == "m"      # one head loaded: w.mtp, whatever the state
    assert heads.launch_key([(a, 0, 1)]) == () and heads.launch_key([(b, 0, 1)]) == (1,)
    assert heads.groups([a, b, a, b, b]) == [[0, 2], [1, 3, 4]]
    assert heads.groups([b, a]) == [[0], [1]]


class FakeSt:
    """The fields State.snapshot/restore touch, on the CPU."""

    def __init__(self):
        self.rec = [torch.zeros(2)]
        self.cur = [0]
        self.conv, self.ple_tail = torch.zeros(2), torch.zeros(2)
        self.ple_history, self.ple_last = None, None
        self.pos, self.mtp_len, self.mtp_drafted, self.mtp_head = 0, 0, 0, 0

    def set_pos(self, p):
        self.pos = p

    def set_rope_delta(self, d):
        pass

    def set_mtp_len(self, n):
        self.mtp_len = n

    def restore(self, snap):
        State.restore(self, snap)


def test_snapshots_carry_the_head_and_restore_keeps_it():
    st = FakeSt()
    st.pos, st.mtp_len, st.mtp_head = 30000, 30000, 1
    snap = State.snapshot(st)
    assert snap["mtp_head"] == 1
    other = FakeSt()
    State.restore(other, snap)
    assert other.mtp_head == 1 and other.mtp_len == 30000
    old = dict(snap)
    del old["mtp_head"]                               # a snapshot from before the switch: head 0
    State.restore(other, old)
    assert other.mtp_head == 0


def test_prefill_begin_picks_fresh_prompts_and_keeps_a_kept_ends_head(monkeypatch):
    w = _w(16384)
    st = FakeSt()
    e = SimpleNamespace(w=w, st=st, reset=lambda: (setattr(st, "mtp_head", 0), setattr(st, "mtp_len", 0)))
    assert decode_module.prefill_begin(e, [1] * 20000) == 0 and st.mtp_head == 1
    assert decode_module.prefill_begin(e, [1] * 2000) == 0 and st.mtp_head == 0
    # a kept end of a short conversation (head 0) resumed by a long prompt keeps head 0: its MTP cache is head 0's
    kept = FakeSt()
    kept.pos, kept.mtp_len, kept.mtp_head = 1500, 1500, 0
    snap = State.snapshot(kept)
    assert decode_module.prefill_begin(e, [1] * 20000, mtp=False, resume={"state": snap, "tail": None}) == 1500
    assert st.mtp_head == 0
    kept.mtp_head = 1
    decode_module.prefill_begin(e, [1] * 2000, mtp=False, resume={"state": State.snapshot(kept), "tail": None})
    assert st.mtp_head == 1
    # a kept end with no MTP cache yet: the prompt picks
    kept.mtp_len, kept.mtp_head = 0, 0
    decode_module.prefill_begin(e, [1] * 20000, mtp=False, resume={"state": State.snapshot(kept), "tail": None})
    assert st.mtp_head == 1


def _graphs(monkeypatch, w, st):
    captures = []
    monkeypatch.setattr(graphs_module, "mtp_stage", lambda w_, b, windows: [(windows[0][0], 0, len(windows[0][1]))])
    monkeypatch.setattr(graphs_module, "stage", lambda w_, b, windows: [(windows[0][0], 0, len(windows[0][1]))])
    monkeypatch.setattr(graphs_module, "mtp_compute", lambda w_, segs, b, **kw: heads.of(w_, segs))
    monkeypatch.setattr(graphs_module, "compute", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    g = graphs_module.Graphs.__new__(graphs_module.Graphs)
    g.max_rows, g.main, g.mtp, g.mtp_out, g.captures = 8, {}, {}, {}, 0
    g.e = SimpleNamespace(w=w, st=st, buf=SimpleNamespace(logits=torch.zeros(8), streams=torch.zeros(8, 2)),
                          mbuf=SimpleNamespace(), capacity=262144)

    def capture(fn):
        captures.append(fn())
        g.captures += 1
        return SimpleNamespace(replay=lambda: None)

    g._capture = capture
    return g, captures


def test_graphs_key_the_head_and_never_replay_one_for_the_other(monkeypatch):
    w = _w()
    st = SimpleNamespace(pos=100, mtp_len=100, cur=[0], capacity=262144, mtp_head=0)
    g, captures = _graphs(monkeypatch, w, st)
    assert g.mtp_forward([1, 2], None) == "primary"
    st.mtp_head = 1
    assert g.mtp_forward([1, 2], None) == "long"
    assert set(g.mtp) == {(2, 8192), (2, 8192, 1)}             # head 0 keeps today's key
    assert captures == ["primary", "long"]
    st.mtp_head = 0
    assert g.mtp_forward([1, 2], None) == "primary" and len(captures) == 2


def test_warm_captures_every_heads_mtp_steps_and_restores_the_states_head(monkeypatch):
    w = _w()
    st = SimpleNamespace(pos=0, mtp_len=0, cur=[0], capacity=262144, mtp_head=1)
    g, captures = _graphs(monkeypatch, w, st)
    g.warm(3)
    assert {k for k in g.mtp} == {(n, 8192) for n in (1, 2, 3)} | {(n, 8192, 1) for n in (1, 2, 3)}
    assert st.mtp_head == 1
    g1, _ = _graphs(monkeypatch, _w(two=False), SimpleNamespace(pos=0, mtp_len=0, cur=[0], capacity=262144))
    g1.e.w = SimpleNamespace(mtp="m")
    g1.warm(3)
    assert set(g1.mtp) == {(n, 8192) for n in (1, 2, 3)}            # one head: today's captures exactly


class St:
    def __init__(self, sid, pos, head):
        self.sid, self.pos, self.mtp_len, self.mtp_drafted, self.mtp_head = sid, pos, pos, 0, head
        self.image_positions = None

    def set_mtp_len(self, n):
        self.mtp_len = n


def test_shared_round_draft_steps_run_one_launch_a_head_in_stream_order(monkeypatch):
    dec = object.__new__(MultiDecoder)
    dec.w = _w()
    dec.depth, dec.confidence, dec.cost, dec.timing, dec.joint = 3, 0.0, 0.0, None, None
    dec.rounds = None
    dec.buf = SimpleNamespace(streams=torch.zeros(64, 2))
    dec.mbuf = SimpleNamespace(streams=torch.zeros(64, 2))
    launches = []

    def stage(w, b, windows):
        segs, a = [], 0
        for st, tokens, x in windows:
            assert x.shape[0] == len(tokens)
            segs.append((st, a, a + len(tokens)))
            a += len(tokens)
        return segs

    def mtp(segs):
        heads.of(dec.w, segs)                                       # raises on a mixed launch
        launches.append([st.sid for st, _, _ in segs])
        for st, _, a1 in segs:
            dec.mbuf.streams[a1 - 1] = float(st.sid)                # the stream's output row
        return torch.tensor([[float(st.sid)] for st, _, _ in segs])

    seen = []

    def picks(logits, positions, samplings):
        rows = [int(x) for x in logits[:, 0].tolist()]
        seen.append(rows)
        return [(10 * sid + pos, 0.9) for sid, pos in zip(rows, positions)]

    monkeypatch.setattr(multi_module, "mtp_stage", stage)
    dec._mtp, dec._picks = mtp, picks
    ss = [SimpleNamespace(sid=i, st=St(i, 100 * (i + 1), h), count=100, out=[1], drafts=[], sampling=None)
          for i, h in enumerate([0, 1, 0, 1])]
    dec._draft_all([(s, 0, [7]) for s in ss], rows=4)
    # absorb + 2 chained steps, each split by head: [0, 2] then [1, 3]
    assert launches == [[0, 2], [1, 3]] * 3
    assert seen == [[0, 1, 2, 3]] * 3                               # logits back in stream order
    for s in ss:
        assert s.drafts == [10 * s.sid + s.st.pos + 1 + j for j in range(3)]


def test_one_head_shared_round_is_one_launch(monkeypatch):
    dec = object.__new__(MultiDecoder)
    dec.w = SimpleNamespace(mtp="m", mtp_heads=None)
    dec.depth, dec.confidence, dec.cost, dec.timing, dec.joint = 2, 0.0, 0.0, None, None
    dec.rounds = None
    dec.buf = SimpleNamespace(streams=torch.zeros(64, 2))
    dec.mbuf = SimpleNamespace(streams=torch.zeros(64, 2))
    launches = []
    monkeypatch.setattr(multi_module, "mtp_stage",
                        lambda w, b, windows: [(st, i, i + 1) for i, (st, _, _) in enumerate(windows)])
    dec._mtp = lambda segs: (launches.append(len(segs)), torch.tensor([[float(st.sid)] for st, _, _ in segs]))[1]
    dec._picks = lambda logits, positions, samplings: [(p, 0.9) for p in positions]
    ss = [SimpleNamespace(sid=i, st=St(i, 10, 0), count=100, out=[1], drafts=[], sampling=None) for i in range(3)]
    dec._draft_all([(s, 0, [7]) for s in ss], rows=3)
    assert launches == [3, 3]
