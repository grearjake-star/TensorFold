"""House: joint draft pricing for --parallel's shared rounds (TF_JOINT_PRICE=1), host only (stand-in MTP steps).

The rule (draft_cost.JointBars): a stream's next draft needs its chain's running product to reach the aggregate
rate x (the round's marginal verify row at its total rows + one MTP step). Drafts never change the output: pricing
only changes how many of the same keyed drafts a stream proposes.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import multi as multi_module
from tensorfold.families.qwen4_exp.cuda.draft_cost import (JOINT_ROWS, JointBars, cost_bars, joint_bars,
                                                           joint_rate, joint_settings, window_ms)
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

HOUSE = (27.18, 32.68, 36.42, 40.36, 42.52, 45.47, 48.95, 51.84, 54.76, 56.65, 60.68)     # ops/env.sh TF_VERIFY_MS
DMS, COST = 1.18, 0.06


# ---- the rule -------------------------------------------------------------------------------------------------

def test_settings_default_off_and_parsed():
    assert joint_settings({}) is None
    assert joint_settings({"TF_JOINT_PRICE": "0"}) is None
    assert joint_settings({"TF_JOINT_PRICE": "1"}) == {"rate": None, "verify": None}
    got = joint_settings({"TF_JOINT_PRICE": "1", "TF_JOINT_RATE": "0.11", "TF_JOINT_VERIFY_MS": "30,34,37"})
    assert got == {"rate": 0.11, "verify": (30.0, 34.0, 37.0)}
    for bad in ({"TF_JOINT_PRICE": "yes"}, {"TF_JOINT_PRICE": "1", "TF_JOINT_RATE": "0"},
                {"TF_JOINT_PRICE": "1", "TF_JOINT_VERIFY_MS": "30,20"}):
        with pytest.raises(ValueError):
            joint_settings(bad)


def test_window_ms_extends_the_table_by_its_last_step():
    assert window_ms(HOUSE, 1) == HOUSE[0]
    assert window_ms(HOUSE, 11) == HOUSE[10]
    step = HOUSE[10] - HOUSE[9]
    assert window_ms(HOUSE, 13) == pytest.approx(HOUSE[10] + 2 * step)


def test_one_stream_is_the_single_stream_rule_exactly():
    """n = 1: rate = cost, rows = 1 + drafts, so every bar is cost_bars' bar at that depth."""

    assert joint_rate(COST, 1, HOUSE) == COST
    single = cost_bars(COST, 10, (HOUSE, DMS))
    jb = joint_bars({"rate": None, "verify": None}, COST, (HOUSE, DMS), 1, 1)
    for j in range(10):
        assert jb.bar() == pytest.approx(single[j])
        assert jb.bar(1) == pytest.approx(single[j + 1])
        jb.take()


def test_aggregate_rate_grows_with_streams():
    r = [joint_rate(COST, n, HOUSE) for n in (1, 2, 4, 8)]
    assert r == sorted(r) and r[0] == COST
    # c4 on the house table: 0.06 x 4 x V(4) / V(16) ~ 0.12 tokens a ms (R-SPEC: c4 measured ~0.11)
    assert r[2] == pytest.approx(COST * 4 * window_ms(HOUSE, JOINT_ROWS) / window_ms(HOUSE, 16))
    assert 0.10 < r[2] < 0.14
    # a shared table prices the n-stream window
    shared = tuple(x * 0.5 for x in HOUSE)
    assert joint_rate(COST, 4, HOUSE, shared) == pytest.approx(2 * r[2])


def test_bar_is_the_marginal_row_at_the_rounds_total_rows():
    jb = JointBars(0.1, HOUSE, DMS, rows=5)
    assert jb.bar() == pytest.approx(0.1 * (HOUSE[5] - HOUSE[4] + DMS))
    jb.take()
    assert jb.rows == 6
    assert jb.bar() == pytest.approx(0.1 * (HOUSE[6] - HOUSE[5] + DMS))
    far = JointBars(0.1, HOUSE, DMS, rows=20)                  # past the table: its last step
    assert far.bar() == pytest.approx(0.1 * (HOUSE[10] - HOUSE[9] + DMS))


def test_fixed_rate_and_shared_table_override():
    jb = joint_bars({"rate": 0.2, "verify": (10.0, 12.0, 15.0)}, COST, (HOUSE, DMS), 4, 2)
    assert jb.rate == 0.2
    assert jb.bar() == pytest.approx(0.2 * (15.0 - 12.0 + DMS))


# ---- the shared round's draft loop (stand-in MTP steps) -------------------------------------------------------

def _token(sid, pos):
    return 1000 * sid + pos                     # the keyed draft at a position: one value, whatever the pricing


class St:
    def __init__(self, pos, head=0):
        self.pos, self.mtp_len, self.mtp_drafted, self.mtp_head = pos, pos, 0, head
        self.image_positions = None

    def set_mtp_len(self, n):
        self.mtp_len = n


def _decoder(monkeypatch, *, joint, depth=6, confidence=0.0, cost=COST):
    dec = object.__new__(MultiDecoder)
    dec.w = SimpleNamespace(mtp_heads=None)
    dec.depth, dec.confidence, dec.cost, dec.timing = depth, confidence, cost, (HOUSE, DMS)
    dec.joint = joint_settings({"TF_JOINT_PRICE": "1"} if joint else {})
    dec.rounds = None
    dec.buf = SimpleNamespace(streams=torch.zeros(64, 4))
    dec.mbuf = SimpleNamespace(streams=torch.zeros(64, 4))
    dec.launches = []

    def stage(w, b, windows):
        segs, a = [], 0
        for st, tokens, _ in windows:
            segs.append((st, a, a + len(tokens)))
            a += len(tokens)
        return segs

    def mtp(segs):
        dec.launches.append([st.mtp_head for st, _, _ in segs])
        assert len({st.mtp_head for st, _, _ in segs}) == 1             # one head a launch
        for st, a0, a1 in segs:
            dec.mbuf.streams[a1 - 1] = st.sid
        return torch.tensor([[float(st.sid)] for st, _, _ in segs])

    monkeypatch.setattr(multi_module, "mtp_stage", stage)
    dec._mtp = mtp
    return dec


def _streams(n, probs, heads=None):
    out = []
    for i in range(n):
        st = St(100 * (i + 1), head=(heads[i] if heads else 0))
        st.sid = i
        out.append(SimpleNamespace(sid=i, st=st, count=1000, out=[1], drafts=[], sampling=None))
    return out


def _picks_from(probs):
    def picks(logits, positions, samplings):
        got = []
        for row, pos in zip(logits[:, 0].tolist(), positions):
            sid = int(row)
            depth = pos - (100 * (sid + 1) + 1)
            got.append((_token(sid, pos), probs[sid][depth]))
        return got
    return picks


def _run(monkeypatch, joint, n, probs, heads=None, **kw):
    dec = _decoder(monkeypatch, joint=joint, **kw)
    dec._picks = _picks_from(probs)
    ss = _streams(n, probs, heads)
    dec._draft_all([(s, 0, [7]) for s in ss], rows=n)
    return dec, ss


MIDDLING = [0.9, 0.8, 0.75, 0.7, 0.7, 0.7, 0.7]          # chain: .9 .72 .54 .38 .26 .19 .13
SURE = [0.99] * 7


def test_drafts_are_the_same_keyed_tokens_only_fewer_under_joint_pricing(monkeypatch):
    probs = {i: (SURE if i == 0 else MIDDLING) for i in range(4)}
    _, single = _run(monkeypatch, False, 4, probs)
    dec, joint = _run(monkeypatch, True, 4, probs)
    for a, b in zip(single, joint):
        # the same draft at every position (serial's keyed pick): one list is a prefix of the other
        assert a.drafts[:len(b.drafts)] == b.drafts
        assert a.drafts == [_token(a.sid, a.st.pos + 1 + j) for j in range(len(a.drafts))]
        assert b.drafts and len(b.drafts) <= len(a.drafts)
    # the middling chains stop earlier at c4's aggregate rate and marginal row; the sure chain still goes deep
    assert sum(len(s.drafts) for s in joint) < sum(len(s.drafts) for s in single)
    assert len(joint[0].drafts) >= 5


def test_one_stream_round_drafts_exactly_as_single_pricing(monkeypatch):
    for probs in ({0: MIDDLING}, {0: SURE}, {0: [0.6, 0.5, 0.95, 0.9, 0.9, 0.9, 0.9]}):
        _, single = _run(monkeypatch, False, 1, probs)
        _, joint = _run(monkeypatch, True, 1, probs)
        assert single[0].drafts == joint[0].drafts


def test_joint_off_by_default_matches_single_bars(monkeypatch):
    monkeypatch.delenv("TF_JOINT_PRICE", raising=False)
    assert joint_settings() is None
    probs = {i: MIDDLING for i in range(3)}
    _, a = _run(monkeypatch, False, 3, probs)
    bars = cost_bars(COST, 6, (HOUSE, DMS))
    chain, want = 1.0, 0
    for j, p in enumerate(MIDDLING[:6]):                     # decode.draft's rule, stream by stream
        chain *= p
        if j > 0 and chain < bars[j]:
            break
        want += 1
        if chain < bars[j + 1]:
            break
    assert [len(s.drafts) for s in a] == [want] * 3


def test_first_draft_always_kept_and_confidence_floor_still_applies(monkeypatch):
    low = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    _, ss = _run(monkeypatch, True, 4, {i: low for i in range(4)})
    assert all(len(s.drafts) == 1 for s in ss)
    _, ss = _run(monkeypatch, True, 2, {i: [0.99, 0.4, 0.99, 0.99, 0.99, 0.99, 0.99] for i in range(2)},
                 confidence=0.5)
    assert all(len(s.drafts) == 1 for s in ss)


def test_rows_count_streams_that_do_not_draft(monkeypatch):
    """The next round's other live streams' pending rows raise the round's rows: drafts are priced there."""

    probs = {0: MIDDLING}
    dec = _decoder(monkeypatch, joint=True)
    dec._picks = _picks_from(probs)
    alone = _streams(1, probs)
    dec._draft_all([(alone[0], 0, [7])], rows=1)
    dec = _decoder(monkeypatch, joint=True)
    dec._picks = _picks_from(probs)
    crowd = _streams(1, probs)
    dec._draft_all([(crowd[0], 0, [7])], rows=8)               # seven more streams verify beside it
    assert len(crowd[0].drafts) <= len(alone[0].drafts)
    assert alone[0].drafts[:len(crowd[0].drafts)] == crowd[0].drafts
