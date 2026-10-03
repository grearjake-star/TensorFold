"""W5-1: a lone stream under --parallel replays the one-stream CUDA graphs in a graph slot, with the same bits as the
eager shared rounds and the single-stream engine; moving in and out of the slot (a second stream joins, leaves,
a different lone request takes the slot) keeps every reply's bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]


def _ref(w, prompt, count, sampling, kv_dtype="bf16"):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
    return serial_decode(e, prefill(e, list(prompt), sampling), count, sampling).tokens


def _run(dec, prompt, count, sampling, draft=True):
    s = Stream(list(prompt), count, sampling, draft=draft)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    return s


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_a_lone_stream_replays_the_graphs_with_its_eager_bits(kv_dtype):
    w = _model()
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, graphs=True)
    eager = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
    assert dec.solo is not None and eager.solo is None
    for prompt, sampling in zip(PROMPTS, SAMPLINGS):
        before = dec.solo_rounds
        s = _run(dec, prompt, 24, sampling)
        assert s.out == _ref(w, prompt, 24, sampling, kv_dtype) == _run(eager, prompt, 24, sampling).out
        assert dec.solo_rounds > before                       # its rounds went through the graphs
    assert eager.solo_rounds == 0


def test_streams_move_in_and_out_of_the_graph_slot_with_their_bits():
    """Two streams decode together (eager), the short one ends, the long one goes on alone in the graph slot (its
    held DeltaNet rows folded, its caches copied in), then a third joins (eager again) and leaves."""

    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, graphs=True)
    counts = [60, 8, 12]
    refs = [_ref(w, p, n, smp) for p, n, smp in zip(PROMPTS[:3], counts, SAMPLINGS[:3])]
    a, b = Stream(list(PROMPTS[0]), counts[0], SAMPLINGS[0]), Stream(list(PROMPTS[1]), counts[1], SAMPLINGS[1])
    dec.admit(b)                                              # the short one takes the graph slot
    dec.admit(a)
    assert b.st is dec.solo.st and a.st is not dec.solo.st    # so the long one moves in when b ends
    c, rounds, modes = None, 0, []
    while dec.live():
        before = dec.solo_rounds
        dec.finish(dec.round())
        modes.append(dec.solo_rounds > before)
        rounds += 1
        if b.done and c is None and len(a.out) > 25:
            c = Stream(list(PROMPTS[2]), counts[2], SAMPLINGS[2])
            dec.admit(c)
    assert [a.out, b.out, c.out] == refs
    # eager while two decode, graphs alone, eager again with the third, graphs after it ends
    assert not modes[0] and True in modes and modes.index(True) > 0
    assert any(not m for m in modes[modes.index(True):])
    assert dec.solo_moves >= 1 and a.st is dec.solo.st


@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_a_different_lone_request_keeps_the_graph_slot_and_its_graphs(sampling):
    """Requests one after another: the first prompt's kept end moves out of the graph slot rather than the graphs
    moving; the slot keeps its rows; a later turn that extends the first prompt still resumes from its kept end."""

    w = _model()
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, graphs=True)
    graphs = dec.solo.graphs
    first = _run(dec, PROMPTS[1], 16, sampling)
    assert first.st is dec.solo.st and first.out == _ref(w, PROMPTS[1], 16, sampling)
    captured = graphs.captures
    other = _run(dec, PROMPTS[3], 16, sampling)
    assert other.out == _ref(w, PROMPTS[3], 16, sampling)
    assert dec.solo.graphs is graphs and other.st is dec.solo.st     # same graphs object: nothing dropped
    assert any(k[0] == PROMPTS[1][:-1] and k[1] is not dec.solo.st for k in dec.kept)   # moved, not evicted
    longer = PROMPTS[1] + first.out[:-1] + [42, 43]
    turn = _run(dec, longer, 12, sampling)
    assert turn.cached == len(PROMPTS[1]) - 1 and turn.out == _ref(w, longer, 12, sampling)
    assert dec.solo.graphs is graphs and captured > 0


def test_a_lone_stream_grows_its_graph_slot_by_doubling_and_keeps_its_bits():
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, graphs=True)
    prompt = [(7 * i + 3) % (V - 1) + 1 for i in range(240)]
    smp = Sampling(seed=5, top_k=20, top_p=0.95)
    s = _run(dec, prompt, 80, smp)
    assert s.out == _ref(w, prompt, 80, smp)
    assert s.st is dec.solo.st and s.st.capacity == 1024 and s.st.version >= 1 and dec.solo_rounds > 0
    graphs = dec.solo.graphs
    _run(dec, PROMPTS[0], 4, None)                                # the long prompt's kept end moves to a free slot
    assert dec.solo.graphs is graphs and dec.solo.st.capacity == 1024   # an idle graph slot keeps its rows
    while dec._evict_kept(None):                                  # memory short: kept ends, then the slot's rows
        pass
    assert dec.solo.st.capacity == FIRST and dec.solo.graphs is not graphs


def test_warm_leaves_the_graph_slot_captured_and_empty():
    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, prefill_rows=16, graphs=True)
    dec.warm()
    assert dec.solo.graphs.captures > 0 and dec.solo.st.pos == 0 and not dec.live() and not dec.kept
    captured = dec.solo.graphs.captures
    s = _run(dec, PROMPTS[1], 16, None)
    assert s.out == _ref(w, PROMPTS[1], 16, None) and dec.solo.graphs.captures == captured


def test_copy_from_takes_a_smaller_state_into_its_first_rows():
    w = _model()
    small, big = State(w, 256, 4, limit=1024), State(w, 512, 4, limit=1024)
    small.kc[0].k.normal_()
    small.rec.normal_()
    small.set_pos(200)
    small.cur = [1] * len(small.cur)
    big.copy_from(small)
    assert big.capacity == 512 and big.pos == 200 and big.cur == small.cur
    assert torch.equal(big.kc[0].k[:256], small.kc[0].k) and torch.equal(big.rec, small.rec)
    with pytest.raises(ValueError):
        small.copy_from(big)


@pytest.mark.parametrize("slots", [3, 4])
def test_lone_requests_on_a_full_pool_never_recapture(slots):
    """Regression (tier D, 592ee62): once every slot held a kept prompt end, each new lone request swapped the graphs
    to another slot and recaptured them. Now a fresh prompt takes the graph slot at admission (its kept ends move to
    the slot the prompt would have taken) and a resumed one evicts the oldest other kept end, never the graphs."""

    w = _model()
    dec = MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3, graphs=True)
    graphs = dec.solo.graphs
    smp = Sampling(seed=3, top_k=20, top_p=0.95)
    prompts = [[(13 * i + 7 * j + 1) % (V - 1) + 1 for i in range(5 + j)] for j in range(2 * slots + 2)]
    for p in prompts:
        s = _run(dec, p, 12, smp)
        assert s.out == _ref(w, p, 12, smp) and s.st is dec.solo.st
        assert dec.solo.graphs is graphs
    a, b = prompts[0], prompts[1]                     # two conversations taking turns: each turn resumes its own end
    for turn in range(3):
        for i, p in enumerate((a, b)):
            s = _run(dec, p, 10, smp)
            assert s.out == _ref(w, p, 10, smp) and dec.solo.graphs is graphs
            nxt = p + s.out[:-1] + [40 + turn]
            if i == 0:
                a = nxt
            else:
                b = nxt
            if turn:
                assert s.cached > 0                   # resumed from its kept end, wherever it was moved


def test_a_move_without_room_for_the_copy_takes_the_graphs_to_the_stream():
    """Budget guard (0.6.3's absolute caps): when the graph slot can't grow to hold a lone stream's caches beside its
    own, the graphs move to the stream's slot (no copy, no error); the reply keeps its serial bits, and its solo
    rounds give a prompt arriving next a round time to size its first passes by."""

    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, graphs=True)
    grow = dec._grow
    dec._grow = lambda st, rows, **kw: False if dec._is_solo(st) and rows > st.capacity else grow(st, rows, **kw)
    counts = [40, 6]
    refs = [_ref(w, p, n, smp) for p, n, smp in zip(PROMPTS[:2], counts, SAMPLINGS[:2])]
    a, b = Stream(list(PROMPTS[0]), counts[0], SAMPLINGS[0]), Stream(list(PROMPTS[1]), counts[1], SAMPLINGS[1])
    dec.admit(b)                                              # b takes the graph slot
    dec.admit(a)
    b.st.resize(FIRST)                                        # the graph slot is smaller than a's caches
    a.st.resize(512)
    slot = dec.solo.st
    while dec.live():
        dec.finish(dec.round())
    assert [a.out, b.out] == refs
    assert dec.solo.st is a.st and dec.solo.st is not slot and dec.solo_rounds > 0
    assert dec.round_s is not None
