"""Shared-round CUDA graphs share one private memory pool but replay in LRU/key order, not capture order.

PyTorch's pool-sharing rule: a graph captured later may be given memory an earlier graph freed at the end of its
capture (its temporaries). Every replay of the earlier graph writes those temporaries again, so a later graph's
OUTPUT placed there would be overwritten whenever the earlier graph replays before the output is read. The first
test shows that hazard on a toy pair of graphs. The engine is not exposed to it: every round graph's output (main
logits, MTP draft logits) is a view of the decoder's persistent ``Buffers.logits``, allocated before any capture and
outside the pool, so only temporaries live in the pool, and a temporary is written before it is read within its own
graph. The second test pins that invariant for every graph captured; the third replays graphs in an order far from
their capture order (two graphs kept, so keys are dropped and recaptured, streams joining and leaving) and checks
every reply against the serial engine.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95),
             Sampling(seed=9, top_k=40, top_p=0.9, temperature=0.7)]


def _inside(t: torch.Tensor, buf: torch.Tensor) -> bool:
    lo, size = buf.untyped_storage().data_ptr(), buf.untyped_storage().nbytes()
    p = t.data_ptr()
    return lo <= p and p + t.numel() * t.element_size() <= lo + size


def _ref(w, prompt, count, sampling):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    return serial_decode(e, prefill(e, list(prompt), sampling), count, sampling).tokens


def test_pytorch_reuses_an_earlier_graphs_temporaries_for_a_later_graphs_output():
    """The hazard itself, on two toy graphs in one pool: B's output takes A's freed temporary; replaying A after B
    overwrites it. (Skipped if this allocator version does not place B's output there.)"""

    x = torch.ones(1 << 20, device="cuda")
    (x * 2.0).sum(), x * 3.0                              # kernels loaded before capture
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga, pool=pool):
        tmp = x * 2.0
        span = (tmp.data_ptr(), tmp.data_ptr() + tmp.numel() * tmp.element_size())
        out_a = tmp.sum().reshape(1)
        del tmp
    with torch.cuda.graph(gb, pool=pool):
        out_b = x * 3.0
    if not span[0] <= out_b.data_ptr() < span[1]:
        pytest.skip("this allocator placed B's output outside A's temporaries")
    gb.replay()
    torch.cuda.synchronize()
    assert bool((out_b == 3.0).all())
    ga.replay()
    torch.cuda.synchronize()
    assert bool((out_b == 2.0).all()), "A's replay wrote its temporary over B's output"
    assert float(out_a) == 2.0 * x.numel()


@pytest.fixture
def captured(monkeypatch):
    """Every round graph's output and capture index, including graphs the LRU later drops."""

    outs = []
    capture = RoundGraphs._capture

    def recording(self, fn):
        g, out = capture(self, fn)
        g.tf_capture_index = len(outs)                  # on the graph: ids of dropped graphs are reused
        outs.append(out)
        return g, out

    monkeypatch.setattr(RoundGraphs, "_capture", recording)
    return outs


def _drain(dec, streams, slots, joins=()):
    """Admit ``streams``; the last ``len(joins)`` join once at least that many rounds ran and a slot is live-free."""

    first = len(streams) - len(joins)
    for s in streams[:first]:
        dec.admit(s)
    waiting, pending, rounds = list(streams[first:]), list(joins), 0
    while dec.live() or waiting:
        if not dec.live():
            dec.admit(waiting.pop(0))
            pending.pop(0)
            continue
        dec.finish(dec.round())
        rounds += 1
        if waiting and rounds >= pending[0] and len(dec.streams) + len(dec.filling) < slots:
            dec.admit(waiting.pop(0))
            pending.pop(0)


def test_every_round_graphs_output_is_a_persistent_buffer_view(captured):
    outs = captured
    w = _model()
    counts = [30, 22, 40, 17]
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=4, confidence=0.3, graphs=True, round_graphs=True)
    dec.rounds.after = 1
    _drain(dec, [Stream(list(p), c, smp) for p, c, smp in zip(PROMPTS, counts, SAMPLINGS)], 4, joins=(3,))
    kinds = {k[0] for k in dec.rounds.graphs}
    assert {"main", "mtp"} <= kinds and outs
    for out in outs:
        assert _inside(out, dec.buf.logits) or _inside(out, dec.mbuf.logits), \
            "a round graph's output lives in the shared graph pool"


@pytest.mark.parametrize("limit", [2, 3])
def test_replays_out_of_capture_order_keep_every_streams_bits(captured, monkeypatch, limit):
    """Two or three graphs kept: keys are dropped and recaptured all the time, and a graph captured earlier replays
    right after one captured later in the same round (main, then MTP steps). Every reply == serial."""

    inversions = [0]
    run = RoundGraphs.run

    def watched(self, key, fn):
        out = run(self, key, fn)
        hit = self.graphs.get(key)
        index = getattr(hit[0], "tf_capture_index", -1) if hit is not None else -1
        if hit is not None and getattr(self, "_last", -1) > index:
            inversions[0] += 1
        self._last = index
        return out

    monkeypatch.setattr(RoundGraphs, "run", watched)
    w = _model(5)
    g = torch.Generator().manual_seed(21)
    prompts = [PROMPTS[0], PROMPTS[1], torch.randint(1, V, (90,), generator=g).tolist(), PROMPTS[3]]
    counts = [50, 12, 24, 20]
    refs = [_ref(w, p, c, smp) for p, c, smp in zip(prompts, counts, SAMPLINGS)]
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=4, confidence=0.3, prefill_rows=32, graphs=True,
                       round_graphs=True)
    dec.rounds.after, dec.rounds.limit = 1, limit
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(prompts, counts, SAMPLINGS)]
    _drain(dec, streams, 3, joins=(4, 9))
    assert [s.out for s in streams] == refs
    assert dec.rounds.dropped > 0 and dec.rounds.replays > 0
    assert inversions[0] > 0, "no replay ran out of capture order"
