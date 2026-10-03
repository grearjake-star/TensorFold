"""MG: shared --parallel rounds (2-4 streams) and their MTP draft steps replay CUDA graphs with the eager bits.

The graph-mode launches read every per-stream address and row position from persistent device tables and are
bounded by a context bucket; every stream must still emit exactly its solo (serial engine) tokens. The concurrent
round suites (test_flashnext_multi, test_flashnext_L7) are re-run here with round graphs on, capturing at a key's
first sighting so every shared round and draft step replays a graph.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _bf16_table, _cfg, _model, _ple, _Rand  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod, attn_multi, glue, multi  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_graphs import bucket  # noqa: E402

# the eager suites, collected again in this module (the autouse fixture below turns round graphs on for them)
from test_flashnext_multi import (  # noqa: E402,F401
    test_streams_decoded_together_equal_each_alone, test_prompts_that_extend_a_finished_stream_resume_from_its_slot,
    test_packed_passes_keep_each_prompt_one_token_early, test_sparse_streams_decode_together_as_alone,
    test_streams_keep_their_bits_when_their_caches_move, test_streams_that_grow_past_their_first_rows_equal_each_alone,
    test_prompts_fill_between_rounds_while_streams_decode, test_packed_prompt_passes_keep_each_prompt_its_solo_run)
from test_flashnext_L7 import (  # noqa: E402,F401
    test_a_lone_stream_replays_the_graphs_with_its_eager_bits, test_streams_move_in_and_out_of_the_graph_slot_with_their_bits,
    test_lone_requests_on_a_full_pool_never_recapture, test_a_move_without_room_for_the_copy_takes_the_graphs_to_the_stream)

DEV = "cuda"
MADE: list = []
# suites whose decoders never run a shared round with nothing filling (lone streams through the graph slot only)
LONE = {"test_a_lone_stream_replays_the_graphs_with_its_eager_bits", "test_lone_requests_on_a_full_pool_never_recapture",
        "test_prompts_that_extend_a_finished_stream_resume_from_its_slot"}


@pytest.fixture(autouse=True)
def round_graphs(monkeypatch, request):
    """Every MultiDecoder made in a test gets round graphs that capture at a key's first sighting."""

    init = MultiDecoder.__init__
    after = getattr(request, "param", 1)

    def patched(self, *a, **kw):
        kw.setdefault("round_graphs", True)
        init(self, *a, **kw)
        if self.rounds is not None:
            self.rounds.after = after
        MADE.append(self)

    monkeypatch.setattr(MultiDecoder, "__init__", patched)
    MADE.clear()
    yield
    name = request.node.originalname
    if name not in LONE and not name.startswith("test_kernel"):
        assert any(d.rounds is not None and d.rounds.replays > 0 for d in MADE), "no shared round replayed a graph"


def _ref(w, prompt, count, sampling, capacity=1024, kv_dtype="bf16", prefill_rows=16):
    e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows, kv_dtype=kv_dtype)
    return serial_decode(e, prefill(e, list(prompt), sampling), count, sampling).tokens


def _drain(dec, streams):
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())


PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95),
             Sampling(seed=9, top_k=40, top_p=0.9, temperature=0.7)]


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("depth", [3, 6])
def test_n_streams_on_round_graphs_equal_each_alone(n, depth):
    """2-4 streams with different lengths: every shared round and draft step replays a graph; each reply == serial."""

    w = _model()
    counts = [30, 22, 40, 17][:n]
    refs = [_ref(w, p, c, smp) for p, c, smp in zip(PROMPTS, counts, SAMPLINGS)]
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=depth, confidence=0.3, graphs=True)
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(PROMPTS[:n], counts, SAMPLINGS)]
    _drain(dec, streams)
    assert [s.out for s in streams] == refs
    g = dec.rounds
    assert g.replays > 0 and g.eager == 0 and g.captures == len(g.graphs) + g.dropped
    assert any(k[0] == "main" and k[1] == n for k in g.graphs) and any(k[0] == "mtp" for k in g.graphs)


@pytest.mark.parametrize("round_graphs", [2], indirect=True)
def test_keys_run_eagerly_until_captured_and_graphs_are_dropped_lru(round_graphs):
    """The default policy: a key's first sighting runs the graph-mode forward eagerly, the second captures it; past
    the limit the least recently used graph goes (and recaptures when it comes back), bits unchanged throughout."""

    w = _model()
    counts = [36, 28, 33]
    refs = [_ref(w, p, c, smp) for p, c, smp in zip(PROMPTS, counts, SAMPLINGS)]
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=4, confidence=0.3, graphs=True)
    dec.rounds.limit = 3
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(PROMPTS[:3], counts, SAMPLINGS)]
    _drain(dec, streams)
    assert [s.out for s in streams] == refs
    g = dec.rounds
    assert g.eager > 0 and g.replays > 0 and len(g.graphs) <= 3 and g.dropped > 0
    assert g.eager == len(g.seen)                     # each key ran eagerly exactly once


def test_streams_joining_and_leaving_keep_their_bits_on_round_graphs():
    """A third request arrives while two decode (its prompt fills in eager mixed rounds), one ends, another joins:
    round graphs serve every stream count and slot layout in between; every reply == serial."""

    w = _model(5)
    g = torch.Generator().manual_seed(21)
    prompts = [PROMPTS[0], PROMPTS[1], torch.randint(1, V, (90,), generator=g).tolist(), PROMPTS[3]]
    counts = [50, 12, 24, 20]
    refs = [_ref(w, p, c, smp) for p, c, smp in zip(prompts, counts, SAMPLINGS)]
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, prefill_rows=32, graphs=True)
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(prompts, counts, SAMPLINGS)]
    dec.admit(streams[0])
    dec.admit(streams[1])
    rounds, joined = 0, 1
    while dec.live() or joined < 3:
        dec.finish(dec.round())
        rounds += 1
        if rounds == 4:
            dec.admit(streams[2])
            joined = 2
        if joined == 2 and streams[1].done and len(dec.streams) + len(dec.filling) < 3:
            dec.admit(streams[3])
            joined = 3
    assert [s.out for s in streams] == refs
    assert {k[1] for k in dec.rounds.graphs if k[0] == "main"} >= {2, 3}


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_sparse_and_dense_streams_share_round_graphs_across_buckets(kv_dtype):
    """A stream past the attention budget (its own block select) beside dense ones, at a context bucket the short
    streams alone would not use: the graph-mode select gives qsa_rows' lists; every reply == serial."""

    w = _model(7)
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(1, V, (2300,), generator=g).tolist(), PROMPTS[1],
               torch.randint(1, V, (2060,), generator=g).tolist()]
    refs = [_ref(w, p, 20, smp, capacity=4096, kv_dtype=kv_dtype, prefill_rows=256) for p, smp in zip(prompts, SAMPLINGS)]
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=kv_dtype, graphs=True)
    streams = [Stream(list(p), 20, smp) for p, smp in zip(prompts, SAMPLINGS)]
    _drain(dec, streams)
    assert [s.out for s in streams] == refs
    assert any(k[0] == "main" and k[1] == 3 for k in dec.rounds.graphs)


def test_ple_streams_on_round_graphs_equal_each_alone():
    """With the n-gram (PLE) layer live, the graph rounds' one-launch conv (each stream's tail by table) == serial."""

    c = _cfg(ple=True)
    with tempfile.TemporaryDirectory() as tmp:
        table = _bf16_table(Path(tmp) / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(3)))
        counts = [26, 18, 31, 22]
        refs = [_ref(w, p, n, smp) for p, n, smp in zip(PROMPTS, counts, SAMPLINGS)]
        dec = MultiDecoder(w, slots=4, capacity=1024, depth=4, confidence=0.3, graphs=True)
        streams = [Stream(list(p), n, smp) for p, n, smp in zip(PROMPTS, counts, SAMPLINGS)]
        _drain(dec, streams)
        assert [s.out for s in streams] == refs
        assert dec.rounds.replays > 0


def test_kernel_ple_conv_multi_equals_each_streams_launch():
    torch.manual_seed(0)
    S_, D_, taps, dil = 4, 512, 4, 3
    wide = S_ * D_
    counts, firsts = [3, 1, 5], [40, 7, 1000]
    rows = sum(counts)
    gated = torch.randn((rows, wide), device=DEV).to(torch.bfloat16)
    pss = torch.rand((rows, S_), device=DEV) * D_ + 1
    nc = torch.randn((wide,), device=DEV).to(torch.bfloat16)
    cw = torch.randn((wide, taps), device=DEV).to(torch.bfloat16)
    h = torch.randn((rows, wide), device=DEV).to(torch.bfloat16)
    tails = [torch.randn(((taps - 1) * dil, wide), device=DEV).to(torch.bfloat16) for _ in counts]
    want_h, want_n = h.clone(), torch.zeros_like(h)
    a0 = 0
    for t, n in zip(tails, counts):
        glue.ple_conv(gated[a0:a0 + n], pss[a0:a0 + n], nc, t, cw, want_h[a0:a0 + n], want_h[a0:a0 + n],
                      want_n[a0:a0 + n], 1e-6, S_, dil)
        a0 += n
    sid = torch.tensor(sum(([i] * n for i, n in enumerate(counts)), []), dtype=torch.int32, device=DEV)
    posr = torch.tensor(sum((list(range(f, f + n)) for f, n in zip(firsts, counts)), []), dtype=torch.int32,
                        device=DEV)
    first = torch.tensor(firsts, dtype=torch.int32, device=DEV)
    ptrs = torch.tensor([t.data_ptr() for t in tails], dtype=torch.int64, device=DEV)
    got_h, got_n = h.clone(), torch.zeros_like(h)
    glue.ple_conv_multi(gated, pss, nc, ptrs, posr, sid, first, cw, got_h, got_h, got_n, 1e-6, S_, dil)
    assert torch.equal(got_h, want_h) and torch.equal(got_n, want_n)


class _Step:
    def __init__(self, posr, sid, n):
        self.posr, self.sid, self.n = posr, sid, n


@pytest.mark.parametrize("ends,cap", [([2600, 9000, 300], 16384), ([140000, 5000], 262144), ([131000, 3000], 262144)])
def test_kernel_qsa_multi_gives_qsa_rows_lists(ends, cap):
    """_qsa_multi at the window's bucket == each stream's qsa_rows at its own end (ids, lengths, sparse flags), also
    where the bucket takes _select_tiles and a stream alone would take _select; ties broken the same way."""

    torch.manual_seed(1)
    rows_each, hi, di = 3, 4, 128
    n, rows = len(ends), len(ends) * rows_each
    sc = attn_mod.AttnScratch(rows, 8, 64, cap, DEV)
    pooled = [torch.randn((cap // 4, di), device=DEV).to(torch.bfloat16) for _ in ends]
    for p in pooled:                                   # ties: repeated blocks score the same
        p[100:140] = p[60:100]
    iq = torch.randn((rows, hi, di), device=DEV).to(torch.bfloat16)
    firsts = [e - rows_each for e in ends]
    want_ids, want_nk, want_sp = sc.ids.clone(), sc.nk.clone(), sc.sparse.clone()
    for s, (p, f, e) in enumerate(zip(pooled, firsts, ends)):
        a0 = s * rows_each
        pos = torch.tensor([f], dtype=torch.int32, device=DEV)
        attn_mod.qsa_rows(iq[a0:a0 + rows_each], p, pos, attn_multi._rows_from(sc, a0), rows_each, context=e)
        want_ids[a0:a0 + rows_each] = sc.ids[a0:a0 + rows_each]
        want_nk[a0:a0 + rows_each] = sc.nk[a0:a0 + rows_each]
        want_sp[a0:a0 + rows_each] = sc.sparse[a0:a0 + rows_each]
    sc.ids.zero_()
    sc.nk.zero_()
    sc.sparse.zero_()
    posr = torch.tensor(sum((list(range(f, f + rows_each)) for f in firsts), []), dtype=torch.int32, device=DEV)
    sid = torch.tensor(sum(([s] * rows_each for s in range(n)), []), dtype=torch.int32, device=DEV)
    cp = torch.zeros((attn_multi.PTRS, n), dtype=torch.int64)
    cp[5] = torch.tensor([p.data_ptr() for p in pooled])
    b = type("B", (), {"iq": iq})()
    attn_multi._qsa_multi(b, sc, _Step(posr, sid, n), cp.to(DEV).view(-1), rows, bucket(max(ends)))
    assert torch.equal(sc.nk, want_nk) and torch.equal(sc.sparse, want_sp)
    assert int(want_sp.sum()) > 0
    for r in range(rows):
        if int(want_sp[r]):
            k = int(want_nk[r])
            assert torch.equal(sc.ids[r, :k], want_ids[r, :k]), r


# --- the real checkpoint, cut to a few layers (gputest.sh sets TENSORFOLD_EXL3_FLASHNEXT) ---------------------------

import os  # noqa: E402

MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")


@pytest.fixture(scope="module")
def cut_model():
    if not MODEL or not Path(MODEL).is_dir():
        pytest.skip("set TENSORFOLD_EXL3_FLASHNEXT to an EXL3 Flash Next checkpoint")
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = 4
        c.ple_layers = [i for i in c.ple_layers if i < 4]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = exl3.load(MODEL, "cuda", mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    yield w
    del w
    torch.cuda.empty_cache()


@pytest.mark.parametrize("n", [2, 4])
def test_kernel_exl3_streams_on_round_graphs_equal_each_alone(cut_model, n):
    """The EXL3 pack (n-gram rows staged per buffer, grouped EXL3 experts, the trained-head layout): 2 and 4 streams
    on round graphs, each reply == the serial engine's."""

    w = cut_model
    vocab = w.cfg.vocab
    rng = np.random.default_rng(5)
    prompts = [[int(t) for t in rng.integers(0, vocab, size=k)] for k in (9, 150, 5, 40)][:n]
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None, Sampling(seed=4, top_k=20, top_p=0.95), None][:n]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=False)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 24, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=4, confidence=0.3, prefill_rows=64, graphs=True)
    streams = [Stream(p, 24, s) for p, s in zip(prompts, samplings)]
    _drain(dec, streams)
    assert [s.out for s in streams] == refs
    assert dec.rounds is not None and dec.rounds.replays > 0
