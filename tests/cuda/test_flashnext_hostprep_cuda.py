"""S1-HOST on the GPU: shared rounds whose tables come from hostprep (TF_MULTI_HOST_AHEAD=1: numpy pointers,
double-buffered pinned staging) decode the same tokens and draft the same drafts every round as the list-based tables
(TF_MULTI_HOST_AHEAD=0), on round graphs and eagerly, at 2, 4 and 8 streams; every reply == the serial engine's."""

import gc

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import hostprep  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95),
             Sampling(seed=9, top_k=40, top_p=0.9, temperature=0.7)]


def _prompts(n):
    g = torch.Generator().manual_seed(40 + n)
    return [torch.randint(1, V, (3 + 2 * i,), generator=g).tolist() for i in range(n)]


def _run(w, prompts, counts, samplings, graphs, depth):
    dec = MultiDecoder(w, slots=len(prompts), capacity=1024, depth=depth, confidence=0.3, round_graphs=graphs)
    if dec.rounds is not None:
        dec.rounds.after = 1
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(prompts, counts, samplings)]
    for s in streams:
        dec.admit(s)
    drafts = []
    while dec.live():
        dec.finish(dec.round())
        drafts.append([list(s.drafts) for s in streams])
    replays = dec.rounds.replays if dec.rounds is not None else 0
    out = [s.out for s in streams]
    del dec
    gc.collect()
    return out, drafts, replays


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("graphs", [True, False])
def test_hostprep_tables_decode_and_draft_the_same(monkeypatch, n, graphs):
    w = _model()
    prompts = _prompts(n)
    counts = [24 + 3 * i for i in range(n)]
    samplings = [SAMPLINGS[i % len(SAMPLINGS)] for i in range(n)]
    monkeypatch.setattr(hostprep, "ON", False)
    ref_out, ref_drafts, ref_replays = _run(w, prompts, counts, samplings, graphs, 3)
    monkeypatch.setattr(hostprep, "ON", True)
    out, drafts, replays = _run(w, prompts, counts, samplings, graphs, 3)
    assert out == ref_out and drafts == ref_drafts
    if graphs:
        assert replays > 0 and replays == ref_replays
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    for p, c, smp, got in zip(prompts, counts, samplings, out):
        assert got == serial_decode(e, prefill(e, list(p), smp), c, smp).tokens


def test_pinned_ring_reuses_a_buffer_only_after_its_copy_ran():
    """Tables put while the stream is busy (a long kernel queued first) land intact, though both buffers of a site
    are rewritten many times before the stream reaches the copies."""

    outs = [torch.full((300,), -1, dtype=torch.int64, device="cuda") for _ in range(12)]
    torch.cuda._sleep(50_000_000)                      # ~tens of ms of queued GPU work
    want = []
    for i, out in enumerate(outs):
        vals = np.arange(300, dtype=np.int64) * (i + 1)
        hostprep.put(vals, torch.int64, "cuda", out, "ring_test")
        want.append(vals)
    eager = hostprep.put(np.arange(5, dtype=np.int64), torch.int32, "cuda", None, "ring_test_new")
    torch.cuda.synchronize()
    for out, vals in zip(outs, want):
        assert out.cpu().numpy().tolist() == vals.tolist()
    assert eager.dtype == torch.int32 and eager.cpu().tolist() == [0, 1, 2, 3, 4]
