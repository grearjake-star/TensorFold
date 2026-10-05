"""D4 on the GPU: with the n-gram (PLE) layer live, decode rounds that ask for their next windows' table pages
(TF_DECODE_AHEAD=1) decode the same tokens and draft the same drafts every round as without (0), for a lone stream
(graph slot) and 2-4 shared streams, on round graphs and eagerly; every row asked for is one a later gather reads, and
every reply == the serial engine's."""

import gc
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _bf16_table, _cfg, _model, _ple, _Rand  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import decode_ahead  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95),
             Sampling(seed=9, top_k=40, top_p=0.9, temperature=0.7)]


class Asked:
    def __init__(self):
        self.ids = []

    def willneed(self, ids):
        self.ids.append(np.asarray(ids).reshape(-1).copy())
        return 0


def _run(w, n, counts, graphs):
    dec = MultiDecoder(w, slots=max(2, n), capacity=1024, depth=4, confidence=0.3, graphs=graphs,
                       round_graphs=graphs)
    if dec.rounds is not None:
        dec.rounds.after = 1
    streams = [Stream(list(p), c, smp) for p, c, smp in zip(PROMPTS[:n], counts, SAMPLINGS)]
    for s in streams:
        dec.admit(s)
    drafts = []
    while dec.live():
        dec.finish(dec.round())
        drafts.append([list(s.drafts) for s in streams])
    out = [s.out for s in streams]
    solo = dec.solo_rounds
    del dec
    gc.collect()
    return out, drafts, solo


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("graphs", [True, False])
def test_decode_ahead_same_tokens_and_drafts(monkeypatch, n, graphs):
    c = _cfg(ple=True)
    with tempfile.TemporaryDirectory() as tmp:
        table = _bf16_table(Path(tmp) / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(3)))
        ple = next(layer.ple for layer in w.layers if layer.ple is not None)
        counts = [26, 18, 31, 22][:n]
        gathered = []
        real = ple.table.gather
        monkeypatch.setattr(ple.table, "gather", lambda ids: gathered.append(np.asarray(ids).reshape(-1).copy())
                            or real(ids), raising=False)
        monkeypatch.setattr(decode_ahead, "ON", False)
        ref_out, ref_drafts, ref_solo = _run(w, n, counts, graphs)
        asked = Asked()
        monkeypatch.setattr(decode_ahead, "ON", True)
        monkeypatch.setattr(decode_ahead, "_ple", lambda w_: SimpleNamespace(ngram=ple.ngram, table=asked))
        monkeypatch.setattr(decode_ahead, "_submit", lambda job: job())
        gathered.clear()
        out, drafts, solo = _run(w, n, counts, graphs)
        assert out == ref_out and drafts == ref_drafts and solo == ref_solo
        if n == 1 and graphs:
            assert solo > 0                          # the lone stream ran in the graph slot
        got = np.concatenate(asked.ids) if asked.ids else np.zeros(0, np.int64)
        assert got.size > 0
        assert set(got.tolist()) <= set(np.concatenate(gathered).tolist())   # every row asked for is read later
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        for p, cnt, smp, toks in zip(PROMPTS, counts, SAMPLINGS, out):
            assert toks == serial_decode(e, prefill(e, list(p), smp), cnt, smp).tokens
