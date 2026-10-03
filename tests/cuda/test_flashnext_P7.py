"""Guards for the prompt indexer (W5-5): ``_scores_rows`` gives ``_scores``' bits, ``_select_cand`` ``_select``'s lists."""

from types import SimpleNamespace

import pytest
import torch
import triton

from tensorfold.families.qwen4_exp.cuda import attention as A

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HI, DI, RATIO = 4, 128, 4


def _inputs(kind, nb, rows):
    g = torch.Generator(device="cuda").manual_seed(7)
    if kind == "ties":      # few distinct small integers: many exactly equal block scores
        pooled = torch.zeros(nb, DI, device="cuda")
        pooled[:, :2] = torch.randint(-1, 3, (nb, 2), device="cuda", generator=g).float()
        iq = torch.zeros(rows, HI, DI, device="cuda")
        iq[:, :, :2] = torch.randint(0, 3, (rows, HI, 2), device="cuda", generator=g).float()
        return pooled.to(torch.bfloat16), iq.to(torch.bfloat16)
    pooled = torch.randn(nb, DI, device="cuda", generator=g)
    iq = torch.randn(rows, HI, DI, device="cuda", generator=g)
    if kind == "wide":      # exponents 2^-8..2^8: any other summation order shows in the bits
        pooled *= torch.exp2(torch.randint(-8, 9, (nb, DI), device="cuda", generator=g).float())
        iq *= torch.exp2(torch.randint(-8, 9, (rows, HI, DI), device="cuda", generator=g).float())
    return pooled.to(torch.bfloat16), iq.to(torch.bfloat16)


def _stock(iq, pooled, pos, sc, rows, blocks):
    A._scores[(rows, triton.cdiv(blocks, 64))](iq, pooled, pos, sc.scores, sc.nb, HI=HI, DI=DI, RATIO=RATIO,
                                                TOP=sc.budget // RATIO, BB=64, num_warps=4)
    A._launch_select(sc, pos, rows, blocks)
    torch.cuda.synchronize()
    return sc.scores.clone(), sc.ids.clone(), sc.nk.clone(), sc.sparse.clone()


@pytest.mark.parametrize("kind", ["randn", "wide", "ties"])
@pytest.mark.parametrize("cap,p0,rows", [(65536, 1500, 256), (65536, 30000, 256), (65536, 65536 - 200, 200),
                                         (65536, 9000, 100), (262144, 200000, 256)])
@pytest.mark.parametrize("small_cap", [False, True])
def test_prompt_indexer_matches_stock(kind, cap, p0, rows, small_cap, monkeypatch):
    if small_cap:           # most rows overflow the candidate list: the radix fallback must list the same blocks
        monkeypatch.setattr(A, "PROMPT_CAP", 512)
    nb = cap // RATIO
    pooled, iq = _inputs(kind, nb, rows)
    pos = torch.tensor([p0], dtype=torch.int32, device="cuda")
    blocks = triton.cdiv(p0 + rows, RATIO)
    sc = A.AttnScratch(256, 24, 256, cap, "cuda")
    assert sc.smx is not None
    scores, ids, nk, sparse = _stock(iq, pooled, pos, sc, rows, blocks)
    sc.scores.fill_(float("nan")); sc.ids.fill_(-1); sc.nk.zero_(); sc.sparse.fill_(7)
    monkeypatch.setattr(A, "PROMPT_INDEXER", True)
    monkeypatch.setattr(A, "PROMPT_MIN_BLOCKS", 0)       # the new path at every position, short ones too
    A.qsa_rows(iq, pooled, pos, sc, rows, context=p0 + rows)
    torch.cuda.synchronize()
    sparse_rows = sparse[:rows].bool()
    for r in range(rows):
        if sparse_rows[r]:
            done = (p0 + r + 1) // RATIO
            assert torch.equal(sc.scores[r, :done].view(torch.int32), scores[r, :done].view(torch.int32)), r
    assert torch.equal(sc.nk[:rows], nk[:rows]) and torch.equal(sc.sparse[:rows], sparse[:rows])
    for r in range(rows):                   # a dense row reads its keys in order: only sparse rows have lists
        if sparse_rows[r]:
            assert torch.equal(sc.ids[r, :nk[r]], ids[r, :nk[r]]), r


def test_decode_scratch_keeps_the_stock_indexer():
    sc = A.AttnScratch(11, 24, 256, 65536, "cuda")
    assert sc.smx is None and sc.cand is None


def test_short_contexts_keep_the_stock_indexer(monkeypatch):
    """Below PROMPT_MIN_BLOCKS blocks the stock kernels run (as fast there); past it the prompt kernels."""
    calls = []
    monkeypatch.setattr(A, "PROMPT_INDEXER", True)
    monkeypatch.setattr(A, "_prompt_ext", lambda: SimpleNamespace(scores=lambda *a: calls.append("new")))
    monkeypatch.setattr(A, "_launch_select_cand", lambda *a: calls.append("cand"))
    monkeypatch.setattr(A, "_launch_select", lambda *a: calls.append("stock"))
    sc = A.AttnScratch(256, 24, 256, 65536, "cuda")
    iq = torch.zeros(256, HI, DI, dtype=torch.bfloat16, device="cuda")
    pooled = torch.zeros(sc.nb, DI, dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    A.qsa_rows(iq, pooled, pos, sc, 256, context=4 * A.PROMPT_MIN_BLOCKS - 4)
    A.qsa_rows(iq, pooled, pos, sc, 256, context=4 * A.PROMPT_MIN_BLOCKS)
    assert calls == ["stock", "new", "cand"]
