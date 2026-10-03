"""K5 (EXL3 4.05 decode speed): guards for each change.

TF_EXL3_HC_FP8 (opt-in, numerics-changing): off by default; the MXFP8 copy of an fp16 hyper-connection face tracks
the rows it came from, keeps a row's bits whatever its window (drafted == serial), and serves both the read-out's
down (``partials``) and up (``__call__``) calls."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _face(n, k, seed):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((n, k), generator=g) * 0.02).half().cuda()
    sc = exl3_mm.Scratch(11)
    face = exl3_mm.f16(sc, [w], "cuda")
    return w, face


def test_hc_fp8_is_off_by_default(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    monkeypatch.delenv("TF_EXL3_HC_FP8", raising=False)
    assert not exl3_mm.hc_fp8()
    monkeypatch.setenv("TF_EXL3_HC_FP8", "yes")
    assert not exl3_mm.hc_fp8()
    monkeypatch.setenv("TF_EXL3_HC_FP8", "1")
    assert exl3_mm.hc_fp8()


@pytest.mark.parametrize("n,k", [(324, 10240), (10240, 320)])
def test_the_fp8_copy_tracks_its_rows_and_keeps_rows_apart(n, k, tol=0.05, shrink=0.65):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    w, face = _face(n, k, n)
    f8 = exl3_mm.F8(face)
    x = (torch.randn((7, k)) * 0.5).to(torch.bfloat16).cuda()
    ref = x.float() @ w.float().t()
    got = f8.partials(x)
    assert got.shape == (7, n) and got.dtype == torch.bfloat16
    err = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, err                           # e4m3: a few percent
    for r in range(7):                              # a row alone gives its bits in the window
        assert torch.equal(f8.partials(x[r:r + 1].contiguous()).view(torch.int16), got[r:r + 1].view(torch.int16))
    out = torch.empty((7, n), dtype=torch.bfloat16, device="cuda")
    f8(x, out)
    assert torch.equal(out.view(torch.int16), got.view(torch.int16))
    assert f8.nbytes() < face.nbytes() * shrink     # ~1/2 of the fp16 bytes, padding included


def test_the_exl3_ngram_table_takes_the_residency_hooks(tmp_path):
    """EXL3's n-gram table reports, locks and refreshes its pages like HostTable (TF_NGRAM_LOCK / TF_NGRAM_REFRESH
    and the start-up refresh); none of it changes a gathered byte."""

    import numpy as np

    from tensorfold.families.qwen4_exp import host_residency
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable

    t = object.__new__(NgramTable)
    path = tmp_path / "rows.bin"
    rows = np.arange(4096 * 61, dtype=np.int16).reshape(4096, 61)
    rows.tofile(path)
    t.words = [np.memmap(path, dtype=np.int16, mode="r", shape=rows.shape)]
    assert isinstance(t, host_residency._Residency) and t.DENSE == ("words",)
    have, total = t.residency()["words"]
    assert total >= rows.nbytes
    assert t.refresh(0) >= 0
    t.PART_BYTES = 61 * 2 * 1000                    # ~1000-row runs: auto can lock part of one big array
    runs = t.parts()["words"]
    assert len(runs) == 5 and sum(r.shape[0] for r in runs) == 4096
    got = t.lock_parts(("words",), budget=2 * 61 * 2 * 1000 + 100)
    assert 0 < got <= 3 * 4096 * 2 * 61
    t.unlock()
    assert np.array_equal(np.asarray(t.words[0]), rows)
