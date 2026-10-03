"""TENSORFOLD_PREFILL_ROWS on an EXL3 pack: unset keeps 2048-row staging and routed windows; set, it sizes both
(the routed window at most 4096 rows). Rows never change bits, so this sizes memory and launches only."""

import importlib

import pytest

pytest.importorskip("torch")


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", str(value))
    monkeypatch.delenv("TF_EXL3_MOE_WINDOW", raising=False)
    from tensorfold.families.qwen4_exp.cuda import exl3, exl3_pack

    return importlib.reload(exl3_pack), importlib.reload(exl3)


@pytest.mark.parametrize("rows,staging,window", [(None, 2048, 2048), (1024, 2048, 2048), (4096, 4096, 4096),
                                                  (8192, 8192, 4096)])
def test_exl3_sizes_follow_the_prompt_rows(monkeypatch, rows, staging, window):
    from tensorfold.cuda.exl3 import experts

    pack, x3 = _reload(monkeypatch, rows)
    try:
        assert x3.PREFILL_ROWS == staging
        if experts.PROMPT == "prompt":
            assert pack.MOE_WINDOW == window
    finally:
        _reload(monkeypatch, None)
