"""The n-gram residency hooks on the synthetic checkpoint (no n-gram tables: they must be a no-op, the tokens the
default's); bad values are refused before the load. Row for row: tests/test_flashnext_ngram_residency.py."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def _tokens(eng, prompt, sampling):
    got: list[int] = []
    eng.generate(prompt, 24, sampling, lambda new: got.extend(new))
    return got


@pytest.mark.parametrize("sampling", [None, Sampling(seed=41, top_k=20, top_p=0.95)])
def test_residency_hooks_change_no_tokens(tmp_path, monkeypatch, sampling):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    from test_flashnext_tp import _checkpoint

    _checkpoint(tmp_path)
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(1, 4096, (700,), generator=g).tolist()
    for key in ("TF_NGRAM_LOCK", "TF_NGRAM_REFRESH", "TF_NGRAM_ADVICE"):
        monkeypatch.delenv(key, raising=False)

    def engine():
        return FlashNextEngine(tmp_path, depth=4, confidence=0.001, draft_vocab=None, max_len=1024, prefetch=False)

    ref = engine()
    want = _tokens(ref, prompt, sampling)
    del ref
    torch.cuda.empty_cache()
    monkeypatch.setenv("TF_NGRAM_LOCK", "auto")
    monkeypatch.setenv("TF_NGRAM_REFRESH", "1")
    eng = engine()
    assert _tokens(eng, prompt, sampling) == want
    assert _tokens(eng, prompt[:300], sampling) == _tokens(eng, prompt[:300], sampling)
    del eng
    for key, bad in (("TF_NGRAM_ADVICE", "sequential"), ("TF_NGRAM_LOCK", "yes"), ("TF_DRAFT_COST", "-0.1")):
        monkeypatch.setenv(key, bad)
        with pytest.raises(ValueError, match=key):
            engine()
        monkeypatch.delenv(key)
