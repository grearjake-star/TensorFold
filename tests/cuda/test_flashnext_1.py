"""A trained MTP head loaded from its own file (``TF_MTP_HEAD``, DECODE-PLAN item #1): it replaces only the
checkpoint's MTP tensors of the same names, never touches the main model or the model directory, refuses
anything that is not an MTP tensor, and drafting with it still emits serial decoding's tokens (greedy and
sampled): a different draft head changes speed only."""

import hashlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import load_file, save_file  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

from test_flashnext_tp import PROMPT, _checkpoint  # noqa: E402


def _head_file(root, path, extra=None):
    """A re-randomized MTP head in the checkpoint's format (new 4-bit codes on the stored grids, scaled norms,
    a different router): a head whose drafts differ from the stock head's."""

    t = load_file(str(root / "model.safetensors"))
    g = torch.Generator().manual_seed(123)
    out = {}
    for name in ("mtp.fc_hidden", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.mlp.shared_expert.up_proj"):
        w = t[name + ".weight"]
        out[name + ".weight"] = torch.randint(-(2**31), 2**31 - 1, w.shape, generator=g, dtype=torch.int64).to(torch.int32)
        out[name + ".scales"] = t[name + ".scales"]
        out[name + ".biases"] = t[name + ".biases"]
    out["mtp.pre_fc_norm_hidden.weight"] = (t["mtp.pre_fc_norm_hidden.weight"].float() * 1.1)
    out["mtp.layers.0.mlp.gate.weight"] = (torch.randn(t["mtp.layers.0.mlp.gate.weight"].shape, generator=g) * 0.05).to(torch.bfloat16)
    out.update(extra or {})
    save_file(out, str(path))
    return out


def _digest(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}


def test_the_head_file_replaces_mtp_tensors_only(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    before = _digest(model)
    head = tmp_path / "head.safetensors"
    _head_file(model, head)
    stock = load(model, mtp=True)
    trained = load(model, mtp=True, mtp_head=head)
    assert trained.meta["mtp_head"] == str(head) and "mtp_head" not in stock.meta
    # the main model: identical
    for a, b in zip(qmm.to_mlx(stock.head), qmm.to_mlx(trained.head)):
        assert torch.equal(a, b)
    for la, lb in zip(stock.layers, trained.layers):
        assert torch.equal(la.moe.router, lb.moe.router)
        assert torch.equal(la.attn_hc.down.weight, lb.attn_hc.down.weight)
    # the head: the file's tensors
    assert not torch.equal(stock.mtp.fc_h.weight, trained.mtp.fc_h.weight)
    assert torch.equal(stock.mtp.fc_e.weight, trained.mtp.fc_e.weight)
    assert torch.allclose(trained.mtp.norm_h, stock.mtp.norm_h * 1.1, rtol=1e-6)
    assert not torch.equal(stock.mtp.layer.moe.router[:-1], trained.mtp.layer.moe.router[:-1])
    assert torch.equal(stock.mtp.layer.moe.router[-1], trained.mtp.layer.moe.router[-1])    # shared gate row
    assert _digest(model) == before                                       # the model directory is never written
    # the environment hook is the default
    import os

    os.environ["TF_MTP_HEAD"] = str(head)
    try:
        assert load(model, mtp=True).meta["mtp_head"] == str(head)
        assert "mtp_head" not in load(model, mtp=False).meta                # no MTP head, nothing to replace
    finally:
        del os.environ["TF_MTP_HEAD"]


def test_the_head_file_refuses_other_tensors(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    bad = tmp_path / "bad.safetensors"
    t = load_file(str(model / "model.safetensors"))
    _head_file(model, bad, {"lm_head.scales": t["lm_head.scales"]})
    with pytest.raises(ValueError, match="only mtp"):
        load(model, mtp=True, mtp_head=bad)
    unknown = tmp_path / "unknown.safetensors"
    _head_file(model, unknown, {"mtp.layers.0.not_a_tensor.weight": torch.zeros(4)})
    with pytest.raises(ValueError):
        load(model, mtp=True, mtp_head=unknown)
    # a head in another format (e.g. an MLX head on a checkpoint whose MTP is bf16) is refused by dtype and shape
    wrong = tmp_path / "wrong.safetensors"
    _head_file(model, wrong, {"mtp.fc_hidden.scales": t["mtp.fc_hidden.weight"].view(torch.int32)[:, :t["mtp.fc_hidden.scales"].shape[1]].contiguous()})
    with pytest.raises(ValueError, match="checkpoint's is"):
        load(model, mtp=True, mtp_head=wrong)
    short = tmp_path / "short.safetensors"
    _head_file(model, short, {"mtp.fc_hidden.biases": t["mtp.fc_hidden.biases"][:1]})
    with pytest.raises(ValueError, match="checkpoint's is"):
        load(model, mtp=True, mtp_head=short)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=17, top_k=20, top_p=0.95)])
def test_drafts_with_a_trained_head_give_serial_tokens(tmp_path, sampling):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    head = tmp_path / "head.safetensors"
    _head_file(model, head)
    import os

    os.environ["TF_MTP_HEAD"] = str(head)
    try:
        eng = FlashNextEngine(model, depth=5, confidence=0.0, draft_vocab=None, max_len=512, prefetch=False)
    finally:
        del os.environ["TF_MTP_HEAD"]
    assert eng.w.meta["mtp_head"] == str(head)
    stock = FlashNextEngine(model, depth=5, confidence=0.0, draft_vocab=None, max_len=512, prefetch=False)
    first = prefill(eng.e, PROMPT, sampling)
    ref = serial_decode(eng.e, first, 40, sampling).tokens
    prefill(stock.e, PROMPT, sampling)
    assert serial_decode(stock.e, first, 40, sampling).tokens == ref          # the main model is the same
    for depth in (1, 3, 5):
        prefill(eng.e, PROMPT, sampling)
        got = mtp_decode(eng.e, first, 40, sampling, depth=depth, confidence=0.0)
        assert got.tokens == ref, depth
        prefill(stock.e, PROMPT, sampling)
        base = mtp_decode(stock.e, first, 40, sampling, depth=depth, confidence=0.0)
        assert base.tokens == ref, depth
    # the heads really differ: their draft chains are not the same
    prefill(eng.e, PROMPT, sampling)
    a = eng.e.mtp_forward([first], eng.e.last_streams).float().clone()
    prefill(stock.e, PROMPT, sampling)
    b = stock.e.mtp_forward([first], stock.e.last_streams).float().clone()
    assert not torch.equal(a, b)
