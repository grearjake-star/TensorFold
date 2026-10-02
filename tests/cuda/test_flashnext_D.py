"""Trained draft heads (TF_MTP_HEAD) on an EXL3 pack, the cost rule's timing table, and the cost stop under
--parallel: drafts change speed only, so drafted == serial and each stream == its solo run."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")


def test_only_a_head_with_4bit_triples_is_mlx_format() -> None:
    from tensorfold.families.qwen4_exp.cuda.weights import is_mlx_head

    w = torch.zeros(4, 32, dtype=torch.bfloat16)
    assert is_mlx_head({"mtp.fc_hidden.weight": torch.zeros(4, 4, dtype=torch.int32),
                        "mtp.fc_hidden.scales": w, "mtp.fc_hidden.biases": w})
    assert not is_mlx_head({"mtp.fc_hidden.weight": w, "mtp.pre_fc_norm_hidden.weight": torch.zeros(8)})


# -- a trained head on an EXL3 pack (bf16 head format; trellis linears replaced by fp16 matrices) ---------------------
EXL3 = __import__("os").environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
needs_exl3 = pytest.mark.skipif(not EXL3 or not Path(EXL3).is_dir(),
                                reason="set TENSORFOLD_EXL3_FLASHNEXT to an EXL3 Flash Next checkpoint")


def _exl3_head(pk, seed: int = 5) -> dict:
    """A re-randomized bf16 head for the pack: its replaceable linears, an HC matrix, the router, two norms."""

    from tensorfold.families.qwen4_exp.cuda.exl3 import HEAD_LINEARS

    g = torch.Generator().manual_seed(seed)
    head = {}
    for stem in HEAD_LINEARS:
        k = pk.entry(stem + ".suh")[4][0]
        n = pk.entry(stem + ".svh")[4][0]
        head[stem + ".weight"] = (torch.randn((n, k), generator=g) * k ** -0.5).to(torch.bfloat16)
    for name in ("mtp.layers.0.mlp.gate.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight"):
        t = pk.get(name)
        head[name] = (t.float() + 0.01 * torch.randn(t.shape, generator=g)).to(torch.bfloat16)
    for name in ("mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.self_attn.q_norm.weight"):
        head[name] = (pk.get(name).float() + 0.0625).to(torch.bfloat16)          # the pack stores gamma - 1
    return head


@needs_exl3
def test_an_exl3_pack_checks_a_bf16_head(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from tensorfold.families.qwen4_exp.cuda.exl3 import exl3_head
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    pk = Pack(Path(EXL3))
    head = _exl3_head(pk)
    save_file(head, str(tmp_path / "ok.safetensors"))
    assert set(exl3_head(pk, tmp_path / "ok.safetensors")) == set(head)
    bad = {
        "shape": {"mtp.fc_hidden.weight": torch.zeros((2560, 2048), dtype=torch.bfloat16)},
        "expert": {"mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros((640, 2560), dtype=torch.bfloat16)},
        "mlx": {"mtp.fc_hidden.weight": torch.zeros((2560, 320), dtype=torch.int32),
                "mtp.fc_hidden.scales": torch.zeros((2560, 80), dtype=torch.bfloat16),
                "mtp.fc_hidden.biases": torch.zeros((2560, 80), dtype=torch.bfloat16)},
        "main": {"model.language_model.norm.weight": torch.zeros((2560,), dtype=torch.bfloat16)},
    }
    for label, tensors in bad.items():
        save_file(tensors, str(tmp_path / f"{label}.safetensors"))
        with pytest.raises(ValueError):
            exl3_head(pk, tmp_path / f"{label}.safetensors")


@cuda
@needs_exl3
def test_a_bf16_head_on_an_exl3_cut_model_drafts_the_serial_tokens(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    head = _exl3_head(Pack(Path(EXL3)))
    path = tmp_path / "head.safetensors"
    save_file(head, str(path))
    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = 4
        c.ple_layers = [i for i in c.ple_layers if i < 4]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = exl3.load(EXL3, "cuda", mtp=True, draft_vocab="default", mtp_head=path)
    finally:
        W.Config.read = real
    assert w.meta["mtp_head"] == str(path)
    assert torch.equal(w.mtp.fc_h.w, head["mtp.fc_hidden.weight"].to(torch.float16).cuda())
    assert torch.equal(w.mtp.norm_h, (head["mtp.pre_fc_norm_hidden.weight"].float() + 1.0).cuda())
    prompt = [9707, 11, 1246, 525, 498, 30, 3555, 374, 279, 6722, 315, 9625, 30, 220]
    for sampling in (None, Sampling(seed=3, temperature=0.7, top_k=20, top_p=0.8)):
        e = Engine(w, capacity=512, max_rows=8, graphs=False)
        s = serial_decode(e, prefill(e, prompt, sampling, mtp=False), 32, sampling)
        for depth in (1, 4, 7):
            d = mtp_decode(e, prefill(e, prompt, sampling, mtp=True), 32, sampling, depth=depth, confidence=0.0)
            assert d.tokens == s.tokens, depth


def test_the_cost_rule_reads_a_checkpoints_own_timing_table() -> None:
    from tensorfold.families.qwen4_exp.cuda.decode import DRAFT_MS, VERIFY_MS, cost_bars, timing_table

    assert timing_table({}) == (VERIFY_MS, DRAFT_MS)
    assert cost_bars(0.06, 4) == cost_bars(0.06, 4, timing_table({}))               # unset: exactly as before
    t = timing_table({"TF_VERIFY_MS": "29.5,33.5,37.8", "TF_DRAFT_MS": "1.3"})
    assert t == ((29.5, 33.5, 37.8), 1.3)
    bars = cost_bars(0.1, 3, t)                          # past the table: its last step again
    assert [round(b, 6) for b in bars] == [round(0.1 * (x + 1.3), 6) for x in (4.0, 4.3, 4.3, 4.3)]
    for bad in ({"TF_VERIFY_MS": "30"}, {"TF_VERIFY_MS": "30,20"}, {"TF_DRAFT_MS": "-1"}):
        with pytest.raises(ValueError):
            timing_table(bad)


@cuda
@pytest.mark.parametrize("cost", [0.03, 0.3])
def test_concurrent_streams_with_the_cost_stop_equal_each_alone(cost: float) -> None:
    """M3: --parallel's drafter now applies the expected-time stop (cost bars, as one stream's); drafts change speed
    only, so streams decoded together still emit what each does alone."""

    from test_flashnext_forward import _model

    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode, timing_table
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = _model()
    prompts = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13]]
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, temperature=0.7, top_k=20, top_p=0.8)]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 20, sampling).tokens)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=4, confidence=0.3, cost=cost,
                       timing=timing_table({"TF_VERIFY_MS": "29.5,33.5,37.8,41.8,47.9", "TF_DRAFT_MS": "1.3"}))
    streams = []
    for prompt, sampling in zip(prompts, samplings):
        got: list[int] = []
        s = Stream(prompt, 20, sampling, draft=True, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for i, (s, got) in enumerate(streams):
        assert got == refs[i], i
