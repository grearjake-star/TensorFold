"""TF_EXL3_PROMPT_SIDE (v8.8): the shared expert's prompt instance (6-bit beside the routed 4-bit) runs on a side stream
beside the routed experts' instance instead of after it. Each pair is computed by the same program of the same instance
and writes only its own rows, so every output is bit-identical with the side stream off:

- a Flash Next-shaped synthetic layer (4-bit routed experts, a 6-bit shared expert as the last expert, one pick a row
  for it) at prompt window sizes: the combined output, the bf16 per-slot output (the engine's prompt call) and the fp32
  per-slot output, with the fused down kernel and with split down partials;
- a CUDA graph captured with the side stream replays to the eager single-stream bits;
- the real model cut to a few layers (TENSORFOLD_EXL3_FLASHNEXT): prompt first token, state and last streams.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

FOUR, SIX = 8, 12                       # K2: half-bits a value


def _trellis(k, n, k2, gen):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, gen):
    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().cuda()


def _layer(E, D, I, seed, cb="mul1"):
    """E routed 4-bit experts and the shared expert (index E) at 6 bits, as the 4.05 pack holds them."""

    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E + 1):
        k2 = SIX if e == E else FOUR
        gate.append((_trellis(D, I, k2, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, k2, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, k2, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _picks(E, R, k, gen):
    sel = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(R)]).to(torch.int32)
    sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)         # the shared expert, every row
    w = torch.cat([torch.rand((R, k), generator=gen) * 0.2 + 0.05, torch.ones((R, 1))], 1)
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


E, D, I, TOPK = 32, 512, 256, 8


@pytest.fixture(scope="module")
def layer():
    from tensorfold.cuda.exl3 import experts

    ex = _layer(E, D, I, seed=7)
    assert ex.widths_gu == ex.widths_d == (1 << FOUR) | (1 << SIX) and ex.main_gu == ex.main_d == 1 << FOUR
    assert experts.launch_widths(ex.widths_gu, ex.main_gu, side=True) >> 32 == 1 << SIX
    return ex


def _bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _both(monkeypatch, fn):
    from tensorfold.cuda.exl3 import experts

    out = []
    for side in (False, True, False, True):              # A B A B: the side stream is created on first use
        monkeypatch.setattr(experts, "PROMPT_SIDE", side)
        out.append(fn())
        torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("R", (65, 256, 300))
@pytest.mark.parametrize("split_down", (False, True), ids=("fused-down", "split-down"))
def test_side_stream_gives_the_same_bits(layer, monkeypatch, R, split_down):
    from tensorfold.cuda.exl3 import experts

    ex = layer
    cfg_d = experts.default_config(I, D, False)
    if split_down:
        cfg_d = (cfg_d[0], cfg_d[1], 2, cfg_d[3])
        assert I % (16 * 2 * cfg_d[1]) == 0
    scratch = experts.Scratch(ex, R, TOPK + 1, cfg_d=cfg_d)
    assert experts._mode(R, True) == "prompt"
    g = torch.Generator().manual_seed(R)
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, R, TOPK, g)
    yb = torch.empty((R * (TOPK + 1), D), dtype=torch.bfloat16, device="cuda")

    combined = _both(monkeypatch, lambda: experts.routed(x, sel, w, ex, scratch, None, R).clone())
    per_slot = _both(monkeypatch, lambda: experts.routed(x, sel, None, ex, scratch, None, R, y_out=yb).clone())
    fp32 = _both(monkeypatch, lambda: experts.routed(x, sel, None, ex, scratch, None, R).clone())
    for runs in (combined, per_slot, fp32):
        assert torch.isfinite(runs[0].float()).all()
        for other in runs[1:]:
            assert torch.equal(_bits(other), _bits(runs[0]))
    # the shared expert's slot is really computed (not skipped): its per-slot rows are non-zero
    assert per_slot[1].view(R, TOPK + 1, D)[:, TOPK].float().abs().sum() > 0


def test_captured_side_stream_replays_the_single_stream_bits(layer, monkeypatch):
    from tensorfold.cuda.exl3 import experts

    ex, R = layer, 192
    scratch = experts.Scratch(ex, R, TOPK + 1)
    g = torch.Generator().manual_seed(3)
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, R, TOPK, g)
    out = torch.empty((R, D), dtype=torch.float32, device="cuda")
    monkeypatch.setattr(experts, "PROMPT_SIDE", True)
    experts.routed(x, sel, w, ex, scratch, out, R)                # warm: the side stream exists before the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.routed(x, sel, w, ex, scratch, out, R)
    monkeypatch.setattr(experts, "PROMPT_SIDE", False)
    for trial in range(2):
        if trial:
            nx = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
            nsel, nw = _picks(E, R, TOPK, g)
            x.copy_(nx), sel.copy_(nsel), w.copy_(nw)
        eager = experts.routed(x, sel, w, ex, scratch, None, R).clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(eager).all() and torch.equal(out, eager), trial


MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
LAYERS = int(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT_LAYERS", "4"))


@pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TENSORFOLD_EXL3_FLASHNEXT to the 4.05 pack")
def test_cut_model_prompt_is_bit_identical_with_the_side_stream(monkeypatch):
    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    monkeypatch.setattr(W.Config, "read", staticmethod(cut))
    w = exl3.load(MODEL, "cuda", mtp=True, draft_vocab="default")      # as test_qwen4_exp_exl3's cut model
    prompt = [int(t) for t in np.random.default_rng(88).integers(0, w.cfg.vocab, size=700)]
    runs = []
    for side in (False, True, False, True):
        monkeypatch.setattr(experts, "PROMPT_SIDE", side)
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=512, graphs=False)
        first = prefill(e, prompt, None)
        torch.cuda.synchronize()
        runs.append((first, e.st.snapshot(), e.last_streams.clone()))
        del e
    for first, snap, tail in runs[1:]:
        assert first == runs[0][0]
        assert torch.equal(tail, runs[0][2])
        for key in ("rec", "conv", "ple_tail"):
            assert torch.equal(snap[key], runs[0][1][key]), key
