"""S1-ROUTE (TF_MOE_ROUTE_FUSED): a decode window's top-k + weights (tensorfold.cuda.moe.select_rows, Triton) and its
grouping (group_kernel) in one block (select_group; rot_in keeps its own launch: "two"), or all three in one grid
(route_kernel: "single"). Every output must be bit-identical with the three launches:

- the kernel alone on many random windows (1..64 rows, Flash Next's 512 experts + the shared gate logit, top 10;
  logits with forced ties, -inf and very large spreads; bf16 and fp16 rows): picks, weights, the groups (count, ids,
  members) and the rotated rows xg/xu, eager and replayed from a CUDA graph with new inputs;
- a small synthetic EXL3 layer end to end (routed experts' per-slot outputs, fused vs not);
- TF_ROUTER_BE (an experiment): router logits with 16 experts a program equal the default 32's;
- the real model cut to a few layers (TENSORFOLD_EXL3_FLASHNEXT): verify-shaped forward windows (logits and the last
  layer's picks/weights) on/off, and MTP drafts (eager and graphs) emit the serial tokens with the fused route.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

NE, TOPK, D = 512, 10, 2560               # Flash Next: 512 routed experts + the shared expert (id 512), top 10
SLOTS = TOPK + 1


def _ext():
    from tensorfold.cuda.exl3 import experts

    return experts._ext()


def _logits(R, kind, g):
    if kind == "normal":
        L = torch.randn((R, NE + 1), generator=g) * 2.0
    elif kind == "ties":                      # few distinct values: many equal maxima (the lower id must win)
        L = torch.randint(-3, 4, (R, NE + 1), generator=g).float() * 0.5
    elif kind == "flat":                      # every routed logit equal
        L = torch.zeros((R, NE + 1))
        L[:, NE] = torch.randn((R,), generator=g)
    elif kind == "spread":                    # exp underflows for most picks; -inf and +-0 sprinkled in
        L = torch.randn((R, NE + 1), generator=g) * 60.0
        L[:, 5] = float("-inf")
        L[:, 7] = -0.0
        L[:, 9] = 0.0
    elif kind == "few":                       # fewer than TOPK finite logits in some rows
        L = torch.full((R, NE + 1), float("-inf"))
        L[:, :4] = torch.randn((R, 4), generator=g)
        L[:, NE] = torch.randn((R,), generator=g)
    else:
        raise ValueError(kind)
    return L.float().cuda().contiguous()


def _fake_layer(E, g):
    sign = lambda: torch.randint(0, 2, (E, D), generator=g).float() * 2 - 1      # noqa: E731
    suh_g = (sign() * (torch.rand((E, D), generator=g) + 0.5) / math.sqrt(D)).half().cuda()
    suh_u = (sign() * (torch.rand((E, D), generator=g) + 0.5) / math.sqrt(D)).half().cuda()
    return SimpleNamespace(suh_g=suh_g, suh_u=suh_u, count=E, dims=D)


class _Scr:
    def __init__(self, R, E):
        P = R * SLOTS
        maxu = min(P, E)
        self.xg = torch.zeros((P, D), dtype=torch.float16, device="cuda")
        self.xu = torch.zeros((P, D), dtype=torch.float16, device="cuda")
        self.ids = torch.full((maxu,), -7, dtype=torch.int32, device="cuda")
        self.count = torch.zeros((1,), dtype=torch.int32, device="cuda")
        self.members = torch.full((maxu, R), -9, dtype=torch.int32, device="cuda")
        self.pick = torch.full((R, SLOTS), -5, dtype=torch.int32, device="cuda")
        self.wts = torch.full((R, SLOTS), -5.0, dtype=torch.float32, device="cuda")
        self.done = torch.zeros((1,), dtype=torch.int32, device="cuda")


def _old(L, x, ex, s, R):
    from tensorfold.cuda import moe

    moe.select_rows(L, SimpleNamespace(pick=s.pick, wts=s.wts), TOPK, NE)
    _ext().group(s.pick, s.ids, s.count, s.members, R, SLOTS, ex.count)
    _ext().rot_in(x, x.stride(0), s.pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, SLOTS, ex.count, False)


def _new(L, x, ex, s, R, kind="two"):
    if kind == "single":
        _ext().route(L, s.pick, s.wts, x, x.stride(0), ex.suh_g, ex.suh_u, s.xg, s.xu, s.ids, s.count, s.members,
                     s.done, R, D, SLOTS, NE, TOPK, ex.count)
        return
    _ext().select_group(L, s.pick, s.wts, s.ids, s.count, s.members, R, SLOTS, NE, TOPK, ex.count)
    _ext().rot_in(x, x.stride(0), s.pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, SLOTS, ex.count, False)


def _same(a, b, R, what):
    torch.cuda.synchronize()
    assert torch.equal(a.pick, b.pick), (what, "pick")
    assert torch.equal(a.wts.view(torch.int32), b.wts.view(torch.int32)), (what, "wts")
    assert torch.equal(a.count, b.count), (what, "count")
    n = int(a.count)
    assert torch.equal(a.ids[:n], b.ids[:n]), (what, "ids")
    assert torch.equal(a.members[:n], b.members[:n]), (what, "members")
    assert torch.equal(a.xg.view(torch.int16), b.xg.view(torch.int16)), (what, "xg")
    assert torch.equal(a.xu.view(torch.int16), b.xu.view(torch.int16)), (what, "xu")
    assert int(b.done) == 0, (what, "counter not reset")


@pytest.mark.parametrize("kind", ("two", "single"))
@pytest.mark.parametrize("E", (NE + 1, NE), ids=("shared-in-table", "routed-only"))
def test_route_kernel_matches_the_three_launches(E, kind):
    route = kind
    g = torch.Generator().manual_seed(17 + E)
    ex = _fake_layer(E, g)
    checked = 0
    for R in (1, 2, 3, 4, 5, 6, 8, 11, 16, 23, 33, 48, 64):
        for kind in ("normal", "ties", "flat", "spread", "few"):
            for dt in (torch.bfloat16, torch.float16):
                for trial in range(3 if kind == "normal" else 1):
                    L = _logits(R, kind, g)
                    big = torch.randn((R, D + 256), generator=g).to(dt).cuda()     # strided rows, as b.mixed[:R]
                    x = big[:, :D]
                    a, b = _Scr(R, E), _Scr(R, E)
                    _old(L, x, ex, a, R)
                    _new(L, x, ex, b, R, route)
                    _same(a, b, R, (R, kind, dt, trial, route))
                    checked += R
    assert checked > 1500


@pytest.mark.parametrize("route", ("two", "single"))
def test_route_kernel_replays_from_a_graph_with_new_inputs(route):
    g = torch.Generator().manual_seed(5)
    ex = _fake_layer(NE + 1, g)
    for R in (3, 11, 40):
        L = _logits(R, "normal", g)
        x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
        s = _Scr(R, NE + 1)
        _new(L, x, ex, s, R, route)                           # warm
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):                                # several launches a graph: the counter resets each
                _new(L, x, ex, s, R, route)
        for trial in range(4):
            L.copy_(_logits(R, ("normal", "ties", "spread", "few")[trial], g))
            x.copy_(torch.randn((R, D), generator=g).to(torch.bfloat16).cuda())
            ref = _Scr(R, NE + 1)
            _old(L, x, ex, ref, R)
            graph.replay()
            _same(ref, s, R, (R, trial))


def test_router_block_e_16_gives_the_default_bits(monkeypatch):
    from tensorfold.cuda import moe

    g = torch.Generator().manual_seed(9)
    rows = (torch.randn((NE + 1, D), generator=g) * 0.02).to(torch.bfloat16).cuda()
    for R in (1, 2, 3, 5, 8, 11, 16):
        for trial in range(4):
            big = torch.randn((R, D + 128), generator=g).to(torch.bfloat16).cuda()
            x = big[:, :D]
            outs = []
            for be in (32, 16):
                monkeypatch.setattr(moe, "ROUTER_BE", be)
                outs.append(moe.router(x, rows).clone())
            assert torch.equal(outs[0].view(torch.int32), outs[1].view(torch.int32)), (R, trial)


# --- a small synthetic EXL3 layer end to end (the routed experts' per-slot outputs) -----------------------------------

def test_synthetic_layer_routed_outputs_fused_equal_unfused(monkeypatch):
    from test_exl3_prompt_side import _layer

    from tensorfold.cuda import moe
    from tensorfold.cuda.exl3 import experts

    E_, D_, I_, K_ = 40, 512, 256, 6
    ex = _layer(E_, D_, I_, seed=11)                       # E_ routed 4-bit + the shared 6-bit (id E_)
    g = torch.Generator().manual_seed(2)
    rows = (torch.randn((E_ + 1, D_), generator=g) * 0.05).to(torch.bfloat16).cuda()
    for R in (1, 3, 7, 12, 33, 64):
        cfg = SimpleNamespace(num_experts_per_tok=K_, num_experts=E_, moe_intermediate_size=I_, hidden_size=D_)
        buf = moe.MoEBuffers(R, cfg, "cuda")
        s = experts.Scratch(ex, R, K_ + 1)
        x = torch.randn((R, D_), generator=g).to(torch.bfloat16).cuda()
        moe.router(x, rows, buf.logits[:R])
        assert experts.route_ok(R, x, ex, s, K_, E_, fused=True)
        moe.select_rows(buf.logits[:R], buf, K_, E_)
        ref_pick, ref_wts = buf.pick[:R].clone(), buf.wts[:R].clone()
        ref = experts.routed(x, buf.pick[:R], None, ex, s, None, R).clone()
        buf.pick.fill_(-1), buf.wts.fill_(0)
        experts.route(buf.logits[:R], x, buf.pick[:R], buf.wts[:R], ex, s, R, K_, E_)
        got = experts.routed(x, buf.pick[:R], None, ex, s, None, R, prepped=True).clone()
        torch.cuda.synchronize()
        assert torch.equal(buf.pick[:R], ref_pick) and torch.equal(buf.wts[:R], ref_wts), R
        assert torch.isfinite(ref).all() and torch.equal(got.view(torch.int32), ref.view(torch.int32)), R


# --- the real checkpoint, cut to a few layers ---------------------------------------------------------------------

MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
LAYERS = int(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set TENSORFOLD_EXL3_FLASHNEXT to the 4.05 pack")


@pytest.fixture(scope="module")
def cut_model():
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = exl3.load(MODEL, "cuda", mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    yield w
    del w
    torch.cuda.empty_cache()


@needs_model
def test_cut_model_verify_windows_are_bit_identical(cut_model, monkeypatch):
    """Verify-shaped windows (1..64 rows) after a real prompt: logits and the last layer's picks/weights, on vs off."""

    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    w = cut_model
    toks = [int(t) for t in np.random.default_rng(21).integers(0, w.cfg.vocab, size=400)]
    used = []
    real_route = experts.route

    def spy(*a, **k):
        used.append(a[6])
        return real_route(*a, **k)

    monkeypatch.setattr(experts, "route", spy)
    runs = {}
    for mode in ("off", "two", "off", "two", "single"):
        fused = mode != "off"
        monkeypatch.setattr(experts, "ROUTE_FUSED", fused)
        monkeypatch.setattr(experts, "ROUTE_SINGLE", mode == "single")
        e = Engine(w, capacity=1024, max_rows=64, prefill_rows=128, graphs=False)
        e.reset()
        out, at = [], 0
        for rows in (1, 2, 3, 4, 6, 8, 11, 16, 33, 64, 3, 2, 1, 5):
            chunk = toks[at:at + rows]
            out.append((forward(w, e.st, e.buf, chunk)[:rows].clone(), e.buf.moe.pick[:rows].clone(),
                        e.buf.moe.wts[:rows].clone()))
            commit(w, e.st, e.buf, rows, max(1, rows - 1))         # keep all but the last row, as a verify does
            at += max(1, rows - 1)
        torch.cuda.synchronize()
        runs.setdefault(fused, []).append(out)
        del e
    assert used, "the fused route never ran"
    ref = runs[False][0]
    for out in runs[False][1:] + runs[True]:
        for (lg, pk, wt), (rlg, rpk, rwt) in zip(out, ref):
            assert torch.equal(pk, rpk) and torch.equal(wt.view(torch.int32), rwt.view(torch.int32))
            assert torch.equal(lg.view(torch.int16) if lg.element_size() == 2 else lg.view(torch.int32),
                               rlg.view(torch.int16) if rlg.element_size() == 2 else rlg.view(torch.int32))


@needs_model
@pytest.mark.parametrize("graphs", [False, True])
def test_cut_model_drafts_emit_serial_tokens_fused_and_not(cut_model, monkeypatch, graphs):
    from tensorfold.cuda.exl3 import experts
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, w.cfg.vocab, size=37)]
    toks = {}
    for fused in (False, True):
        monkeypatch.setattr(experts, "ROUTE_FUSED", fused)
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=graphs)
        for sampling in (None, Sampling(seed=1234, top_k=20, top_p=0.95)):
            s = serial_decode(e, prefill(e, prompt, sampling, mtp=False), 48, sampling)
            d = mtp_decode(e, prefill(e, prompt, sampling, mtp=True), 48, sampling, depth=6)
            assert s.tokens == d.tokens, (fused, sampling)
            toks[(fused, sampling is None)] = d.tokens
        del e
    assert toks[(True, True)] == toks[(False, True)] and toks[(True, False)] == toks[(False, False)]
