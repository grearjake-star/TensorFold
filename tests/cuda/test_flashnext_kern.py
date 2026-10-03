"""K (format-agnostic speed): every change keeps each row's bits.

EXL3 routed experts on a plan's items (prompt windows): each pair's output equals the grouping kernel's, bit for bit,
for every codebook, mixed and 4-bit-only widths, 1, 2 and 4 member tiles an item, skewed routing (experts with many
member tiles), skipped picks, and windows on both sides of the switch row count."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _trellis(k, n, k2, gen):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, gen):
    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().cuda()


def _layer(E, D, I, kfun, cb, seed):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = kfun(e)
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _skewed_picks(E, R, k, gen):
    """Top-k of a strongly biased router (a few experts take most rows: many member tiles), the last slot the shared
    expert (id E - 1 here), and every 7th row's slot 0 a skipped pick (id E)."""

    bias = torch.linspace(4.0, -4.0, E - 1)
    logits = torch.randn((R, E - 1), generator=gen) + bias
    sel = logits.topk(k, dim=1).indices.to(torch.int32)
    sel = torch.cat([sel, torch.full((R, 1), E - 1, dtype=torch.int32)], 1)
    sel[::7, 0] = E
    w = torch.rand((R, k + 1), generator=gen) * 0.2 + 0.05
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


CASES = [
    ("mul1-4bit", 2, lambda e: (8, 8, 8)),                                                  # EXL3 4.05's experts
    ("mul1-mixed", 2, lambda e: ((6, 6, 6) if e % 4 else (12, 12, 12))),                     # 3.05-like + 6-bit shared
    ("mcg", 1, lambda e: ((2, 4, 6, 8, 10, 12, 14, 16)[e % 8],) * 2 + ((4, 6, 8)[e % 3],)),
    ("3inst", 0, lambda e: ((4, 8, 16)[e % 3],) * 3),
]


@pytest.mark.parametrize("name,cb,kfun", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("tiles", [1, 4])     # the plan holds 16 or 64 pairs an item: 2 tiles never ran
def test_items_give_the_grouping_kernels_bits(name, cb, kfun, tiles, monkeypatch):
    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, ROWS = 33, 512, 256, 6, 300
    ex = _layer(E, D, I, kfun, cb, seed=5 + cb)
    g = torch.Generator().manual_seed(17 + tiles)
    x = (torch.randn((ROWS, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    sel, w = _skewed_picks(E, ROWS, TOPK, g)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)

    def run(R, item_rows, wts):
        monkeypatch.setattr(experts, "PROMPT", "items" if item_rows else "group")
        monkeypatch.setattr(experts, "ITEM_ROWS", item_rows)
        monkeypatch.setattr(experts, "ITEM_TILES", tiles)
        got = experts.routed(x[:R], sel[:R].contiguous(), w[:R].contiguous() if wts else None, ex, scratch, None, R)
        return got.clone()

    for R in (65, 129, 300):
        for wts in (False, True):
            ref = run(R, 0, wts)                       # the grouping kernel, one program a (expert, member tile)
            got = run(R, 64, wts)                      # the plan's items
            assert torch.isfinite(ref).all()
            assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), (name, tiles, R, wts)


def test_decode_windows_keep_the_grouping_kernel(monkeypatch):
    """At or below ITEM_ROWS rows (decode, graphs) routed() takes the grouping path: no plan is made."""

    from tensorfold.cuda.exl3 import experts

    ex = _layer(9, 512, 256, lambda e: (8, 8, 8), 2, seed=1)
    scratch = experts.Scratch(ex, 64, 3)
    x = torch.randn((8, 512)).to(torch.bfloat16).cuda()
    sel = torch.randint(0, 9, (8, 3), dtype=torch.int32).cuda()
    experts.routed(x, sel, None, ex, scratch, None, 8)
    assert scratch._plan is None
