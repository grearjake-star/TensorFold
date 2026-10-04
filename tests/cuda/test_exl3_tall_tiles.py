"""The 17-128-row EXL3 linear kernels (mid-M and ``linear_wc``, ashhart/TensorFold#260 commits 5-6) against today's
``linear_kernel`` on the Flash Next EXL3 pack's own projections: every output bit equal (``torch.equal``) at the
verify-window row counts, at row offsets inside a window, in every output dtype the engine uses, under a CUDA graph,
and the split counters left at zero for the next call.

Needs TENSORFOLD_EXL3_FLASHNEXT=<the EXL3 pack's folder> (read only; only the named projections are read)."""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import pytest
import torch

from tensorfold.cuda.exl3 import linear

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TENSORFOLD_EXL3_FLASHNEXT")
DEV = "cuda"
ROWS = (17, 24, 32, 48, 64, 96, 128)
L = "model.language_model.layers."
# every kind of Exl3Linear call a Flash Next 4.05 round makes (6-bit mul1 unless noted), one layer each
PREFIXES = [
    L + "0.linear_attn.in_proj_qkv",       # 2560 x 10240, one split: linear_wc
    L + "0.linear_attn.in_proj_z",         # 2560 x 6144, 4 splits over 48 blocks: linear_wc, splits summed in registers
    L + "0.linear_attn.out_proj",          # 6144 x 2560, 8 splits: linear_wc through Z
    L + "3.self_attn.q_proj",              # 2560 x 12288
    L + "3.self_attn.k_proj",              # 2560 x 512, 4 splits
    L + "3.self_attn.v_proj",
    L + "3.self_attn.o_proj",
    L + "3.self_attn.indexer.index_qk_proj",   # 4 bits: linear_wc, and the mid-M kernels at 33-40 rows
    "mtp.fc_hidden",                       # 5 bits: the mid-M kernels (the stock MTP head; the house head is fp16)
    "lm_head",                             # 2560 x 248320: every verify row's logits
]


def _pack():
    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    return exl3_pack.Pack(MODEL)


def _layer(pk, prefix: str) -> linear.Exl3Linear:
    """As the engine builds it (exl3_mm.x3)."""

    return linear.Exl3Linear.from_tensors(pk.get(prefix + ".trellis"), pk.scales(prefix, "su", "suh"),
                                          pk.scales(prefix, "sv", "svh"), pk.codebook(prefix),
                                          pk.get(prefix + ".bias") if pk.has(prefix + ".bias") else None, DEV)


@pytest.fixture(scope="module")
def pack():
    if not MODEL or not Path(MODEL).is_dir():
        pytest.skip("set TENSORFOLD_EXL3_FLASHNEXT")
    return _pack()


@needs_model
@pytest.mark.parametrize("prefix", PREFIXES, ids=[p.replace(L, "L") for p in PREFIXES])
def test_tall_tiles_give_linear_kernels_bits_on_the_pack(pack, prefix):
    layer = _layer(pack, prefix)
    assert layer.codebook == "mul1"
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(prefix.encode()))
    x = (torch.randn((128 + 40, layer.k), device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    for out_dtype in (torch.bfloat16, torch.float32):
        ref = layer(x[:128], out_dtype=out_dtype, mode=0)          # today's kernel, 16 rows a pass
        ref_off = layer(x[40:168], out_dtype=out_dtype, mode=0)
        for mode in (6, 7):
            for m in ROWS:
                kern = linear.tall_kernel(layer.bits, layer.codebook, m, layer.k, layer.n, layer.split, mode)
                got = layer(x[:m], out_dtype=out_dtype, mode=mode)
                assert torch.equal(got, ref[:m]), f"{prefix} mode {mode} ({kern}) {m} rows {out_dtype}"
                assert torch.equal(layer(x[40:40 + m], out_dtype=out_dtype, mode=mode), ref_off[:m]), \
                    f"{prefix} mode {mode} ({kern}) {m} rows at offset 40"
            assert int(layer.counters.abs().sum()) == 0, f"{prefix} mode {mode}: split counters left non-zero"
        # windows are row-invariant: a row's bits do not depend on the window it rides in
        for r in (0, 17, 127):
            assert torch.equal(layer(x[r:r + 1], out_dtype=out_dtype, mode=7), ref[r:r + 1])
        assert torch.equal(layer(x[:16], out_dtype=out_dtype, mode=7), ref[:16])     # <= 16 rows: today's kernel


@needs_model
def test_every_row_count_on_the_split_shapes(pack):
    """17..128 rows, every count, on the three split layouts (one split, register-folded splits, Z splits) and 4 bits."""

    for prefix in (L + "0.linear_attn.in_proj_qkv", L + "0.linear_attn.in_proj_z", L + "0.linear_attn.out_proj",
                   L + "3.self_attn.indexer.index_qk_proj"):
        layer = _layer(pack, prefix)
        x = (torch.randn((128, layer.k), device=DEV) * 0.5).to(torch.bfloat16)
        ref = layer(x, out_dtype=torch.bfloat16, mode=0)
        for mode in (6, 7):
            for m in range(17, 129):
                assert torch.equal(layer(x[:m], out_dtype=torch.bfloat16, mode=mode), ref[:m]), (prefix, mode, m)


@needs_model
def test_tall_tiles_under_a_cuda_graph(pack):
    """The engine replays rounds on CUDA graphs: a captured tall call replays linear_kernel's bits for new inputs."""

    layer = _layer(pack, L + "0.linear_attn.out_proj")
    m = 48
    x = torch.zeros((m, layer.k), device=DEV, dtype=torch.bfloat16)
    out = torch.empty((m, layer.n), device=DEV, dtype=torch.bfloat16)
    xh = torch.empty((m, layer.k), device=DEV, dtype=torch.float16)
    z = torch.empty((layer.split[0] * m * layer.n,), device=DEV, dtype=torch.float32)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        layer(x, out=out, xh=xh, z=z, mode=7)                     # warm-up (lazy smem attributes) outside capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer(x, out=out, xh=xh, z=z, mode=7)
    for seed in range(3):
        x.copy_((torch.randn((m, layer.k), device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed))
                 * 0.5).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, layer(x, out_dtype=torch.bfloat16, mode=0)), seed
