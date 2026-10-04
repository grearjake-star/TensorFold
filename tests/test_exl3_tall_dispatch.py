"""The 17-128-row EXL3 linear kernels (mid-M and ``linear_wc``, ashhart/TensorFold#260 commits 5-6) on the CPU: the
``TF_EXL3_TALL`` switch, the mode ``Exl3Linear`` hands ``ext.linear``, and the dispatch for Flash Next 4.05's real
projection shapes (which kernel, scratch that fits, counters that fit). The bits themselves are checked on the GPU in
``tests/cuda/test_exl3_tall_tiles.py`` and ``tests/cuda/test_exl3_linear.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear

# (name, bits, K, N): the qwen38-flash-next-exl3-4.05bpw projections that go through Exl3Linear (experts do not)
PACK_SHAPES = [("in_proj_qkv", 6, 2560, 10240), ("in_proj_z", 6, 2560, 6144), ("out_proj", 6, 6144, 2560),
               ("q_proj", 6, 2560, 12288), ("k_proj", 6, 2560, 512), ("o_proj", 6, 6144, 2560),
               ("shared_expert.gate_proj", 6, 2560, 640), ("shared_expert.down_proj", 6, 640, 2560),
               ("index_qk_proj", 4, 2560, 640), ("lm_head", 6, 2560, 248320), ("mtp.fc_hidden", 5, 2560, 2560)]
SMEM_LIMIT = 99 * 1024            # GB10 (sm_121): the most dynamic shared memory one block may opt into


@pytest.mark.parametrize("env,mode", [({}, 7), ({"TF_EXL3_TALL": "1"}, 7), ({"TF_EXL3_TALL": "0"}, 0),
                                      ({"TF_EXL3_TALL": "off"}, 0), ({"TF_EXL3_TALL": "midm"}, 6),
                                      ({"TENSORFOLD_EXL3_MIDM": "0"}, 0), ({"TENSORFOLD_EXL3_WC": "0"}, 6),
                                      ({"TF_EXL3_TALL": "1", "TENSORFOLD_EXL3_MIDM": "0"}, 0)])
def test_the_switch(monkeypatch, env, mode):
    for name in ("TF_EXL3_TALL", "TENSORFOLD_EXL3_MIDM", "TENSORFOLD_EXL3_WC"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert linear.tall_mode() == mode


class _FakeExt:
    def __init__(self) -> None:
        self.modes: list[tuple[int, int]] = []

    def rot_in(self, x, suh, xh) -> None:
        pass

    def linear(self, xh, words, sk_stride, nb_stride, svh, bias, out, z, counters, k2, cb, sk, wk, mode) -> None:
        self.modes.append((xh.shape[0], mode))


def _layer(bits: float = 6, codebook: str = "mul1", kt: int = 16, nt: int = 8) -> linear.Exl3Linear:
    rng = np.random.default_rng(0)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    return linear.Exl3Linear.from_tensors(trellis, torch.ones(16 * kt, dtype=torch.float16),
                                          torch.ones(16 * nt, dtype=torch.float16), codebook, device="cpu")


def test_the_layer_hands_the_mode_to_the_kernel(monkeypatch):
    fake = _FakeExt()
    monkeypatch.setattr(linear, "_ext", lambda: fake)
    layer = _layer()
    x = torch.zeros((32, layer.k), dtype=torch.float16)
    monkeypatch.setattr(linear, "MODE", 7)
    layer(x)
    layer(x, mode=0)
    monkeypatch.setattr(linear, "MODE", 0)
    layer(x)
    layer(x[:5], mode=6)
    assert fake.modes == [(32, 7), (32, 0), (32, 0), (5, 6)]


@pytest.mark.parametrize("mode", [0, 6, 7])
@pytest.mark.parametrize("m", range(1, 17))
def test_sixteen_rows_or_fewer_keep_todays_kernel(m, mode):
    for _, bits, k, n in PACK_SHAPES:
        assert linear.tall_kernel(bits, "mul1", m, k, n, linear.plan(k, n), mode) == "linear_kernel"


def test_the_off_switch_keeps_todays_kernel_at_every_row_count():
    for _, bits, k, n in PACK_SHAPES:
        for m in range(1, 129):
            assert linear.tall_kernel(bits, "mul1", m, k, n, linear.plan(k, n), 0) == "linear_kernel"


def _wc_smem(k2: int, mt: int, ks: int, ns: int = 4) -> int:
    """WcLayout<K2, 16 MT, KS, NS>::SMEM in linear_wc.cuh."""

    raw = 8 * fmt.tile_words(k2 / 2) // 2       # fmt.tile_words counts int16 words, the kernel int32 words
    stage = ks * raw * 4 + ks * 16 * mt * 32
    return max(ns * stage, 16 * mt * 128 * 4)


@pytest.mark.parametrize("name,bits,k,n", PACK_SHAPES, ids=[s[0] for s in PACK_SHAPES])
def test_the_packs_projections(name, bits, k, n):
    """Every 17-128-row call of a 4.05 projection takes a taller kernel, with shared memory and counters that fit."""

    sk, wk = linear.plan(k, n)
    nb = n // 128
    for m in range(17, 129):
        got = linear.tall_kernel(bits, "mul1", m, k, n, (sk, wk), 7)
        fold = sk > 1 and nb >= 32
        if bits in (4, 6) and not (bits == 4 and not fold and 32 < m <= 40):
            assert got.startswith("linear_wc"), (name, m, got)
        else:
            assert got.startswith("linear_mpg"), (name, m, got)
        if got.startswith("linear_wc"):
            mt = int(got.split("MT=")[1][0])
            ks = int(got.split("KS=")[1][0])
            assert (k // 16 // sk // wk) % ks == 0                         # a stage never straddles a K range
            assert _wc_smem(linear.k2_of(bits), mt, ks) <= SMEM_LIMIT
            groups = -(-m // (16 * mt))
            assert groups * nb <= 8 * nb                                   # counters hold 8 N / 128 ints
        else:
            p = 2 if m > 48 else 1
            assert -(-m // 16) <= 8 and -(-m // (16 * p)) * p <= 8         # passes index counters [pass * NB + nb]
        assert linear.tall_kernel(bits, "mul1", m, k, n, (sk, wk), 6).startswith("linear_mpg")


def test_the_cuda_dispatch_keeps_sixteen_rows_on_linear_kernel():
    """The source guards the mirror above describes: mid-M only above 16 rows, linear_wc declines 16 rows or fewer."""

    here = Path(linear.__file__).parent
    cu = (here / "linear.cu").read_text()
    wc = (here / "linear_wc.cuh").read_text()
    assert "if (mode >= MIDM && M > 16)" in cu
    assert "if (per_warp % 2 || M <= 16 || M > 128) return false;" in wc
    assert "mode >= WC && cb == CB_MUL1" in cu
