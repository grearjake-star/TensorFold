"""Prompt-row hyper-connection read-out on an EXL3 pack's fp16 faces: the norm inside the down projection's loads and
the stream mix in the up projection's epilogue, so the normed streams and the up rows never reach memory. Each value
uses the expressions, tiles and K order of glue._hc_normed / exl3_mm.F16 / glue._hc_mix, so the bits are the
five-kernel read-out's (tests/cuda/test_flashnext_hcfuse.py)."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .exl3_mm import F16

FUSED_MIN = 64                       # prompt read-outs past this many rows fuse; decode windows keep the five kernels
PROMPT_TILE = (64, 128, 64, 4)       # down: BM, BN, BK, warps
UPMIX_TILE = (64, 64, 64, 4)         # up + mix: 64 x 128 spills; every tile gives the same bits


@triton.jit
def _rinv_rows(PSS, rm, m_ok, s, eps, D: tl.constexpr, S: tl.constexpr, NC: tl.constexpr):
    """glue._hc_normed's rinv for rows ``rm`` and stream ``s``: the NC partial sums in order from 0.0."""

    total = tl.zeros(rm.shape, dtype=tl.float32)
    for c in range(NC):
        total += tl.load(PSS + (rm * NC + c) * S + s, mask=m_ok, other=0.0)
    return 1.0 / tl.sqrt(total / D + eps)


@triton.jit(do_not_specialize=["M"])
def _hc_down_fused(H, PSS, SCALE, W, OUT, M, N, o_stride, eps, K: tl.constexpr, KS: tl.constexpr, SK: tl.constexpr,
                   D: tl.constexpr, S: tl.constexpr, NC: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, BK: tl.constexpr):
    """_f16_fused on bf16(h * rinv_s * scale) (glue._hc_normed's values) loaded from the streams h; the n blocks of an
    m block are adjacent programs, so its rows are read from DRAM once."""

    pid = tl.program_id(0)
    pm = pid // NB
    pn = pid % NB
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    r0 = _rinv_rows(PSS, rm, m_ok, 0, eps, D, S, NC)          # S == 4 (the wrapper checks)
    r1 = _rinv_rows(PSS, rm, m_ok, 1, eps, D, S, NC)
    r2 = _rinv_rows(PSS, rm, m_ok, 2, eps, D, S, NC)
    r3 = _rinv_rows(PSS, rm, m_ok, 3, eps, D, S, NC)
    total = tl.zeros((BM, BN), dtype=tl.float32)
    for ps in range(SK):
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k0 = ps * KS
        for kk in range(0, KS, BK):
            k = k0 + kk
            s = k // D
            rinv = tl.where(s == 0, r0, tl.where(s == 1, r1, tl.where(s == 2, r2, r3)))
            hv = tl.load(H + rm[:, None] * K + (k + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
            sc = tl.load(SCALE + k + rk).to(tl.float32)
            x = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16)
            w = tl.load(W + rn[:, None] * K + (k + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x.to(tl.float16), tl.trans(w), acc)
        total += acc
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], total.to(OUT.dtype.element_ty),
             mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _bsig_mix(x):
    return (1.0 / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["M"])
def _hc_upmix_fused(ACT, W, H, PSS, SCALE, MIXED, XS, M, eps, K: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                    NC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, XS_ON: tl.constexpr):
    """Program (m block, d block): per stream s in order, _f16_mm's up rows s D + d (rounded to bf16 as stored), and
    glue._hc_mix's sum of bf16(bf16(sigmoid(up_s)) * normed_s) with normed_s recomputed as glue._hc_normed's; mixed and
    its 32-group sums."""

    pm = tl.program_id(0)
    pd = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rd = pd * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    total = tl.zeros((BM, BN), dtype=tl.float32)
    for s in tl.static_range(S):
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in range(0, K, BK):
            x = tl.load(ACT + rm[:, None] * K + (kk + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + (s * D + rd)[:, None] * K + (kk + rk)[None, :])
            acc = tl.dot(x.to(tl.float16), tl.trans(w), acc)
        u = acc.to(tl.bfloat16).to(tl.float32)
        rinv = _rinv_rows(PSS, rm, m_ok, s, eps, D, S, NC)
        hv = tl.load(H + rm[:, None] * (S * D) + s * D + rd[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        sc = tl.load(SCALE + s * D + rd).to(tl.float32)
        n = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16).to(tl.float32)
        total += (_bsig_mix(u) * n).to(tl.bfloat16).to(tl.float32)
    m = (total / S).to(tl.bfloat16)
    tl.store(MIXED + rm[:, None] * D + rd[None, :], m, mask=m_ok[:, None])
    if XS_ON:
        sums = tl.sum(tl.reshape(m.to(tl.float32), (BM, BN // 32, 32)), axis=2)
        g = pd * (BN // 32) + tl.arange(0, BN // 32)
        tl.store(XS + rm[:, None] * (D // 32) + g[None, :], sums, mask=m_ok[:, None])


def fusable(down, up, rows: int) -> bool:
    """Both faces fp16 (F16), the down split in K, and more than FUSED_MIN rows; ``TF_HC_FUSE=0`` keeps five kernels.

    Not written: b.normed, b.up, b.xs_normed, b.xs_mixed (a 2-D tile's 32-group sums do not follow _hc_mix's order).
    On an EXL3 pack no consumer of prompt mixed rows reads the group sums, and the mixer feeding the draft head reads
    at most len(ends) rows, which never fuse."""

    return (os.environ.get("TF_HC_FUSE", "1") != "0" and isinstance(down, F16) and isinstance(up, F16)
            and rows > FUSED_MIN and down.sk > 1)


def hc_down_prompt(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, eps: float, streams: int, down: F16,
                   out: torch.Tensor, stages: int = 2) -> torch.Tensor:
    """The read-out's down projection of the normed streams (computed in the loads) into ``out`` [M, n], bf16."""

    m, wide = h.shape
    if wide != down.k or h.stride(1) != 1 or h.stride(0) != wide or out.stride(-1) != 1:
        raise ValueError(f"hc_down_prompt: h {tuple(h.shape)} vs K={down.k}")
    if streams != 4 or (wide // streams) % 64:
        raise ValueError("hc_down_prompt: four streams of a multiple of 64 dims")
    bm, bn, bk, warps = PROMPT_TILE
    nb = triton.cdiv(down.n, bn)
    _hc_down_fused[(triton.cdiv(m, bm) * nb,)](
        h, pss, scale, down.w, out, m, down.n, out.stride(0), eps, K=down.k, KS=down.k // down.sk, SK=down.sk,
        D=wide // streams, S=streams, NC=pss.shape[1], NB=nb, BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def hc_upmix_prompt(act: torch.Tensor, up: F16, h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, eps: float,
                    streams: int, mixed: torch.Tensor, xs: torch.Tensor | None, tile=None, stages: int = 3) -> None:
    """The up projection of ``act`` and the stream mix with the normed streams recomputed from h: ``mixed`` [M, D]."""

    m, k = act.shape
    d = h.shape[1] // streams
    if k != up.k or up.n != streams * d or up.sk != 1 or act.stride(0) != k or mixed.stride(0) != d:
        raise ValueError("hc_upmix_prompt: shapes")
    bm, bn, bk, warps = tile or UPMIX_TILE
    _hc_upmix_fused[(triton.cdiv(m, bm), d // bn)](
        act, up.w, h, pss, scale, mixed, xs if xs is not None else mixed, m, eps, K=k, D=d, S=streams,
        NC=pss.shape[1], BM=bm, BN=bn, BK=bk, XS_ON=xs is not None, num_warps=warps, num_stages=stages)
