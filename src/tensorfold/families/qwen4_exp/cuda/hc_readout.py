"""Hyper-connection write-back and read-out (normed streams -> down -> SiLU / inject -> up -> mix), per weight format."""

from __future__ import annotations

import torch

from . import bf16, glue, qmm
from .hc_check import fuser as _hc_fuser
from .matmul import mm as _mm
from .state import Buffers
from .weights import HC


def hc_block(hc: HC, b: Buffers, R: int, eps: float, streams: int, low: int, mode: int, inject_prev,
             inject_out, h: torch.Tensor, branch=None, y=None, wts=None) -> None:
    """Write the pending branch back into the streams h (in place), then the hyper-connection's read-out: b.mixed [R, D] (+ group sums), and its inject gates into ``inject_out``."""

    if b.prefill and R > FUSED_ROWS and isinstance(hc.down, qmm.Q4):
        fused = _hc_fuser(h.device)
        if fused is not None:
            fused(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps, mode,
                  branch=branch, inject=inject_prev, y=y, wts=wts)
            _readout_plain(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None, normed=True)
            return
    glue.hc_writeback(h[:R], h[:R], b.pss[:R], streams, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
    _readout(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None)


FUSED_ROWS = 16      # decode windows: the read-out in 3 kernels; wider windows (prefill) in 5, the same bits


def _readout(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """normed streams -> down -> SiLU / inject -> up -> mix: b.mixed [R, D] and its group sums."""

    if getattr(hc.down, "kernel", "qmm") == "b16":     # an NVFP4 checkpoint: the same steps, bf16 kernels
        _readout_b16(hc, b, h, R, eps, streams, low, inject)
    elif R <= FUSED_ROWS and not b.prefill and isinstance(hc.down, qmm.Q4):
        _readout_fused(hc, b, h, R, eps, streams, low, inject)
    else:
        _readout_plain(hc, b, h, R, eps, streams, low, inject)


def _readout_b16(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The read-out on the bf16 kernels: norm, down, activation and inject gates, up, the mix; same bits per row."""

    glue.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps)
    got = bf16.matmul(b.normed[:R], hc.down.b, out=torch.empty((R, hc.down.n), dtype=torch.float32,
                                                               device=h.device), f32=True)
    glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    _mm(b.act[:R], hc.up, b.xs_act[:R], b.up[:R], b)
    glue.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_fused(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The norm inside the down projection, the mix inside the up projection."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    got = qmm.hc_down(h[:R], b.pss[:R], hc.scale, b.normed[:R], hc.down, eps, streams, out=out, part=b.part)
    if got.dim() == 3:
        glue.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    qmm.hc_upmix(b.act[:R], b.xs_act[:R], hc.up, b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_plain(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject,
                   normed: bool = False) -> None:
    """The norm, the down projection with SiLU and the inject gates, the up projection, the mix: separate kernels."""

    if b.prefill and not normed and _exl3_hc().fusable(hc.prefill_down, hc.prefill_up, R):    # EXL3 fp16 faces
        _readout_exl3_prompt(hc, b, h, R, eps, streams, low, inject)
        return
    if not normed:
        glue.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps)
    _down_act(hc, b, R, streams, low, inject)
    if b.prefill and R >= 512 and streams == 4 and low == 320 and isinstance(hc.up, qmm.Q4) and hc.up.n == 10240:
        upmix = _hc_fuser(h.device, upmix=True)
        if upmix is not None:
            upmix(b.act[:R], hc.up, b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)
            return
    _mm(b.act[:R], hc.prefill_up if b.prefill else hc.up, b.xs_act[:R], b.up[:R], b)
    glue.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_exl3_prompt(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int,
                         inject) -> None:
    """_readout_plain's bits on an EXL3 pack's fp16 faces in three kernels (``exl3_hc``): norm in down, act, up + mix."""

    out = b.dn[:R] if hc.prefill_down.n == b.dn.shape[1] else b.dn_mix[:R]
    exl3_hc = _exl3_hc()
    got = exl3_hc.hc_down_prompt(h[:R], b.pss[:R], hc.scale, eps, streams, hc.prefill_down, out)
    glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    exl3_hc.hc_upmix_prompt(b.act[:R], hc.prefill_up, h[:R], b.pss[:R], hc.scale, eps, streams, b.mixed[:R], None)


def _down_act(hc: HC, b: Buffers, R: int, streams: int, low: int, inject) -> None:
    """A hyper-connection's down projection, then SiLU and the inject gates: b.act, b.xs_act (and ``inject``). With a split K the slice sum is fused into the activation kernel (the same bits as reduce, then act)."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    if isinstance(hc.down, qmm.Q4):
        got = _mm(b.normed[:R], hc.prefill_down if b.prefill else hc.down, b.xs_normed[:R], out, b, reduce=False)
    elif b.prefill:                                   # an EXL3 pack's fp16 matrix: summed slices, any row count
        got = (hc.prefill_down if hc.prefill_down is not None else hc.down)(b.normed[:R], out)
    else:
        got = hc.down.partials(b.normed[:R])          # fp32 slices [SK, R, N] that the activation sums in order
    if got.dim() == 3:
        glue.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)


def _exl3_hc():
    """The EXL3 fused read-out, imported on first use (its triton kernels need a real triton)."""

    from . import exl3_hc

    return exl3_hc
