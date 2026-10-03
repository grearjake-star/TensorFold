"""A buffer pass's matmul for any matrix the weights hold: 4-bit Q4, an NVFP4 checkpoint's bf16, an EXL3 pack's."""

from __future__ import annotations

import torch

from . import bf16, qmm
from .state import Buffers


def mm(x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, out: torch.Tensor, b: Buffers, **kw) -> torch.Tensor:
    if getattr(q, "kernel", "qmm") == "b16":      # an NVFP4 checkpoint's BF16 linear (non-experts)
        return bf16.matmul(x, q, out=out)
    if not isinstance(q, qmm.Q4):                 # an EXL3 pack's matrix (``exl3_mm``): prompts on its prompt path
        return q.prefill(x, out) if b.prefill else q(x, out)
    mm = qmm.prefill_matmul if b.prefill else qmm.matmul
    return mm(x, q, xs, out=out, part=b.part, **kw)
