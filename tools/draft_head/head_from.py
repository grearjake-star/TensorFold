#!/usr/bin/env python3
"""Carry one checkpoint's MTP head over to another checkpoint's format, untrained (workstream D's zero-GPU-training
probe): e.g. hibrid48's or the bf16 original's bf16 MTP linears as a TF_MTP_HEAD file for an EXL3 pack, whose own
MTP linears are 3-4 bit trellises. The routed experts, embedding and draft head stay the target's (the file holds only
what the target's loader lets a head replace). Drafts change speed only, never output.

usage: head_from.py <from_model_dir> <to_model_dir> <out.safetensors>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as SRC  # noqa: E402

a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
src, dst = SRC.source(a, "cpu"), SRC.source(b, "cpu")
tensors = {n: src.linear(n) for n in dst.trainable}
tensors.update({n: src.norm(n) for n in SRC.NORMS})
diff = {n: float((tensors[n].float() - dst.linear(n).float()).norm() / dst.linear(n).float().norm()) for n in dst.trainable}
dst.export(tensors, out, {"format": "tensorfold-mtp-head", "item": "D", "from": str(a), "to": str(b),
                          "source": src.kind, "target": dst.kind})
print(f"{src.kind} -> {dst.kind}: {len(tensors)} tensors to {out}")
for n, v in diff.items():
    print(f"  rel. difference {n}: {v:.4f}")
