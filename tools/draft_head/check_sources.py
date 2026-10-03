#!/usr/bin/env python3
"""CPU check of a checkpoint's Source (workstream D): build the PyTorch head, run two chained passes on random
streams with a backward, export the untrained head, and check (1) the file is what the format's loader takes:
names/dtypes/shapes against the checkpoint (EXL3: exl3.exl3_head), (2) reading it back gives the canonical tensors,
(3) for MLX and ModelOpt, the untrained export equals the checkpoint's own tensors bit for bit.
usage: check_sources.py <model_dir> <scratch_dir> [--no-experts]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as SRC  # noqa: E402
from mtp_head import MTPHead, trainable  # noqa: E402
from train import draft_ids  # noqa: E402

model, tmp = Path(sys.argv[1]), Path(sys.argv[2])
tmp.mkdir(parents=True, exist_ok=True)
t0 = time.time()
src = SRC.source(model, "cpu")
if "--no-experts" in sys.argv:
    src.experts = lambda: (torch.zeros((SRC.E, 640, 2560), dtype=torch.bfloat16),) * 2 + (
        torch.zeros((SRC.E, 2560, 640), dtype=torch.bfloat16),)
ids = draft_ids()
m = MTPHead(src, ids[::16] if "--no-experts" in sys.argv else ids)
print(f"{src.kind}: built in {time.time() - t0:.0f}s; trainable {sum(p.numel() for p in trainable(m)) / 1e6:.1f}M; "
      f"grid {len(m.grids)} linears; head rows {tuple(m.head.shape)}", flush=True)
T = 10
streams = (torch.randn(T, 4 * 2560) * 0.5).to(torch.bfloat16)
tokens = torch.randint(0, 200000, (T,))
pos = torch.arange(T)
outs = m(streams, tokens, torch.zeros(T, dtype=torch.int64), pos, 2, [pos + d <= T - 1 for d in (1, 2)])
lg = m.logits(outs[1])
lg.logsumexp(-1).mean().backward()
assert torch.isfinite(lg).all() and all(p.grad is not None for p in trainable(m))
path = tmp / f"untrained-{src.kind}.safetensors"
m.export(path, {"check": "untrained"})
back = src.read_head(path)
want = m.tensors()
worst = max(float((back[n].float() - (want[n] if n not in m.grids else m.W(n).float().detach())).abs().max())
            for n in want)
print(f"export -> read_head: {len(back)} tensors, max |diff| {worst:.3g} (bf16 rounding / grid)")
from safetensors.torch import load_file  # noqa: E402

sd = load_file(str(path))
if src.kind == "exl3":
    from tensorfold.families.qwen4_exp.cuda.exl3 import exl3_head
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    print("exl3 loader check: accepted", len(exl3_head(Pack(model), path)))
else:
    bad = [k for k, v in sd.items() if not (src.h.has(k) and src.h.shape(k) == list(v.shape)
                                             and torch.equal(src.h.get(k).view(v.dtype) if src.h.get(k).dtype != v.dtype and src.h.get(k).element_size() == v.element_size() else src.h.get(k), v))]
    print(f"untrained export vs checkpoint: {len(sd)} tensors, {len(bad)} differ {bad[:4]}")
print("CHECK-OK")
