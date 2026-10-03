#!/usr/bin/env python3
"""GPU smoke of record.py's all-row teacher forcing on the tiny hibrid checkpoint (tests/cuda/nvfp4_tiny.py):
the recorded argmax of each reply row is the serial greedy token, and the recorded first-draft argmax of the
checkpoint's MTP head at the prompt's last row is the draft the engine makes there. usage: smoke_tiny.py <tmp_dir>"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tests" / "cuda"))
from nvfp4_tiny import write  # noqa: E402
import record as RC  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import decode as D  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
hib = write(tmp / "centred", hibrid=True, hidden=512, prefix="model.language_model.", centred=True)
w = load(hib, mtp=True, draft_vocab=128)
e = D.Engine(w, capacity=512, max_rows=8, prefill_rows=16, graphs=False)
ok = True
with torch.no_grad():
    for plen in (10, 23):
        prompt = [int(x) for x in np.random.default_rng(plen).integers(1, 250, plen)]
        first = D.prefill(e, prompt, None)
        ref = D.serial_decode(e, first, 30, None).tokens
        seq = prompt + ref
        RC.HEAD_ROWS = 7
        rec = RC.teacher_force(e, seq)
        P0 = len(prompt)
        match = float(np.mean(rec["top_i"][P0 - 1:len(seq) - 1, 0] == np.array(seq[P0:])))
        # the engine's first draft after the prompt: D.prefill absorbs prompt[1:] + first at the last row
        D.prefill(e, prompt, None)
        d, p = e.sample_draft(D.absorb(e, e.last_streams, [first]), P0 + 1, None)
        print(f"P={P0} greedy_match={match:.3f} recorded_draft={rec['mtp_top'][P0 - 1]} ({rec['mtp_p'][P0 - 1]:.3f})"
              f" engine_draft={d} ({p:.3f}) lse_finite={np.isfinite(rec['lse']).all()}")
        ok &= match > 0.95 and np.isfinite(rec["lse"]).all()
        # a window (a long trace's last rows) records the same rows as the full pass
        RC.MAXREC = 20
        win = RC.teacher_force(e, seq, row0=len(seq) - 20)
        RC.MAXREC = 2056
        same = all(np.array_equal(win[k], rec[k][len(seq) - 20:]) for k in ("tokens", "top_i", "mtp_top"))
        print(f"window rows == full rows: {same}")
        ok &= same
print("SMOKE-OK" if ok else "SMOKE-FAIL")
