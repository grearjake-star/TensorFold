#!/usr/bin/env python3
"""In-process decode speed on a Flash Next checkpoint (--model) with the loaded MTP head (the checkpoint's own, or TF_MTP_HEAD), eager as
hibrid48 decodes, on held-out prompts (the decode suite's; never trained on): the household suite's settings
(tf_decode_sampled.py: T 0.7, top-k 20, top-p 0.8, thinking off, 256 tokens past EOS) on chat/prose/advice/code/
reasoning with seeds 1234-1238, plus greedy on each and the agent prompt. Each case runs under every --policy
(depth/confidence/cost; drafts never change output); tok/s, drafts, accepted, rounds and the reply's token hash are
recorded. --serial also decodes each case serially and checks drafted == serial.

usage: speed.py <out.json> --model <dir> [--serial] [--tokens 256] [--policy 10/0.5/0.06 ...] [--seeds 1234 ...]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record import BENCH  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import decode as D  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", required=True)
ap.add_argument("--serial", action="store_true")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--policy", nargs="+", default=["10/0.5/0.06"])
ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236, 1237, 1238])
ap.add_argument("--kinds", nargs="+", default=["chat", "prose", "advice", "code", "reasoning", "agent"])
ap.add_argument("--greedy", action="store_true", help="also a greedy run per prompt")
ap.add_argument("--lookup", nargs="+", default=["off"], help="prompt-lookup modes to run per policy (off agree prefer)")
ap.add_argument("--edit", type=int, default=0, help="also N repo-edit prompts (bakeoff coding tasks with files)")
a = ap.parse_args()
M = Path(a.model)
tok = Tokenizer.from_file(str(M / "tokenizer.json"))


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts_v1 as PV1  # noqa: E402

TEXTS = {k: BENCH[f"bench-{k}"] for k in ("chat", "prose", "advice", "code", "reasoning", "agent")
         if f"bench-{k}" in BENCH}   # the prompts file's ``bench`` entries (prompts_v1)
for i, item in enumerate([x for x in PV1._bakeoff() if x["kind"] == "code" and "Workspace files" in
                          x["messages"][0]["content"]][:a.edit]):
    TEXTS[f"edit{i}"] = item["messages"][0]["content"]
    a.kinds.append(f"edit{i}")


def ids(text):
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids


t0 = time.time()
fe = FlashNextEngine(M, max_len=8192, context_explicit=True)
e = fe.e
head = fe.w.meta.get("mtp_head") or "checkpoint"
print(f"loaded in {time.time() - t0:.0f}s head={head} graphs={e.graphs is not None}", flush=True)
policies = [tuple(float(x) for x in p.split("/")) for p in a.policy]
runs = []
with torch.no_grad():
    D.prefill(e, ids(BENCH["bench-chat"]), None)                   # warm-up
    D.mtp_decode(e, e.first, 32, None)
    for kind in a.kinds:
        text = TEXTS[kind]
        cases = ([("greedy", None)] if a.greedy else []) + [
            (f"s{s}", Sampling(seed=s, temperature=0.7, top_k=20, top_p=0.8)) for s in a.seeds]
        for label, samp in cases:
            p = ids(text)
            if len(p) + a.tokens > 8000:
                print(f"skip {kind}: prompt {len(p)}", flush=True)
                continue
            ref = None
            for (depth, conf, cost), lk in [(pp, m) for pp in policies for m in a.lookup]:
                first = D.prefill(e, p, samp)
                e.lookup_rounds = 0
                r = D.mtp_decode(e, first, a.tokens, samp, depth=int(depth), confidence=conf, cost=cost, context=p,
                                 lookup="" if lk == "off" else lk)
                h = hashlib.sha256(json.dumps(r.tokens).encode()).hexdigest()[:16]
                row = {"kind": kind, "run": label, "policy": f"{int(depth)}/{conf}/{cost}", "lookup": lk,
                       "lookup_rounds": e.lookup_rounds,
                       "tps": round(r.tokens_per_second, 2), "drafted": r.drafted, "accepted": r.accepted,
                       "rounds": r.rounds, "tpr": round((r.accepted + r.rounds) / max(1, r.rounds), 3), "hash": h}
                ref = ref or h
                row["same_as_first_policy"] = h == ref
                runs.append(row)
                print(head, row, flush=True)
            if a.serial:
                first = D.prefill(e, p, samp)
                s = D.serial_decode(e, first, a.tokens, samp)
                sh = hashlib.sha256(json.dumps(s.tokens).encode()).hexdigest()[:16]
                runs.append({"kind": kind, "run": label, "policy": "serial", "tps": round(s.tokens_per_second, 2),
                             "hash": sh, "serial_equal": sh == ref})
                print(head, runs[-1], flush=True)
Path(a.out).write_text(json.dumps({"head": head, "tokens": a.tokens, "runs": runs}, indent=1))
print(f"done {time.time() - t0:.0f}s", flush=True)
