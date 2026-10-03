#!/usr/bin/env python3
"""Self-distill a Flash Next MTP head on any checkpoint format (workstream D; v1's train.py on mtp_head + sources):
one chunked GPU job, resumable. The model is --model or the data dir's model.txt; the export is in its format.

usage: train.py <run_dir> <minutes> --data <data_dir> [...] [--steps N] [--lr 1e-4] [--depth 3]

Loss per chained pass d (1..depth) at row i: cross-entropy of the head's draft-vocabulary distribution against the
main model's own distribution at row i+d (its top-64 probabilities, exp(logit - lse), restricted to the draft
vocabulary), weight 1 where row i+d predicts a reply token and --prompt-weight on prompt rows; pass weights
--pass-weights. Checkpoints (trainable params + AdamW) in <run_dir>/ckpt.pt; the log in <run_dir>/train.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as DS  # noqa: E402
import sources as SRC  # noqa: E402
from mtp_head import MTPHead, long_forward, trainable  # noqa: E402

VOCAB = str(Path(__file__).resolve().parents[2] / "src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt")


def draft_ids() -> np.ndarray:
    return np.unique(np.loadtxt(VOCAB, dtype=np.int64).reshape(-1))


def chunk_loss(head, col, mixed, tv, ti, lse, w):
    logp = torch.log_softmax(torch.nn.functional.linear(mixed, head).float(), dim=-1)
    p = torch.exp(tv - lse[:, None])                         # main-model probabilities of its top 64
    c = col[ti]
    ok = c >= 0
    lp = torch.gather(logp, 1, c.clamp(min=0))
    ce = -(p * lp * ok).sum(-1)
    hit = (logp.argmax(-1) == c[:, 0]).float()               # greedy agreement (argmax main in draft vocab)
    return (ce * w).sum(), (hit * (w > 0)).sum()


def seq_loss(m: MTPHead, seqs, depth, pass_w, prompt_w, rows=512):
    streams, tokens, seq_id, pos, valid, gen = DS.pack(seqs, depth)
    top_v = torch.cat([s["top_v"] for s in seqs])
    top_i = torch.cat([s["top_i"] for s in seqs])
    lse = torch.cat([s["lse"] for s in seqs])
    outs = m(streams, tokens, seq_id, pos, depth, valid)
    T = streams.shape[0]
    total = torch.zeros((), device=streams.device)
    stats = []
    for d, mixed in enumerate(outs, start=1):
        tgt = torch.clamp(torch.arange(T, device=streams.device) + d, max=T - 1)
        reply = gen[torch.clamp(torch.arange(T, device=streams.device) + d + 1, max=T - 1)]   # token i+d+1
        w = valid[d - 1].float() * torch.where(reply, 1.0, prompt_w)
        wsum = w.sum().clamp(min=1)
        lsum = torch.zeros((), device=streams.device)
        for r0 in range(0, T, rows):
            sl = slice(r0, r0 + rows)
            l, _ = checkpoint(chunk_loss, m.head, m.col, mixed[sl], top_v[tgt[sl]], top_i[tgt[sl]], lse[tgt[sl]],
                              w[sl], use_reentrant=False)
            lsum = lsum + l
        total = total + pass_w[d - 1] * lsum / wsum
        stats.append(float(lsum / wsum))
    return total, stats


LONG_ROWS = 1536        # query rows sampled from a long sequence (> DS.MAXLEN rows) a step


def long_rows(u: dict) -> int:
    return LONG_ROWS if u["n"] > DS.MAXLEN else min(u["n"], DS.MAXLEN)


def seq_loss_long(m: MTPHead, s: dict, depth, pass_w, prompt_w, rng, rows=512):
    """A long sequence: queries at LONG_ROWS sampled rows whose drafted tokens are generated (sparse attention past
    BUDGET keys, ``long_forward``), weighted as seq_loss."""

    T, dev = s["n"], s["streams"].device
    gen = s["gen"]
    idx = torch.arange(T - depth - 1, device=dev)
    cand = idx[gen[idx + 2]]                                 # pass 1 drafts token i+2: a generated one
    if len(cand) == 0:
        cand = idx
    pick = torch.tensor(sorted(rng.sample(range(len(cand)), min(LONG_ROWS, len(cand)))), device=dev)
    S = cand[pick]
    pos = torch.arange(T, device=dev) + s.get("row0", 0)
    outs = long_forward(m, s["streams"], s["tokens"], pos, S, depth)
    total = torch.zeros((), device=dev)
    stats = []
    for d, mixed in enumerate(outs, start=1):
        tgt = S + d
        reply = gen[torch.clamp(S + d + 1, max=T - 1)]
        w = torch.where(reply, 1.0, prompt_w)
        wsum = w.sum().clamp(min=1)
        lsum = torch.zeros((), device=dev)
        for r0 in range(0, len(S), rows):
            sl = slice(r0, r0 + rows)
            l, _ = checkpoint(chunk_loss, m.head, m.col, mixed[sl], s["top_v"][tgt[sl]], s["top_i"][tgt[sl]],
                              s["lse"][tgt[sl]], w[sl], use_reentrant=False)
            lsum = lsum + l
        total = total + pass_w[d - 1] * lsum / wsum
        stats.append(float(lsum / wsum))
    return total, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("minutes", type=float)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--pass-weights", type=float, nargs="+", default=[1.0, 0.8, 0.6])
    ap.add_argument("--prompt-weight", type=float, default=0.1)
    ap.add_argument("--tokens", type=int, default=4096, help="rows per optimizer step (whole sequences)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--model", default=None, help="checkpoint dir (default: <first data dir>/model.txt)")
    ap.add_argument("--init", default=None, help="warm-start from a trained head file (the model's format)")
    ap.add_argument("--long-weight", type=float, default=1.0, help="sampling weight of long units (> 2048 rows)")
    ap.add_argument("--export-always", action="store_true", help="export at the time budget too (a warm start)")
    ap.add_argument("--house-weight", type=float, default=2.0,
                    help="sampling weight of household units (chat/prose/advice) relative to the rest")
    a = ap.parse_args()
    t0 = time.time()
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a)))
    torch.manual_seed(0)
    model = Path(a.model or (Path(a.data[0]) / "model.txt").read_text().strip())
    src = SRC.source(model)
    print(f"source {src.kind} {model}", flush=True)
    m = MTPHead(src, draft_ids(), override=src.read_head(Path(a.init)) if a.init else None)
    params = trainable(m)
    print(f"model ready {time.time() - t0:.0f}s; trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    step = 0
    ck = run / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location="cuda")
        with torch.no_grad():
            for p, v in zip(params, state["params"]):
                p.copy_(v)
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed at step {step}", flush=True)
    train = [u for u in DS.units(a.data, ("train",)) if u["n"] >= 16]
    rng = random.Random(1234 + step)
    uw = [a.long_weight if u["n"] > DS.MAXLEN else (a.house_weight if u["kind"] in ("chat", "prose", "advice") else 1.0)
          for u in train]
    print(f"{len(train)} train units ({sum(u['n'] > DS.MAXLEN for u in train)} long), "
          f"{sum(long_rows(u) for u in train)} rows a pass", flush=True)
    log = (run / "train.jsonl").open("a")
    budget = a.minutes * 60

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))))

    while step < a.steps and time.time() - t0 < budget:
        batch, rows = [], 0
        while rows < a.tokens:
            u = rng.choices(train, weights=uw)[0]
            batch.append(u)
            rows += long_rows(u)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tl = [0.0] * a.depth
        ts = time.time()
        # one sequence per forward (dense masks stay small); gradients accumulate over the batch
        total_rows = sum(long_rows(u) for u in batch)
        for u in batch:
            if u["n"] > DS.MAXLEN:
                s = DS.load(u, maxlen=10 ** 9)
                loss, st = seq_loss_long(m, s, a.depth, a.pass_weights, a.prompt_weight, rng)
            else:
                s = DS.load(u)
                loss, st = seq_loss(m, [s], a.depth, a.pass_weights, a.prompt_weight)
            share = long_rows(u) / total_rows
            (loss * share).backward()
            for j in range(a.depth):
                tl[j] += st[j] * share
            del s, loss
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        rec = {"step": step, "lr": lr_at(step - 1), "loss": [round(x, 4) for x in tl], "gnorm": float(gn),
               "rows": total_rows, "s": round(time.time() - ts, 2)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0:
            print(rec, flush=True)
    torch.save({"params": [p.detach() for p in params], "opt": opt.state_dict(), "step": step}, ck)
    print(f"saved step {step}; {time.time() - t0:.0f}s", flush=True)
    if step >= a.steps or a.export_always:
        m.export(run / "mtp_head.safetensors", {"steps": str(step), "args": json.dumps(vars(a))})
        (run / "TRAINED").write_text(f"{step}\n")
        print("exported", run / "mtp_head.safetensors", flush=True)


if __name__ == "__main__":
    main()
