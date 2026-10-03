#!/usr/bin/env python3
"""Offline gate for a Flash Next MTP head on any checkpoint (workstream D; v1's evaluate.py, serve-v3/v4's draft policy).

usage: evaluate.py <out.json> --data <dir> [...] [--head <mtp_head.safetensors | ckpt.pt>] [--splits eval bench]
                   [--timing <timing.json>]

On held-out units (hibrid48's own replies, never trained on), teacher-forced, the head chained to depth 10:
- per pass d: the draft (greedy argmax, or the keyed sample with the reply's own seed and position for sampled
  replies, candidates as ``Engine.sample_draft`` takes them) equals the model's token at that position;
- argmax agreement with the main model on every row, whatever the reply's mode;
- a replay of the decode loop over each reply with serve-v3's policy (``decode.draft``: up to 10 drafts, the first
  always, a later one only at >= 0.5 and while the chain's probability clears the cost bars): drafts and accepted
  drafts per round, and modeled ms from hibrid48's measured verify windows (``--timing``);
- for the stock head only, pass-1 agreement with the CUDA head's recorded argmax (this PyTorch head's fidelity).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as DS  # noqa: E402
import sources as SRC  # noqa: E402
from mtp_head import MTPHead, trainable  # noqa: E402
from train import draft_ids  # noqa: E402

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, sampler_probs  # noqa: E402

DEPTH, CONF, COST = 10, 0.5, 0.06
# (depth, confidence cutoff, cost in tokens/ms) policies replayed besides the default
POLICIES = [(10, 0.5, 0.06), (10, 0.5, 0.0), (10, 0.3, 0.06), (10, 0.4, 0.06), (10, 0.6, 0.06), (10, 0.7, 0.06),
            (6, 0.3, 0.0), (6, 0.5, 0.06), (8, 0.5, 0.06), (4, 0.5, 0.06), (10, 0.5, 0.03), (10, 0.5, 0.09),
            (10, 0.4, 0.03), (8, 0.4, 0.03), (10, 0.0, 0.0), (10, 0.3, 0.0), (10, 0.7, 0.0)]
VERIFY_MS: list[float] = []
DRAFT_MS = 0.0


def set_timing(path: str | None) -> None:
    """hibrid48's verify window of R rows (ms, R = 1..13) and one draft step; without a file the serve-v3 table."""

    global VERIFY_MS, DRAFT_MS
    if path:
        t = json.loads(Path(path).read_text())
        VERIFY_MS = [t["verify_ms"][str(r)] for r in range(1, len(t["verify_ms"]) + 1)]
        DRAFT_MS = float(t["draft_ms"])
    else:
        from tensorfold.families.qwen4_exp.cuda import decode as Dd

        VERIFY_MS, DRAFT_MS = list(Dd.VERIFY_MS), Dd.DRAFT_MS


def bars(cost: float, count: int) -> list[float]:
    ms = list(VERIFY_MS)
    while len(ms) < count + 3:
        ms.append(ms[-1] + (ms[-1] - ms[-2]))
    return [cost * (ms[j + 1] - ms[j] + DRAFT_MS) for j in range(count + 1)]


def verify_ms(rows: int) -> float:
    ms = list(VERIFY_MS)
    while len(ms) < rows:
        ms.append(ms[-1] + (ms[-1] - ms[-2]))
    return ms[rows - 1]


def load_head(path: str | None, model: Path) -> MTPHead:
    """The checkpoint's own head (None), a head file in the checkpoint's format, or a training ckpt.pt."""

    src = SRC.source(model)
    ids = draft_ids()
    if path is None:
        return MTPHead(src, ids)
    if path.endswith(".safetensors"):
        return MTPHead(src, ids, override=src.read_head(Path(path)))
    m = MTPHead(src, ids)
    state = torch.load(path, map_location=src.device)
    with torch.no_grad():
        for p, v in zip(trainable(m), state["params"]):
            p.copy_(v)
    return m


@torch.no_grad()
def unit_eval(m: MTPHead, s: dict, samp: Sampling | None) -> dict:
    n, P = s["n"], s["P"]
    tokens = s["tokens"]
    seq_id = torch.zeros((n,), dtype=torch.int64, device=tokens.device)
    pos = torch.arange(n, device=tokens.device)
    valid = [pos + d <= n - 1 for d in range(1, DEPTH + 1)]
    outs = m(s["streams"], tokens, seq_id, pos, DEPTH, valid)
    tok_np = tokens.cpu().numpy()
    main_arg = s["top_i"][:, 0].cpu().numpy()
    k = 20 + MARGIN
    drafts = np.full((DEPTH, n), -1, dtype=np.int64)       # pass d (0-based) at row i drafts token i + d + 2
    conf = np.zeros((DEPTH, n), dtype=np.float64)
    conf_s = np.zeros((DEPTH, n), dtype=np.float64)        # N1: the sampler-calibrated confidence
    argmax = np.full((DEPTH, n), -1, dtype=np.int64)
    for d, mixed in enumerate(outs):
        lg = m.logits(mixed).to(torch.bfloat16).float()    # the CUDA head's logits are bf16
        lse = torch.logsumexp(lg, dim=-1)
        top, col = lg.max(dim=-1)
        argmax[d] = m.draft_ids[col].cpu().numpy()
        if samp is None:
            drafts[d] = argmax[d]
            conf[d] = torch.exp(top - lse).cpu().numpy()
            conf_s[d] = conf[d]
        else:
            vals, idx = torch.topk(lg, k, dim=-1, sorted=False)
            ids = m.draft_ids[idx].cpu().numpy()
            v = vals.cpu().numpy().astype(np.float32)
            rows = np.arange(n)
            chosen = np.array(choose_rows(v, ids, list(rows + d + 2), samp), dtype=np.int64)
            hit = ids == chosen[:, None]
            pv = np.where(hit.any(1), v[rows, hit.argmax(1)], -np.inf)
            drafts[d] = chosen
            conf[d] = np.exp(pv - lse.cpu().numpy())
            conf_s[d] = sampler_probs(v, ids, list(chosen), samp)
    res = {"pass_hit": [], "pass_rows": [], "arg_hit": [], "arg_rows": []}
    for d in range(DEPTH):
        rows = np.arange(n)
        ok = (rows + d + 2 <= n - 1) & (rows + d + 2 >= P + 1)      # drafted tokens are reply tokens after the first
        tgt = tok_np[np.minimum(rows + d + 2, n - 1)]
        res["pass_hit"].append(int(((drafts[d] == tgt) & ok).sum()))
        res["pass_rows"].append(int(ok.sum()))
        mt = main_arg[np.minimum(rows + d + 1, n - 1)]
        res["arg_hit"].append(int(((argmax[d] == mt) & ok).sum()))
        res["arg_rows"].append(int(ok.sum()))
    # replay of mtp_decode + decode.draft: pending token index q, drafts from row q-1 (pass d drafts token q + d + 1)
    def replay(depth, cut, cost, conf=conf):
        q = P
        rounds = drafted = accepted = 0
        ms = 0.0
        bs = bars(cost, depth) if cost > 0 else None
        while q < n - 1:
            i = q - 1
            nd = 0
            chain = 1.0
            count = min(depth, n - 1 - q)
            for j in range(count):
                p = conf[j, i]
                chain *= p
                low = p < cut or (bs is not None and chain < bs[j])
                if low and j > 0:
                    break
                nd += 1
                if low or (bs is not None and chain < bs[j + 1]):
                    break
            kept = 0
            for d in range(nd):
                if drafts[d, i] != tok_np[q + d + 1]:
                    break
                kept += 1
            rounds += 1
            drafted += nd
            accepted += kept
            ms += verify_ms(1 + nd) + DRAFT_MS * nd
            q += kept + 1
        return rounds, drafted, accepted, ms

    rounds, drafted, accepted, ms = replay(DEPTH, CONF, COST)
    res["sweep"] = {f"{d}/{c}/{k}": list(replay(d, c, k)) for d, c, k in POLICIES if d <= DEPTH}
    res["sweep"].update({f"S:{d}/{c}/{k}": list(replay(d, c, k, conf_s)) for d, c, k in POLICIES if d <= DEPTH})
    res.update(rounds=rounds, drafted=drafted, accepted=accepted, tokens=n - P, ms=ms)
    if s["mtp_top"] is not None:
        rows = np.arange(n)
        ok = (rows + 1 <= n - 1)
        cuda_top = s["mtp_top"].cpu().numpy()
        res["fidelity_hit"] = int(((argmax[0] == cuda_top) & ok & (cuda_top >= 0)).sum())
        res["fidelity_rows"] = int((ok & (cuda_top >= 0)).sum())
    return res


@torch.no_grad()
def long_eval(m: MTPHead, dirs: list[str], rows: int = 512, depth: int = 3) -> dict:
    """Long units (> 2048 rows): at up to ``rows`` generated rows past 2048 keys (the indexer's sparse regime), pass
    d's argmax draft == the text token (acceptance proxy) and pass 1's agreement with the recorded CUDA head."""

    import random

    from mtp_head import long_forward

    rng = random.Random(5)
    hit = [0] * depth
    n = fid = fid_n = 0
    for u in DS.units(dirs, ("train", "eval"), ("long",)):
        s = DS.load(u, maxlen=10 ** 9)
        T = s["n"]
        idx = torch.arange(2048, T - depth - 2, device=s["tokens"].device)
        idx = idx[s["gen"][idx + 2]]
        if len(idx) == 0:
            continue
        S = idx[torch.tensor(sorted(rng.sample(range(len(idx)), min(rows, len(idx)))), device=idx.device)]
        outs = long_forward(m, s["streams"], s["tokens"], torch.arange(T, device=idx.device), S, depth)
        for d, mixed in enumerate(outs):
            arg = m.draft_ids[m.logits(mixed).argmax(-1)]
            hit[d] += int((arg == s["tokens"][S + d + 2]).sum())
            if d == 0:
                ok = s["mtp_top"][S] >= 0
                fid += int(((arg == s["mtp_top"][S]) & ok).sum())
                fid_n += int(ok.sum())
        n += len(S)
    return {"rows": n, "pass_acc": [round(h / max(1, n), 4) for h in hit], "fidelity_pass1": round(fid / max(1, fid_n), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--splits", nargs="+", default=["eval", "bench"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing", default=None)
    ap.add_argument("--long-data", nargs="*", default=[], help="data dirs with long units: sparse-region acceptance")
    ap.add_argument("--model", default=None, help="checkpoint dir (default: <first data dir>/model.txt)")
    a = ap.parse_args()
    set_timing(a.timing)
    t0 = time.time()
    m = load_head(a.head, Path(a.model or (Path(a.data[0]) / "model.txt").read_text().strip()))
    us = DS.units(a.data, tuple(a.splits), ("gen", "v1", "retf"))   # single-reply units (traces train only)
    if a.limit:
        us = us[:a.limit]
    per_unit = []
    for u in us:
        s = DS.load(u)
        samp = None if u["mode"].startswith("greedy") else Sampling(seed=u["seed"], temperature=0.7, top_k=20,
                                                                      top_p=0.8)
        r = unit_eval(m, s, samp)
        r.update({k: u[k] for k in ("uid", "kind", "split", "mode")})
        r["live"] = {"drafted": u.get("drafted") or 0, "accepted": u.get("accepted") or 0, "rounds": u.get("rounds") or 0}
        per_unit.append(r)
    agg: dict = collections.defaultdict(lambda: collections.Counter())
    for r in per_unit:
        mode = "greedy" if r["mode"].startswith("greedy") else "sampled"
        for key in (f"{r['kind']}/{mode}", f"all/{mode}", f"{r['split']}/{mode}"):
            c = agg[key]
            for d in range(DEPTH):
                c[f"hit{d + 1}"] += r["pass_hit"][d]
                c[f"rows{d + 1}"] += r["pass_rows"][d]
                c[f"arg{d + 1}"] += r["arg_hit"][d]
            for f in ("rounds", "drafted", "accepted", "tokens", "fidelity_hit", "fidelity_rows"):
                c[f] += r.get(f, 0)
            c["live_drafted"] += r["live"]["drafted"]
            c["live_accepted"] += r["live"]["accepted"]
            c["live_rounds"] += r["live"]["rounds"]
            c["units"] += 1
            for pol, (ro, dr, ac, ms_) in r["sweep"].items():
                c[f"sw_tok_{pol}"] += ro + ac
                c[f"sw_ms_{pol}"] += ms_
                c[f"sw_rounds_{pol}"] += ro
    summary = {}
    for key, c in sorted(agg.items()):
        summary[key] = {
            "units": c["units"],
            "pass_acc": [round(c[f"hit{d}"] / max(1, c[f"rows{d}"]), 4) for d in range(1, DEPTH + 1)],
            "argmax_agree": [round(c[f"arg{d}"] / max(1, c[f"rows{d}"]), 4) for d in range(1, DEPTH + 1)],
            "replay_acceptance": round(c["accepted"] / max(1, c["drafted"]), 4),
            "replay_tokens_per_round": round((c["accepted"] + c["rounds"]) / max(1, c["rounds"]), 3),
            "live_acceptance": round(c["live_accepted"] / max(1, c["live_drafted"]), 4),
            "live_tokens_per_round": round((c["live_accepted"] + c["live_rounds"]) / max(1, c["live_rounds"]), 3),
            "fidelity_pass1": round(c["fidelity_hit"] / max(1, c["fidelity_rows"]), 4),
            "model_tps": {pol: round(1000 * c[f"sw_tok_{pol}"] / max(1e-9, c[f"sw_ms_{pol}"]), 2)
                          for pol in [f"{d}/{cc}/{k}" for d, cc, k in POLICIES] + [f"S:{d}/{cc}/{k}" for d, cc, k in POLICIES]},
            "model_tpr": {pol: round(c[f"sw_tok_{pol}"] / max(1, c[f"sw_rounds_{pol}"]), 3)
                          for pol in [f"{d}/{cc}/{k}" for d, cc, k in POLICIES] + [f"S:{d}/{cc}/{k}" for d, cc, k in POLICIES]},
        }
    if a.long_data:
        summary["long/sparse"] = long_eval(m, a.long_data)
        print("long/sparse", json.dumps(summary["long/sparse"]), flush=True)
    Path(a.out).write_text(json.dumps({"head": a.head, "verify_ms": VERIFY_MS, "draft_ms": DRAFT_MS,
                                       "summary": summary, "units": per_unit}, indent=1))
    for key, v in summary.items():
        print(key, json.dumps(v), flush=True)
    print(f"{len(us)} units in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
