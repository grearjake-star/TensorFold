#!/usr/bin/env python3
"""Build a teacher-forcing set from saved model outputs (CPU; no decoding on the GPU): token sequences with the
spans the model generated, for record.py --set (workstream D, lean plan).

Sources:
- v1: data/v1 units (the MLX model's replies to v1's prompts; stored token ids): train / eval / bench splits kept.
- traces: bakeoff episodes saved as request/response traces (traces/bakeoff-tensorfold-hib48-*, traces/vllm-compare-*,
  traces/bakeoff-tensorfold-<MLX>): each episode rendered with the chat template (tools, reasoning, tool calls); the
  assistant turns are the generated spans. Long episodes keep their last --window rows (row0).
Selection (--plan): lean = every v1 household-kind train unit (chat/prose/advice), a third of the other v1 train
units, traces up to --trace-tokens, all v1 eval/bench units of every kind (held out); probe = 40 held-out v1 eval
household units + 80 v1 train units.
usage: build_set.py <out.jsonl> [--plan lean|probe] [--trace-tokens 160000] [--window 2056] [--max-episode 6000]
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts_v1 as P  # noqa: E402

ROOT = Path.cwd()  # the data root: run the tools from it
V1 = ROOT / "data/v1"
TOK = Tokenizer.from_file("models/qwen38-flash-next/tokenizer.json")
TRACES = ["traces/bakeoff-tensorfold-hib48-*/accuracy-*/results.jsonl", "traces/vllm-compare-*/accuracy-*/results.jsonl",
          "traces/bakeoff-tensorfold-2026*/accuracy-*/results.jsonl"]
HOUSE = ("chat", "prose", "advice")


def template():
    import jinja2

    tpl = Path("models/qwen38-flash-next/chat_template.jinja").read_text()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(tpl)


TPL = template()


def render(msgs, tools, gen_prompt, thinking=True) -> list[int]:
    return TOK.encode(TPL.render(messages=msgs, tools=tools, add_generation_prompt=gen_prompt,
                                 enable_thinking=thinking)).ids


def clean(m: dict) -> dict:
    m = {k: v for k, v in m.items() if v is not None}
    for tc in m.get("tool_calls") or []:
        f = tc.get("function", tc)
        if isinstance(f.get("arguments"), str):
            try:
                f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
            except json.JSONDecodeError:
                raise ValueError("unparsable tool arguments")
    return m


def text(msgs, tools, gen_prompt, thinking=True) -> str:
    return TPL.render(messages=msgs, tools=tools, add_generation_prompt=gen_prompt, enable_thinking=thinking)


def trace_unit(row: dict, label: str):
    """An episode: the last request's messages + the final response; spans = each assistant turn's tokens (by the
    rendered text's character offsets, so a token merging across a turn boundary counts as generated)."""

    last = row["trace"][-1]
    req = last["request"]
    msgs = [clean(copy.deepcopy(m)) for m in req["messages"]]
    msgs.append(clean(copy.deepcopy(last["response"]["choices"][0]["message"])))
    tools = req.get("tools")
    thinking = (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
    full = text(msgs, tools, False, thinking)
    enc = TOK.encode(full)
    starts = np.array([o[0] for o in enc.offsets])
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        a = text(msgs[:i], tools, True, thinking)
        b = text(msgs[:i + 1], tools, False, thinking)
        if not full.startswith(a) or not full.startswith(b):
            raise ValueError("a turn's text is not a prefix of the episode")
        spans.append([int(np.searchsorted(starts, len(a), side="right")) - 1 if len(a) else 0,
                      int(np.searchsorted(starts, len(b), side="left"))])
    return enc.ids, spans


LONG = [("traces/screen-x-exl3-405*/accuracy-*/results.jsonl", "exl3"),        # on-policy for EXL3 4.05 first
        ("traces/hard-lite-*/accuracy-*/results.jsonl", "hard"),              # Hermes-style ~19K-token prompts
        ("traces/bakeoff-tensorfold-hib48-*/accuracy-*/results.jsonl", "bake"),
        ("traces/vllm-compare-*/accuracy-*/results.jsonl", "bake"),
        ("traces/bakeoff-tensorfold-2026*/accuracy-*/results.jsonl", "bake")]
HELD = ("repo-money", "repo-config")          # speed.py's two repo-edit gate prompts: never trained on


def long_units(a, rng) -> list[dict]:
    """Full-length episodes of min_episode .. max_episode tokens (no window), on-policy EXL3 first, then the
    Hermes-style hard suite, then the bakeoff traces, up to long_tokens teacher-forced tokens."""

    out, total, seen = [], 0, set()
    for pat, src in LONG:
        rows = []
        for f in sorted(glob.glob(str(ROOT / pat))):
            rows += [(f, json.loads(l)) for l in open(f)]
        rng.shuffle(rows)
        for f, r in rows:
            if total >= a.long_tokens or not r.get("trace") or any(h in r["task_id"] for h in HELD):
                continue
            last = (r.get("usage") or [{}])[-1].get("total_tokens", 0)
            if not a.min_episode <= last <= a.max_episode:
                continue
            try:
                toks, spans = trace_unit(r, f)
            except Exception:                                      # noqa: BLE001
                continue
            if not a.min_episode <= len(toks) <= a.max_episode:
                continue
            key = (r["task_id"], len(toks))
            if key in seen:
                continue
            seen.add(key)
            uid = "long--" + hashlib.sha256((f + r["task_id"] + str(r.get("repeat"))).encode()).hexdigest()[:16]
            out.append({"uid": uid, "pid": r["task_id"], "kind": "agent" if r["category"] in ("agentic", "tool_use")
                        else ("code" if r["category"] == "coding" else "reasoning"), "split": "train", "mode": "trace",
                        "seed": 0, "src": "long", "tokens": toks, "spans": spans, "row0": 0,
                        "from": f.split("/traces/")[-1], "origin": src})
            total += len(toks)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--plan", choices=("lean", "probe", "long"), default="lean")
    ap.add_argument("--min-episode", type=int, default=4000, help="long plan: shortest episode (tokens)")
    ap.add_argument("--long-tokens", type=int, default=520000, help="long plan: total teacher-forced tokens")
    ap.add_argument("--trace-tokens", type=int, default=160000)
    ap.add_argument("--window", type=int, default=2056)
    ap.add_argument("--max-episode", type=int, default=6000, help="skip longer episodes (teacher-forcing cost)")
    a = ap.parse_args()
    rng = random.Random(7)
    out = []
    v1 = [json.loads(f.read_text()) for f in sorted((V1 / "units").glob("*.json"))]
    for m in v1:
        m["tokens"] = None
    house_tr = [m for m in v1 if m["split"] == "train" and m["kind"] in HOUSE]
    other_tr = [m for m in v1 if m["split"] == "train" and m["kind"] not in HOUSE]
    held = [m for m in v1 if m["split"] in ("eval", "bench")]
    rng.shuffle(other_tr)
    rng.shuffle(house_tr)
    if a.plan == "probe":
        pick = sum(([m for m in held if m["kind"] == k][:14] for k in HOUSE), []) + house_tr[:50] + other_tr[:30]
    else:
        pick = house_tr + other_tr[:len(other_tr) // 3] + held
    for m in pick:
        toks = np.load(V1 / "units" / (m["uid"] + ".npz"))["tokens"].tolist()
        out.append({"uid": "v1--" + m["uid"], "pid": m["pid"], "kind": m["kind"], "split": m["split"],
                    "mode": m["mode"], "seed": m["seed"], "src": "v1", "tokens": toks,
                    "spans": [[m["prompt_len"], len(toks)]], "row0": 0})
    n_trace = skipped = 0
    if a.plan == "lean":
        rows = []
        for pat in TRACES:
            for f in sorted(glob.glob(str(ROOT / pat))):
                rows += [(f, json.loads(l)) for l in open(f)]
        rng.shuffle(rows)
        for f, r in rows:
            if n_trace >= a.trace_tokens or not r.get("trace") or (r.get("usage") or [{}])[-1].get("total_tokens", 0) > a.max_episode:
                continue
            try:
                toks, spans = trace_unit(r, f)
            except Exception:                                      # noqa: BLE001 (counted, reported)
                skipped += 1
                continue
            row0 = max(0, len(toks) - a.window)
            spans = [s for s in spans if s[1] > row0]
            if not spans:
                continue
            uid = "trace--" + hashlib.sha256((f + r["task_id"] + str(r.get("repeat"))).encode()).hexdigest()[:16]
            split = "eval" if int(hashlib.sha256(r["task_id"].encode()).hexdigest(), 16) % 10 == 0 else "train"
            out.append({"uid": uid, "pid": r["task_id"], "kind": "agent" if r["category"] in ("agentic", "tool_use")
                        else ("code" if r["category"] == "coding" else "reasoning"), "split": split, "mode": "trace",
                        "seed": 0, "src": "trace", "tokens": toks, "spans": spans, "row0": row0,
                        "from": f.split("/traces/")[-1]})
            n_trace += len(toks) - row0
    if a.plan == "long":
        out = long_units(a, rng)
    with open(a.out, "w") as fh:
        for u in out:
            fh.write(json.dumps(u) + "\n")
    stored = sum(min(len(u["tokens"]), a.window) for u in out)
    forced = sum(len(u["tokens"]) for u in out)
    by = {}
    for u in out:
        k = (u["src"], u["split"], u["kind"])
        by[k] = by.get(k, 0) + 1
    print(f"{len(out)} units; rows stored {stored}; tokens teacher-forced {forced}; trace rows {n_trace}; "
          f"traces skipped {skipped}")
    for k, v in sorted(by.items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
