#!/usr/bin/env python3
"""Summarize a D acceptance probe (d_probe.sh) as markdown: per held-out group, each head's pass-1 acceptance,
replayed tokens per round and modeled tok/s at the default policy, and the pre-registered kill rule:
KILL training for this checkpoint if no head beats the checkpoint's own by >= 2 points pass-1 on household
(chat+prose+advice, sampled) AND >= 3% modeled tok/s there. usage: probe_summary.py <probe_dir>"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
POL = "10/0.5/0.06"
evals = {p.stem[len("eval-"):]: json.loads(p.read_text()) for p in sorted(d.glob("eval-*.json"))}


def group(ev, kinds, mode):
    hit = rows = tok = ms = rounds = acc = 0
    for u in ev["units"]:
        if u["kind"] in kinds and u["mode"].startswith(mode):
            hit += u["pass_hit"][0]
            rows += u["pass_rows"][0]
            ro, dr, ac, m = u["sweep"][POL]
            tok += ro + ac
            ms += m
            rounds += ro
    return hit / max(1, rows), tok / max(1, rounds), 1000 * tok / max(1e-9, ms), rows


groups = {"household sampled": (("chat", "prose", "advice"), "sampled"),
          "household greedy": (("chat", "prose", "advice"), "greedy"),
          "code/reasoning/agent": (("code", "reasoning", "agent"), "")}
print(f"# Probe {d}\n\npolicy {POL}; verify table {d / 'data/timing.json'}\n")
print("| head | group | pass-1 | tokens/round | modeled tok/s | rows |\n|---|---|---|---|---|---|")
res = {}
for name, ev in evals.items():
    for g, (kinds, mode) in groups.items():
        p1, tpr, tps, rows = group(ev, kinds, mode)
        res[(name, g)] = (p1, tps)
        print(f"| {name} | {g} | {p1:.3f} | {tpr:.2f} | {tps:.1f} | {rows} |")
own = res.get(("own", "household sampled"))
verdict = "NO-DATA"
if own:
    best = max(((n, v) for (n, g), v in res.items() if g == "household sampled" and n != "own"),
               key=lambda x: x[1][1], default=None)
    if best:
        gain_p1, gain_tps = best[1][0] - own[0], best[1][1] / max(1e-9, own[1]) - 1
        ok = gain_p1 >= 0.02 and gain_tps >= 0.03
        verdict = (f"{'CONTINUE' if ok else 'KILL'}: best {best[0]} pass-1 {gain_p1:+.3f}, modeled tok/s "
                   f"{100 * gain_tps:+.1f}% vs own on household sampled")
print(f"\n**Verdict (pre-registered rule):** {verdict}")
