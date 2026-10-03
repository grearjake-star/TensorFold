#!/usr/bin/env python3
"""Tier A for prompt-lookup drafting (workstream D): on saved sequences (a build_set.py set), how many generated
tokens a lookup proposal (the continuation after the latest earlier occurrence of the last n tokens, n = NMAX..NMIN)
would have covered. Per generated position q (the next token to decode): propose up to L tokens; accepted = the
proposal's agreeing prefix with the actual text (drafts are verified against the sampled tokens, so this is exact for
the saved text). Simulates rounds where lookup drafts are used when a proposal exists (else 1 token a round, i.e.
no MTP), giving tokens/round from lookup alone and the share of positions with a useful proposal.
usage: lookup_sim.py <set.jsonl> [--nmin 2] [--nmax 4] [--L 10] [--min-len 2]"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("--nmin", type=int, default=2)
ap.add_argument("--nmax", type=int, default=4)
ap.add_argument("--L", type=int, default=10)
a = ap.parse_args()


def propose(seq, q, index):
    """The continuation after the latest earlier occurrence of seq[q-n:q], longest n first."""
    for n in range(a.nmax, a.nmin - 1, -1):
        if q < n:
            continue
        key = tuple(seq[q - n:q])
        at = index[n].get(key)
        if at is not None and at < q:
            return seq[at:at + a.L], n
    return [], 0


agg = collections.defaultdict(collections.Counter)
for line in open(a.set):
    u = json.loads(line)
    seq = u["tokens"]
    kind = u["src"] + "/" + u["kind"]
    index = {n: {} for n in range(a.nmin, a.nmax + 1)}
    gen = set()
    for s, e in u["spans"]:
        gen.update(range(s, e))
    q = 0
    # index grows as the sequence is read; occurrence = position right after the n-gram (latest wins)
    def add(upto, frm):
        for i in range(frm, upto + 1):
            for n in index:
                if i >= n:
                    index[n][tuple(seq[i - n:i])] = i
    added = 0
    for s, e in u["spans"]:
        add(s - 1, added)
        added = s
        q = s
        while q < e:
            prop, n = propose(seq, q, index)
            acc = 0
            for t in prop:
                if q + acc < e and seq[q + acc] == t:
                    acc += 1
                else:
                    break
            c = agg[kind]
            c["rounds"] += 1
            c["proposed"] += bool(prop)
            c["useful"] += acc >= 2
            c["long"] += acc >= 5
            c["long_tokens"] += (acc + 1) if acc >= 5 else 0
            c["acc"] += acc
            step = acc + 1                                 # the round's sampled token after the pending one + drafts
            c["tokens"] += step
            add(q + step - 1, added)
            added = q + step
            q += step
        add(e - 1, added)
        added = e
for kind, c in sorted(agg.items()):
    print(f"{kind:22s} gen tokens {c['tokens']:7d}  rounds {c['rounds']:7d}  tokens/round {c['tokens'] / c['rounds']:.2f}"
          f"  proposal {c['proposed'] / c['rounds']:.2f}, >=2 acc {c['useful'] / c['rounds']:.2f}, >=5 acc {c['long'] / c['rounds']:.3f}"
          f" covering {c['long_tokens'] / c['tokens']:.2f} of tokens")
