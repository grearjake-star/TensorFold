"""Recorded units (record.py) as training/eval sequences for the MTP head (v1's data.py; hibrid48 units add
``src``: gen = hibrid48's own reply, retf = a v1 reply teacher-forced through hibrid48)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MAXLEN = 2048          # the head's attention stays dense (no indexer selection) up to 2048 cache entries


def units(data_dirs: list[str], splits: tuple[str, ...], srcs: tuple[str, ...] = ("gen", "retf", "v1", "trace", "long")) -> list[dict]:
    out = []
    for d in data_dirs:
        for f in sorted(Path(d).glob("units/*.json")):
            meta = json.loads(f.read_text())
            if meta["split"] in splits and meta.get("src", "gen") in srcs:
                meta["path"] = str(f.with_suffix(".npz"))
                out.append(meta)
    return out


def load(meta: dict, device="cuda", maxlen: int = MAXLEN) -> dict:
    z = np.load(meta["path"])
    n = min(int(meta["n"]), maxlen)
    t = {
        "tokens": torch.from_numpy(z["tokens"][:n].astype(np.int64)).to(device),
        "streams": torch.from_numpy(z["streams"][:n].view(np.int16)).to(device).view(torch.bfloat16),
        "top_v": torch.from_numpy(z["top_v"][:n]).to(device),
        "top_i": torch.from_numpy(z["top_i"][:n].astype(np.int64)).to(device),
        "lse": torch.from_numpy(z["lse"][:n]).to(device),
        "mtp_top": torch.from_numpy(z["mtp_top"][:n].astype(np.int64)).to(device),
        "n": n, "P": int(meta["prompt_len"]), "meta": meta,
    }
    # rows whose token the model generated (the rows drafting serves): the reply, or a trace's assistant spans;
    # row0: the stored window's first absolute position (a long trace keeps its last rows)
    row0 = int(meta.get("row0", 0))
    gen = torch.zeros((n,), dtype=torch.bool)
    for a, b in meta.get("spans") or [[int(meta["prompt_len"]) + row0, int(meta.get("n_full", n + row0))]]:
        gen[max(0, a - row0):max(0, min(n, b - row0))] = True
    t["gen"], t["row0"] = gen.to(device), row0
    return t


def pack(seqs: list[dict], depth: int):
    """Concatenate sequences: streams, tokens, seq ids, positions, and per pass d the rows whose input token
    i+d exists in the same sequence."""

    streams = torch.cat([s["streams"] for s in seqs])
    tokens = torch.cat([s["tokens"] for s in seqs])
    dev = tokens.device
    seq_id = torch.cat([torch.full((s["n"],), j, dtype=torch.int64, device=dev) for j, s in enumerate(seqs)])
    rel = torch.cat([torch.arange(s["n"], device=dev) for s in seqs])
    pos = torch.cat([torch.arange(s["n"], device=dev) + s.get("row0", 0) for s in seqs])     # absolute (rope)
    length = torch.cat([torch.full((s["n"],), s["n"], dtype=torch.int64, device=dev) for s in seqs])
    gen = torch.cat([s["gen"] for s in seqs])
    valid = [rel + d <= length - 1 for d in range(1, depth + 1)]
    return streams, tokens, seq_id, pos, valid, gen
