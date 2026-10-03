#!/usr/bin/env python3
"""MTP tensors of a Flash Next checkpoint from its safetensors headers (no tensor data read): name, dtype, shape,
grouped with per-expert tensors collapsed. usage: inventory.py <model_dir> [--json out.json]"""
import argparse
import json
import re
import struct
from pathlib import Path


def headers(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], v["shape"], f.name)
    return out


def mtp(d: Path) -> dict:
    h = headers(d)
    got = {}
    for k, v in h.items():
        if ".mtp." not in "." + k and not k.startswith("mtp"):
            continue
        key = re.sub(r"\.experts\.\d+\.", ".experts.*.", k)
        if key in got:
            got[key]["count"] += 1
        else:
            got[key] = {"dtype": v[0], "shape": v[1], "file": v[2], "count": 1}
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = mtp(Path(a.model))
    for k, v in sorted(g.items()):
        print(k, v["dtype"], v["shape"], f"x{v['count']}" if v["count"] > 1 else "", v["file"])
    if a.json:
        Path(a.json).write_text(json.dumps(g, indent=1))
