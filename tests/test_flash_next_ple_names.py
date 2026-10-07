"""Flash Next admission counts n-gram tables named shard_N and shards.N (the loader reads both), from headers only."""

from __future__ import annotations

import json
import struct

from tensorfold.families import qwen4_exp

NGRAM = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _file(path, tensors):
    header = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + offset)     # sparse: only headers take disk space


def test_tables_named_shards_dot_n_are_counted(tmp_path):
    _file(tmp_path / "model-0.safetensors", {
        f"{NGRAM}.shards.0.weight": 640, f"{NGRAM}.shards.0.scales": 80, f"{NGRAM}.shards.0.biases": 80,
        f"{NGRAM}.shards.1.weight": 640, f"{NGRAM}.shards.1.scales": 80, f"{NGRAM}.shards.1.biases": 80,
        "language_model.model.layers.0.ple.key_proj.weight": 2048,
    })
    assert qwen4_exp.ple_bytes(tmp_path) == 1600


def test_tables_named_shard_n_are_still_counted(tmp_path):
    _file(tmp_path / "model-0.safetensors", {
        f"{NGRAM}.shard_0.weight": 640, f"{NGRAM}.shard_0.scales": 80, f"{NGRAM}.shard_0.biases": 80,
        "language_model.model.layers.0.ple.key_proj.weight": 2048,
    })
    assert qwen4_exp.ple_bytes(tmp_path) == 800


def test_other_tensors_are_not_counted(tmp_path):
    _file(tmp_path / "model-0.safetensors", {f"{NGRAM}.shardsX.0.weight": 640,
                                              "language_model.model.layers.0.ple.key_proj.weight": 2048})
    assert qwen4_exp.ple_bytes(tmp_path) == 0
