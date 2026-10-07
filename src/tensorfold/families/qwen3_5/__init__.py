"""Qwen3.8 dense uses row-exact serial and drafted decoding, resuming prefill only at chunk starts to preserve fresh-prefill bits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5",)
TITLE = "Qwen3.8 dense"
MODELS = ("TensorFold/Qwen3.8-27B-MLX-4bit", "turboderp/Qwen3.8-27B-exl3", "nvidia/Qwen3.8-27B-NVFP4")
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
QUANT_METHODS = {"cuda": ("mlx", "exl3", "modelopt", "compressed-tensors")}   # MLX affine, EXL3, NVFP4 / FP8
EXL3_VARIANT = "any"                           # every EXL3 codebook and width (tensorfold.families.EXL3_VARIANT_ANY)


def _language_specs(config: dict[str, Any]):
    from tensorfold.quantization import quantization_block, resolve_affine

    specs = {"": resolve_affine(config)}
    for path, value in (quantization_block(config) or {}).items():
        if ((isinstance(value, dict) or type(value) is bool)
                and not any(part in path.split(".") for part in ("vision_tower", "visual"))
                and not path.endswith("embed_tokens")):
            specs[path] = resolve_affine(config, path)
    return specs


def check_quantization(config: dict[str, Any], backend: str) -> None:
    from tensorfold.quantization import resolve_affine

    if resolve_affine(config) is None:
        raise ValueError("Qwen dense requires MLX affine quantization metadata; this checkpoint has none (unquantized weights)")
    if config.get("tie_word_embeddings") or (config.get("text_config") or {}).get("tie_word_embeddings"):
        raise ValueError("the tied embedding head is not supported by this packed Qwen decoder")
    _language_specs(config)


# The metadata-only info command also displays these CUDA affine formats.
CUDA_AFFINE_BITS = (2, 3, 4, 5, 6, 8)
CUDA_AFFINE_GROUPS = (32, 64, 128)
# --checkpoint-slots on CUDA: the prompt states the concurrent decoder keeps (--parallel 2 or more)
CUDA_CHECKPOINT_SLOTS = True
CUDA_PREFILL_FP8 = True            # --prefill-fp8: MLX 4-bit g64 and NVFP4 checkpoints have FP8 prompt kernels

def gb10() -> bool:
    """Whether GPU 0 is a GB10 (DGX Spark: compute capability 12.1), where the lone stream's wide windows were measured."""

    import torch

    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability(0)) == (12, 1) or "GB10" in torch.cuda.get_device_name(0)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """The CUDA engine for ``tensorfold serve``; tp=2 adds fp32 partials in rank order and needs the drafter on both."""

    from .cuda.engine import Qwen27Engine
    from .cuda.exl3_load import quant_config

    if quant_config(Path(model_dir)) is not None:
        print("[tensorfold] EXL3 packs are experimental: replies are exact; on a DGX Spark decode runs 0.8-1.2x the MLX "
              "checkpoint and prompts about half as fast (docs/recipes/qwen3.8-27b.md#exl3-checkpoints-experimental)", flush=True)
    if not drafter and not no_drafts:
        raise ValueError(f"{TITLE}'s CUDA engine drafts with {DRAFTER}, which is not here: without it every round "
                         f"would decode one token. Run `tensorfold pull {DRAFTER}` once (on both machines for "
                         "--tp 2), or pass --no-drafts for the serial reference")
    draft = Path(drafter) if drafter and not no_drafts else None
    streams = max(1, int(options.get("parallel") or 1))
    # one stream on one GB10 takes the width it affords (16-row trees, widening to 128); other shapes keep 12 rows
    wide = tp == 1 and streams == 1 and gb10()
    return Qwen27Engine(Path(model_dir), draft, max_rows=128 if wide else 12, tree_rows=16 if wide else None,
                        tp=tp, rank=rank, master=master, port=master_port,
                        split_head=tp == 2, tp_draft=tp == 2 and draft is not None, allow_copy=not no_drafts,
                        streams=streams, context=options.get("context"),
                        context_explicit=options.get("context_explicit"), vision=bool(options.get("vision", False)),
                        vision_urls=bool(options.get("vision_urls", False)),
                        vision_offload=bool(options.get("vision_offload", False)), keep=options.get("checkpoint_slots"))
