"""Nemotron-H family support with fused decode kernels and verified MTP draft chains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("nemotron_h",)
TITLE = "Nemotron 3.5 Lightning"
MODELS = ("TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",)
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}


GROUPS = (32, 64)          # 4-bit groups the format check accepts (the CUDA kernels read CUDA_QUANTIZATION)


def refusal(config: dict[str, Any]) -> str | None:
    """Why the kernels cannot read a checkpoint, from config.json: they read MLX 4-bit weights in GROUPS only."""

    from tensorfold.families import describe_quantization, layer_quantization, quantization

    bits, group = quantization(config)
    odd = sorted({f"{b}-bit g{g}" + ("" if m == "affine" else f" {m}") for path, (b, g, m)    # embeddings: a lookup
                  in layer_quantization(config).items()
                  if not (b == 4 and g in GROUPS and m == "affine") and not path.endswith("embeddings")})
    if bits == 4 and group in GROUPS and not odd:
        return None
    found = describe_quantization(config) + (f", with layers at {', '.join(odd)}" if odd else "")
    return (f"its kernels read MLX 4-bit weights in groups of 32 or 64 in every projection and expert (the expert "
            f"kernel reads any other width as 4-bit); this checkpoint has {found}")


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone, before any weight downloads, a checkpoint the kernels cannot read."""

    from tensorfold.families import read_config

    why = refusal(read_config(model_dir))
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")


# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29571, no_drafts: bool = False, mtp_drafts: int | None = None,
                mtp_confidence: float | None = None, context: int | None = None, **options: Any):
    """The CUDA engine: MTP chains verified exactly on one GPU or two (``tp=2``; start rank 1 first)."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP head on CUDA: a separate draft model does not apply")
    from .cuda import CONFIDENCE, CONTEXT, DRAFTS
    from .cuda.app import NemotronEngine

    drafts = 0 if no_drafts else DRAFTS if mtp_drafts is None else int(mtp_drafts)
    confidence = CONFIDENCE if mtp_confidence is None else float(mtp_confidence)
    explicit = bool(options.get("context_explicit"))
    return NemotronEngine(Path(model_dir), drafts=drafts, confidence=confidence,
                          context=context if explicit else CONTEXT, context_explicit=explicit, tp=int(tp),
                          rank=int(rank), master=master, port=int(master_port))
