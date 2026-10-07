"""Argument parser for the tensorfold command."""

from __future__ import annotations

import argparse
from typing import Callable

from tensorfold import __version__
from tensorfold.cuda.prompt_precision import FP8_BY_DEFAULT


def build_parser(handlers: dict[str, Callable[[argparse.Namespace], int]]) -> argparse.ArgumentParser:
    """The ``tensorfold`` parser, each subcommand bound to ``handlers[name]``."""

    parser = argparse.ArgumentParser(
        prog="tensorfold",
        description="Fast, exact LLM decoding on NVIDIA GPUs (DGX Spark / GB10 first) behind an OpenAI-compatible endpoint.",
    )
    parser.add_argument("--version", action="version", version=f"tensorfold {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve a model at an OpenAI-compatible endpoint",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    serve.add_argument("model", help="a Hugging Face repo id (downloaded on first use) or a model directory")
    endpoint = serve.add_argument_group("endpoint")
    endpoint.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: every interface)")
    endpoint.add_argument("--port", type=int, default=8080)
    endpoint.add_argument("--name", default="", help="model id clients ask for (default: the model's name)")
    endpoint.add_argument("--alias", action="append", default=[], help="another model id to answer to")
    endpoint.add_argument("--api-key", action="append", default=[], help="require this API key; repeat for more keys")
    endpoint.add_argument("--api-key-file", help="restricted key file, one key or label: key per line; # comments")
    endpoint.add_argument("--metrics-open", action="store_true", help="allow metrics without an API key")
    endpoint.add_argument("--vision", action="store_true",
                          help="enable image input for supported GLM and Qwen vision checkpoints")
    endpoint.add_argument("--vision-urls", action="store_true",
                          help="with --vision, accept public HTTP(S) image URLs (default: data URLs only)")
    endpoint.add_argument("--vision-offload", action="store_true",
                          help="with --vision on CUDA, keep the image tower in host RAM and copy it to the GPU only "
                               "while an image is encoded (frees about 5 GiB of the startup budget on a small card; "
                               "each image pays the copy)")
    endpoint.add_argument("--vision-max-images", type=int, default=None,
                          help="with --vision, maximum images across the full request history (default: 4); "
                               "byte, pixel and visual-token limits still apply")
    endpoint.add_argument("--vision-image-tokens", type=int, default=None,
                          help="with --vision on CUDA Qwen checkpoints, the visual tokens a request's images share "
                               "(default: 4096, at most 65536); each image keeps at most 4096")

    generation = serve.add_argument_group("generation (requests can override each of these)")
    generation.add_argument("--context", type=int, default=None,
                            help="prompt plus reply window (default/0: the affordable native capacity)")
    generation.add_argument("--max-tokens", type=int, default=4096,
                            help="reply tokens when a request does not say")
    generation.add_argument("--temperature", type=float, default=None,
                            help="0 decodes greedily (default: the model's generation_config.json, else 0)")
    generation.add_argument("--top-p", type=float, default=None, help="(default: the model's generation config)")
    generation.add_argument("--top-k", type=int, default=None, help="(default: the model's generation config)")
    generation.add_argument("--min-p", type=float, default=None,
                            help="keep tokens at least this share of the likeliest one's probability (default: the "
                                 "model's generation config, else 0: off)")
    generation.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                            help="open a think block when the chat template supports it")
    generation.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default=None,
                            help="default effort when a request omits one. An unnamed level maps to the nearest "
                                 "level the template names, and a tie takes the higher one. This flag uses that "
                                 "rule. xhigh stays xhigh, so GLM-5.3 renders it as Max")
    generation.add_argument("--thinking-budget", type=int, default=0,
                            help="most thinking tokens before the server closes the think block (0: no limit)")

    speed = serve.add_argument_group("drafting and caches")
    speed.add_argument("--no-drafts", action="store_true",
                       help="one token a round: the serial reference (same output, slower)")
    speed.add_argument("--drafter", default="auto",
                       help="a draft model (repo id or directory); auto: the family's draft model when it has been "
                            "pulled; none: no draft model")
    speed.add_argument("--mtp-drafts", type=int, default=None,
                       help="most MTP drafts a round (Qwen3.8 Flash Next: 10, stopping under 50%% confidence; "
                            "Nemotron: 15, stopping where a row stops paying); 0: no MTP drafts (any family)")
    speed.add_argument("--mtp-confidence", type=float, default=None,
                       help="stop an MTP chain before a later draft under this probability "
                            "(Flash Next default 0.50; Nemotron: by the row costs it measures at start)")
    speed.add_argument("--checkpoint-slots", type=int, default=None,
                       help="Qwen3.8-27B with --parallel 2 or more: the prompt states its concurrent decoder keeps "
                            "(default 3; one GPU keeps them while memory lasts, two ranks reserve a window each)")
    speed.add_argument("--parallel", default="auto",
                       help="requests decoded together, their windows sharing each round's forward: a number, or "
                            "auto (one at a time, the others waiting their turn)")
    speed.add_argument("--decode-share", type=float, default=None, help="Flash Next --parallel: replies decode inside "
                       "each prompt pass; a share sizes the passes so a round's decoding takes it (default 0: whole "
                       "passes)")
    speed.add_argument("--ple-on-ssd", action="store_true",
                       help="Flash Next: read the n-gram (PLE) tables from the checkpoint on SSD at each lookup "
                            "instead of holding them in memory. A trade: a few percent of decode speed for about "
                            "40 GiB less at peak (the tables are 29.8 GiB)")

    speed.add_argument("--no-update-check", action="store_true",
                       help="don't ask GitHub whether a newer release exists (also TENSORFOLD_NO_UPDATE_CHECK=1)")

    cuda = serve.add_argument_group("NVIDIA GPUs (DGX Spark / GB10)")
    cuda.add_argument("--backend", choices=("auto", "cuda"), default="auto",
                      help="kept for scripts: this build serves on CUDA only")
    cuda.add_argument("--tp", type=int, choices=(1, 2), default=1,
                      help="GPUs (one per machine) the model is split over; run the same command on each")
    cuda.add_argument("--rank", type=int, choices=(0, 1), default=0,
                      help="with --tp 2: this machine's rank; rank 0 serves HTTP, rank 1 follows it")
    cuda.add_argument("--master", default="", help="with --tp 2: rank 0's address on the link between the machines")
    cuda.add_argument("--master-port", type=int, default=29551, help="with --tp 2: rank 0's rendezvous port")
    cuda.add_argument("--kv-dtype", choices=("bf16", "int8", "int4"), default="bf16",
                      help="KV cache: bf16 (the default), int8, or int4. Quantized keys and values use one "
                           "fp16 scale per 32 values (changes the output; Flash Next on CUDA only)")
    cuda.add_argument("--prefill-fp8", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                      help="prompt matmuls take FP8 (e4m3) activations, one scale a row, where the checkpoint has an "
                           "FP8 prompt kernel (Qwen3.8 27B and Qwen3.6 MLX 4-bit, NVFP4 checkpoints' FP8 and MXFP8 "
                           "layers): faster prompts, lower precision (e4m3 keeps 3 mantissa bits, bf16 keeps 7; "
                           "docs/recipes/cuda.md#prompt-precision has the measured cost). Default: "
                           f"{'FP8' if FP8_BY_DEFAULT else 'bf16'} activations. Replies equal this server's own serial "
                           "decoding either way")
    cuda.add_argument("--precision", choices=("checkpoint", "full"), default=argparse.SUPPRESS,
                      help="the math for checkpoints that name their activations' formats (NVFP4): checkpoint, the "
                           "default, runs their own math as their runtimes do (FP4 x FP4 in NVFP4 layers on SM 12.x "
                           "GPUs, FP8 x FP8 in FP8 layers from SM 8.9, under the checkpoint's static input scales; "
                           "layers a GPU has no mma for run W4A16, and the startup line says which); full runs bf16 "
                           "activations against the stored weights exactly. The weights never change, only the math; "
                           "MLX-format checkpoints have one math. Replies equal this server's own serial decoding "
                           "either way")
    serve.set_defaults(func=handlers["serve"])

    pull = commands.add_parser("pull", help="download models (or draft models) from Hugging Face")
    pull.add_argument("repos", nargs="+", help="repo ids, e.g. TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    pull.set_defaults(func=handlers["pull"])

    models = commands.add_parser("models", help="list the model families and the checkpoints they are tested with")
    models.set_defaults(func=handlers["models"])

    update = commands.add_parser("update", help="install the newest TensorFold release from GitHub")
    update.add_argument("--check", action="store_true", help="only say whether a newer release exists")
    update.add_argument("--force", action="store_true", help="reinstall the newest release even when it is current")
    update.set_defaults(func=handlers["update"])

    info = commands.add_parser("info", help="show which family serves a model (reads its config.json only)")
    info.add_argument("model", help="a Hugging Face repo id or a model directory")
    info.set_defaults(func=handlers["info"])
    return parser
