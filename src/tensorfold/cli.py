"""Load local or Hugging Face models and serve them through their family kernels at an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from tensorfold import cli_args
from tensorfold.server import stacks, thinking_notes
from tensorfold.serve_options import check as _check_serve_options, vision_options as _vision_options

COMMANDS = ("serve", "pull", "models", "info", "update", "service", "tui")


def build_parser() -> argparse.ArgumentParser:
    """The ``tensorfold`` parser with this module's subcommand handlers."""

    return cli_args.build_parser({"serve": cmd_serve, "pull": cmd_pull, "models": cmd_models,
                                  "update": cmd_update, "info": cmd_info})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``tensorfold MODEL ...`` is ``tensorfold serve MODEL ...``
    if argv and not argv[0].startswith("-") and argv[0] not in COMMANDS:
        from tensorfold import hub

        if Path(argv[0]).expanduser().is_dir() or hub.is_repo_id(argv[0]):
            argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError) as exc:
        print(f"tensorfold: {exc}", file=sys.stderr)
        return 1


def _config_dir(model: str) -> Path:
    """Resolve config.json without downloading weights so family compatibility checks run first."""

    from tensorfold import hub

    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if not hub.is_repo_id(model):
        raise FileNotFoundError(f"{model} is neither a directory nor a Hugging Face repo id (owner/name)")
    found = hub.cached(model)
    if found is not None and (found / "config.json").is_file():
        return found
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, "config.json")).parent


def cmd_pull(args: argparse.Namespace) -> int:
    from tensorfold import families, hub

    for repo in args.repos:
        if not hub.is_repo_id(repo):
            raise ValueError(f"{repo} is not a Hugging Face repo id (owner/name)")
        config = _config_dir(repo)
        try:
            family = families.detect(config)
        except ValueError:
            family = None          # a draft model, for example
        if family is not None:
            settings = families.read_config(config)
            readable = [b for b in families.backends_of(family)
                        if families.quant_method(settings) in families.readable_quants(family, b)]
            if not readable:
                families.require_readable(family, settings, families.backends_of(family)[0])
            _note_untested(family, repo)
            check = getattr(family.package, "check", None)
            if check is not None:
                check(config)
        path = hub.pull(repo)
        required_files = getattr(family.package, "REQUIRED_FILES", {}).get(repo, ()) if family is not None else ()
        if required_files and not hub._cached_weights_complete(path, required_files=required_files):
            raise FileNotFoundError(f"{repo} is missing required files: {', '.join(required_files)}")
        what = f"{family.title} ({family.model_type})" if family is not None else "no model family (a draft model?)"
        print(f"{repo}: {hub.size_of(path) / 1e9:.1f} GB in {path} [{what}]")
        if required_files:
            print(f"[tensorfold] required model files ready: {', '.join(required_files)}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    from tensorfold import update

    return update.update(check_only=bool(args.check), force=bool(args.force))


def cmd_models(args: argparse.Namespace) -> int:
    from tensorfold import families

    for kind, family in sorted(families.families().items()):
        package = family.package
        print(f"{family.title} ({kind}; {_engines(family)})")
        for repo in getattr(package, "MODELS", ()):
            print(f"  model    {repo}")
        drafter = getattr(package, "DRAFTER", "")
        if drafter:
            print(f"  drafter  {drafter}")
    return 0


def _engines(family: Any) -> str:
    """Which backend serves a family: the CUDA engine."""

    return "CUDA engine" if hasattr(family.package, "cuda_engine") else "no engine"


def cmd_info(args: argparse.Namespace) -> int:
    from tensorfold import families

    directory = _config_dir(args.model)
    config = families.read_config(directory)
    text = config.get("text_config", config)
    family = families.detect(directory)
    print(f"model_type   {family.model_type}")
    print(f"family       {family.title} ({family.module})")
    print(f"engine       {_engines(family)}")
    for key in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "n_routed_experts",
                "vocab_size", "max_position_embeddings"):
        if key in text:
            print(f"{key:12s} {text[key]}" if len(key) <= 12 else f"{key} {text[key]}")
    print(f"quantization {families.describe_quantization(config)}")
    bits = getattr(family.package, "CUDA_AFFINE_BITS", ())
    groups = getattr(family.package, "CUDA_AFFINE_GROUPS", ())
    if bits and groups:
        print(f"CUDA formats affine {'/'.join(map(str, bits))}-bit, groups {'/'.join(map(str, groups))}")
    readers = [b for b in families.backends_of(family)
               if families.quant_method(config) in families.readable_quants(family, b)]
    if readers:
        print("runs on      NVIDIA GPUs (CUDA)")
    else:
        print(f"runs on      not yet: no {family.title} engine reads these weights. {families.OWN_MODEL_HELP}")
    generation = _generation_config(directory)
    if generation:
        print(f"sampling     {generation}")
    check = getattr(family.package, "check", None)
    if check is not None:
        check(directory)
    return 0


def _generation_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "generation_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    sampling = {k: config[k] for k in ("temperature", "top_k", "top_p", "min_p") if config.get(k) is not None}
    if config.get("do_sample") is False:
        sampling["temperature"] = 0.0
    elif config.get("do_sample") is True and "temperature" not in sampling:
        sampling["temperature"] = 1.0
    return sampling


def _model_context(model_dir: Path) -> int:
    from tensorfold.families import read_config

    config = read_config(model_dir)
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


def _drafter(family: Any, choice: str, backend: str = "cuda") -> str:
    """The draft model directory for ``--drafter`` (auto: the family's draft model if it has been pulled)."""

    from tensorfold import hub

    if choice in ("", "none"):
        return ""
    if choice != "auto":
        return str(hub.resolve(choice))
    # a family may name another draft model for CUDA (CUDA_DRAFTER); "" drafts with its own MTP layer there
    repo = getattr(family.package, "CUDA_DRAFTER" if backend == "cuda" else "DRAFTER",
                   getattr(family.package, "DRAFTER", ""))
    if not repo:
        return ""
    found = hub.cached(repo)
    if found is None or not hub._cached_weights_complete(found):
        print(f"[tensorfold] no draft model: `tensorfold pull {repo}` once to draft with it", flush=True)
        return ""
    return str(found)


def _note_untested(family: Any, model: str) -> None:
    """Explain that an unlisted Hugging Face checkpoint runs when its storage format matches the family kernels."""

    from tensorfold import families, hub

    tested = tuple(getattr(family.package, "MODELS", ())) + tuple(filter(None, [getattr(family.package, "DRAFTER", "")]))
    if hub.is_repo_id(model) and model not in tested:
        print(f"[tensorfold] note: {model} is not a checkpoint TensorFold is tested with ({', '.join(tested) or 'none'}). "
              f"It runs when its format matches what the {family.title} kernels read: replies stay exact to serial "
              f"decoding, speed and quality are unmeasured. {families.OWN_MODEL_HELP}", flush=True)


def _backend(choice: str, family: Any) -> str:
    """Always cuda: this fork serves on NVIDIA GPUs only (``--backend auto`` and ``cuda`` both pick it)."""

    if choice not in ("auto", "cuda"):
        raise ValueError(f"--backend {choice}: this build serves on NVIDIA GPUs (CUDA) only")
    if not hasattr(family.package, "cuda_engine"):
        raise ValueError(f"{family.title} has no CUDA engine")
    return "cuda"


def _serve_cuda(args: argparse.Namespace, family: Any, model_dir: Path, context: int | None = None) -> int:
    """Serve with the family's CUDA engine (``cuda_engine``) behind ``tensorfold.cuda.server``."""

    from tensorfold import hub

    if args.tp == 2 and not args.master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    if args.tp == 1 and args.rank != 0:
        raise ValueError("--rank 1 needs --tp 2")
    started = time.perf_counter()
    drafter = "" if args.no_drafts else _drafter(family, args.drafter, "cuda")
    options: dict[str, Any] = {"drafter": drafter, "tp": int(args.tp), "rank": int(args.rank), "master": args.master,
                               "master_port": int(args.master_port), "no_drafts": bool(args.no_drafts)}
    if getattr(args, "kv_dtype", "bf16") != "bf16":
        options["kv_dtype"] = args.kv_dtype
    options.update(_vision_options(args))
    if args.mtp_drafts is not None:
        options["mtp_drafts"] = int(args.mtp_drafts)
    if args.ple_on_ssd:
        options["ple_on_ssd"] = True
    if getattr(args, "mtp_confidence", None) is not None:
        options["mtp_confidence"] = float(args.mtp_confidence)
    if getattr(args, "decode_share", None) is not None:
        options["decode_share"] = float(args.decode_share)
    options["context"] = context if context is not None else args.context
    options["context_explicit"] = args.context is not None
    streams = 1 if str(args.parallel).strip().lower() == "auto" else _parallel(args.parallel)
    if streams > 1:
        options["parallel"] = streams
    if getattr(args, "checkpoint_slots", None) is not None and getattr(family.package, "CUDA_CHECKPOINT_SLOTS", False):
        options["checkpoint_slots"] = int(args.checkpoint_slots)
    served = args.name or (args.model.rstrip("/").split("/")[-1] if hub.is_repo_id(args.model) else model_dir.name)
    where = f", rank {args.rank} of 2" if args.tp == 2 else ""
    print(f"[tensorfold] loading {served}: {family.title} ({family.model_type}) on CUDA{where}", flush=True)
    from tensorfold.cuda import precision, prompt_precision

    asked = getattr(args, "prefill_fp8", None)
    prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT if asked is None else asked)   # before any weight loads
    chosen = getattr(args, "precision", None)
    precision.set_mode(chosen or precision.CHECKPOINT, asked=chosen is not None)
    engine = family.package.cuda_engine(model_dir, **options)
    weights = getattr(engine, "w", None)
    fp8 = prompt_precision.fp8() and bool(getattr(weights, "fast_prefill", False))
    if asked and not fp8 and getattr(weights, "precision", "full") == precision.CHECKPOINT:
        raise ValueError("--prefill-fp8 is for --precision full: the checkpoint's own math already runs its prompts in "
                         "FP4 and FP8")
    if asked and not fp8:
        raise ValueError("--prefill-fp8: this checkpoint's prompt matmuls have no FP8 kernel (EXL3 packs, MLX formats "
                         "other than Qwen's 4-bit g64, Flash Next without MXFP8 layers); drop the flag")
    stacks.arm()            # its warmup may have loaded a compiler that took USR1
    if args.tp == 2 and args.rank == 1:
        print(f"[tensorfold] rank 1 ready in {time.perf_counter() - started:.1f}s, following rank 0", flush=True)
        engine.follow()
        return 0
    from tensorfold.cuda.server import App, serve

    sampling = _generation_config(model_dir)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k),
                       ("min_p", args.min_p)):
        if value is not None:
            sampling[key] = value
    app_class = getattr(family.package, "CUDA_APP", None) or App
    app = app_class(engine, model_dir, served, default_thinking=bool(args.thinking), sampling=sampling,
                    max_tokens=int(args.max_tokens), context_window=context if context is not None else args.context,
                    reasoning_effort=args.reasoning_effort, thinking_budget=int(args.thinking_budget),
                    vision_max_images=getattr(args, "vision_max_images", None),
                    **({"vision_image_tokens": args.vision_image_tokens}
                       if getattr(args, "vision_image_tokens", None) is not None else {}),
                    aliases=list(args.alias))
    shown = "greedy" if float(sampling.get("temperature", 1.0)) <= 0 else ", ".join(
        f"{k} {v}" for k, v in sampling.items())
    effective_context = app.effective_context_window
    own = getattr(weights, "precision", "") == precision.CHECKPOINT
    prompts = "FP8 activations" if fp8 else "the checkpoint math" if own else "bf16 activations"
    print(f"[tensorfold] serving {served} at http://{args.host}:{args.port}/v1 on CUDA{where} "
          f"(sampling: {shown}; drafts: {'off' if args.no_drafts else 'on'}; prompts: {prompts}; "
          f"context: {'unlimited' if effective_context is None else effective_context}; "
          f"loaded in {time.perf_counter() - started:.1f}s)", flush=True)
    app.auth = getattr(args, "auth", None)
    note = thinking_notes.startup(model_dir, bool(args.thinking))
    if note:
        print(note, flush=True)
    serve(app, args.host, int(args.port))
    return 0


def _parallel(value: Any) -> int:
    """``--parallel``: "auto" is up to 8 requests at once; a number caps it."""

    if str(value).strip().lower() == "auto":
        return 8
    try:
        return max(1, int(value))
    except ValueError:
        raise SystemExit(f"--parallel takes a number or auto, not {value!r}") from None


def cmd_serve(args: argparse.Namespace) -> int:
    from tensorfold.server.authentication import configure

    args.auth = configure(args)
    from tensorfold import families, hub

    if not args.no_update_check:
        from tensorfold import update

        update.check_in_background()
        news = update.first_run_notice()
        if news:
            print(news, flush=True)
    config_dir = _config_dir(args.model)
    family = families.detect(config_dir)
    if args.ple_on_ssd and not hasattr(family.package, "ple_bytes"):
        raise ValueError(f"--ple-on-ssd: {family.title} has no n-gram (PLE) tables to read from SSD")
    backend = _backend(args.backend, family)
    _check_serve_options(args, family, backend, config_dir)
    families.require_readable(family, families.read_config(config_dir), backend)
    _note_untested(family, args.model)
    required_files = getattr(family.package, "REQUIRED_FILES", {}).get(args.model, ())
    native_context = _model_context(config_dir)
    context = native_context if args.context is None else int(args.context)
    if context < 0:
        raise ValueError("--context must be 0 or a positive token count")
    if native_context and context > native_context:
        raise ValueError(f"--context {context} exceeds this model's {native_context}-token window")
    check = getattr(family.package, "check", None)
    if check is not None:
        check(config_dir)                        # refuse an unsupported checkpoint before downloading its weights
    needs_full_snapshot = hub.is_repo_id(args.model) and not hub._cached_weights_complete(
        config_dir, required_files=required_files)
    model_dir = hub.resolve(args.model, required_files=required_files)
    if needs_full_snapshot and check is not None:
        check(model_dir)                         # checks that need the complete index, such as an MTP head

    stacks.start()          # `kill -USR1 <pid>` prints every thread's Python stack: where a silent server waits
    return _serve_cuda(args, family, model_dir, context)


if __name__ == "__main__":
    raise SystemExit(main())
