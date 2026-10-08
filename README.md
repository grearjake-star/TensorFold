# SparkFold

SparkFold is a maintained fork of [TensorFold](https://github.com/ashhart/TensorFold) for the NVIDIA DGX Spark (GB10),
on the `spark-exl3` branch. It serves Qwen3.8 Flash Next from EXL3 packs first, and it runs on CUDA only. The package
and command keep TensorFold's names (`tensorfold serve ...`), so upstream's docs and flags apply.

**Why SparkFold exists.** Upstream froze its Python engine at 0.6.5 and moved new work to a Zig engine
([#286](https://github.com/ashhart/TensorFold/issues/286)). That engine targets Apple Silicon first. It serves Flash Next on
Metal today, its CUDA path runs Nemotron, and it has no EXL3 reader yet. GB10 owners who serve Flash Next from EXL3
packs still run the Python engine. SparkFold keeps that CUDA Python engine alive for them: it carries the GB10 work
below on top of 0.6.5 and drops the Apple-only parts. Every fix here is welcome back upstream at any time, in
whatever form suits the project.

Thank you to ashhart and every TensorFold contributor. The engine, the kernels, the exactness contract and the test
suite SparkFold stands on are their work. Several changes here began as pull requests to TensorFold; ashhart read
each one carefully and closed them kindly when the Python engine froze.

```bash
# inside NVIDIA's PyTorch container (CUDA, PyTorch, Triton and the extension compiler come from it)
python -m pip install "git+https://github.com/grearjake-star/TensorFold.git@spark-exl3"
tensorfold serve /path/to/Qwen3.8-Flash-Next-exl3 --parallel 8 --decode-share 0.25 --context 262144
```

The API, flags and recipes are upstream's, less the Apple Silicon parts: see the [runbook](RUNBOOK.md), the
[API reference](docs/api.md) and the [recipe book](docs/recipes/README.md).

## Receipts

All numbers below come from one DGX Spark (GB10, 128 GB) serving Qwen3.8-Flash-Next EXL3 4.05 bpw (turboderp
`4.05bpw_h6_ng6`) with a trained MTP head for 4.05, `--parallel 8 --context 262144`. The first table and the board
comparison were measured 2026-10-06 on the build this branch was cut from (the same CUDA engine before the Apple parts
were removed); SparkFold's own gate, on this branch, follows them.

| Check | Result |
| --- | --- |
| Exactness | Drafted == serial on every gate run. Each stream under `--parallel` equals its solo reply at c2/c4/c8, and the lone stream equals the single-stream reference (16/16). |
| Decode, gate (256 tokens, T 0.7) | Lone 63.0 tok/s; c2 84.9; c4 110.6; c8 about 131-138 aggregate |
| TTFT, 6K prompt, cold | 2.94 s |
| A 24K prompt arriving beside two decoding streams | First token in 19.3 s, while the streams' longest gap is 1.0 s |
| 128K needle | Recalled 3/3, TTFT 68.3 s |
| Not run | c=16 (this build serves 8 seats), TP=2, vision |

**SparkFold's own quick gate, 2026-10-07** (spark-exl3 at 6431c1b, same box and settings): 14 of 14 checks passed;
450 GPU tests passed; drafted == serial and every stream == its solo reply (c2/c4/c8 16/16); lone 63.4 tok/s, c8 145.2
aggregate; a 24K prompt arriving beside two decoding streams: longest gap 1.12 s, its first token in 19.3 s; 6K TTFT
2.97 s; 128K needle 3/3.

**Against a public DGX Spark board, 2026-10-06** (aggregate tok/s, thinking off, one full answer at c=1 and kept-busy
streams at c>=2; the other columns are [My LLM Box](https://myllmbox.com)'s own published runs on one DGX Spark):

| Prompt, streams | SparkFold (EXL3 4.05, all 10 experts, exact) | mbx v5.2 (INT4 A5B: 5 of 10 experts) | mbx v5.1 (hibrid48, all experts) | TensorFold 0.6.1 stock (their run) |
| --- | --- | --- | --- | --- |
| mixed c=1 | **69.4** | 64.7 | 55.6 | 58.6 |
| JSON c=1 | **110.3** | 85.0 | 72.2 | 81.4 |
| prose c=1 | **56.1** | 49.8 | 43.8 | 49.3 |
| mixed c=2 | **130.0** | 111.0 | 84.6 | 85.7 |
| mixed c=4 | **176.4** | 145.5 | 119.7 | 116.0 |
| mixed c=8 | 217.8 | **225.1** | 192.1 | 165.9 |
| JSON c=8 | 280.5 | **282.3** | 228.6 | 195.9 |
| prose c=8 | 142.8 | **187.4** | 152.1 | 133.0 |
| c=16 | not run (8 seats) | 338.8 (mixed) | 282.7 | 230.2 |

How to read it: their prompts ship inside their image and are not public, so ours are equivalents of the same three
kinds (code with explanation, a JSON array, a 7,000-word story), at temperature 0.7, top-p 0.8, top-k 20. Our c>=2 cells
ran 120 s rather than their 300 s, one run per cell, on a server that was also in use. Their v5.2 runs half the experts per token, a
different and cheaper model. We lead at 1, 2 and 4 streams, are even at 8 on mixed and JSON, and trail on long prose at
8 streams (-24%). They scale to 16 seats; this build serves 8.

**Settings used** (environment, on top of the flags above): `TF_NGRAM_LOCK=27`, `TF_SOLO_ROWS=65536`, `TF_KEEP=16`,
`TF_FIRST_PASS=1024`, `TF_PASS_MIN=1024`, `TF_WARM_REPLAY=4`, `TF_JOINT_PRICE=1`, `TENSORFOLD_PREFILL_ROWS=4096`,
`TF_EXL3_MOE_WINDOW=4096`, `TF_EXL3_HC_FP8=1`, a measured `TF_VERIFY_MS`/`TF_DRAFT_MS` table, and a trained MTP draft head
(`TF_MTP_HEAD`). Two of these matter for reproducing the numbers:
- `TF_EXL3_HC_FP8=1` reads the hyper-connection faces from MXFP8 copies. It changes output bits (it passed a fast-KL
  check against the default); drafted == serial still holds with it on. Leave it unset for upstream's arithmetic.
- The trained draft head is not shipped. Without `TF_MTP_HEAD` the engine drafts with the checkpoint's own MTP head,
  which is exact too but accepts fewer drafts, so expect lower decode numbers than the tables above.

## What this fork adds

Every item is exact (replies equal serial decoding) unless marked. Knobs are environment variables; most default on.

- **Kept prompt states per slot** (`TF_KEEP`, default 16): a resumed conversation continues from its own kept end,
  and each slot keeps its own lone-stream CUDA graphs, so a returning chat never pays a recapture.
- **System-block checkpoint** (`TF_SYS_CHECKPOINT`, 2048-row pieces): a shared system prompt is checkpointed where the
  first message starts, so a new chat with the same tools and system text skips it (TTFT 18.7 s -> 1.58 s on a 13K
  block).
- **Warm starts** (`TF_WARM_STARTS`, `TF_WARM_REPLAY`): kept system blocks are recorded (file mode 0600) and prefilled
  again after a restart, before serving.
- **N-gram table pin and read-ahead** (`TF_NGRAM_LOCK=<GiB>`, `TF_NGRAM_AHEAD`): pins a fixed budget of Flash Next's
  host n-gram table in whole runs, re-pins when stream caches shrink, and reads the table ahead for long prompts
  (cold 24K TTFT -55%).
- **Decode read-ahead** (`TF_DECODE_AHEAD`, default on): decode rounds ask for the next window's n-gram pages while
  the draft chain runs (cold table: lone +10%, c8 +7%).
- **Batched stage gather** (`TF_STAGE_BATCH`, default on): one n-gram gather per layer for every window in a round.
- **Joint draft pricing** (`TF_JOINT_PRICE`): `--parallel` rounds choose every stream's draft depth together from a
  measured verify-cost table, instead of each stream alone.
- **Expected-time draft stop** (`TF_DRAFT_COST`, `TF_VERIFY_MS`, `TF_DRAFT_MS`): a chain stops drafting when another
  draft no longer pays for its verify time.
- **Trained MTP draft heads** (`TF_MTP_HEAD`, `TF_MTP_HEAD_LONG`, `TF_HEAD_SWITCH_ROWS`, `TF_DRAFT_VOCAB`): load a
  retrained draft head and a draft vocabulary; `tools/draft_head` holds the offline training pipeline. No head weights
  are shipped.
- **Shared-round CUDA graphs** (`TF_MULTI_GRAPHS`): `--parallel` rounds and their draft steps replay graphs, with
  counters for replays, eager rounds and captures in the done line.
- **Host prep off the critical path** (`TF_MULTI_HOST_AHEAD`, default on): shared-round tables built from numpy into
  double-buffered pinned staging.
- **4096-row prompt pieces** (`TENSORFOLD_PREFILL_ROWS=4096`, `TF_EXL3_MOE_WINDOW=4096`): EXL3 prompt passes in wider
  routed windows.
- **EXL3 prompt kernels**: a prompt expert kernel with routed windows (#212) and jschmied's two-chunk trellis ring for
  gate|up below 4 bits (#283), fused fp16 prompt matmuls, the shared expert's
  prompt instance on a side stream (`TF_EXL3_PROMPT_SIDE`), tall-tile linear windows of 17-128 rows
  (`TF_EXL3_TALL`), HC read-out fusion and a QSA prompt indexer.
- **Fused MoE routing** (`TF_MOE_ROUTE_FUSED`, default on): top-k, weights and grouping in one block.
- **Prompt passes converge with `--parallel`**, the first pass at 256 rows; `TF_FIRST_PASS`, `TF_PASS_MIN`,
  `TF_SOLO_ROWS` tune live pass sizes and pre-size the lone-stream graph slot (all checked at start-up).
- **Prefix keep policy for tool-call sessions**: `--parallel` keeps a shared system block's state through long tool-call
  sessions, and eviction keeps every prompt's own end.
- **Slot choice without a free slot**: a fork claims the idle slot that loses the fewest kept tokens (from
  philip-pentatonic's #315), and with `TF_FRESH_SLACK=<tokens>` (opt-in, off by
  default) a fresh request skips idle slots that would cost more than that many tokens beyond the cheapest, so an
  unrelated task arriving between two long conversations' turns does not evict one of them (scottleimroth's report on
  #315). Off, a fresh request takes the oldest idle slot, as on the build the receipts were measured on.
- **Metrics** from other contributors' open pull requests: a TPOT histogram (juliankang4, #367) and how each request
  ended (cshintov, #343).
- **HC faces from MXFP8 copies** (`TF_EXL3_HC_FP8=1`, off by default): changes output bits; see above.

Removed from upstream 0.6.5: the MLX backend and its lane engine, the Metal kernels, DFlash drafting on MLX, SSD
expert streaming, the macOS launchd control plane, and the MLX-only families (DeepSeek-V4-Flash, Gemma 4, Ternary
Bonsai 2). The other CUDA families (Qwen3.8-27B, Qwen3.6-35B-A3B, Nemotron 3.5, GLM-5.3-Flash on two ranks, and Flash
Next from MLX 4-bit and NVFP4 checkpoints) are kept as upstream left them; only Flash Next from EXL3 is tested on every
change here. [CHANGELOG.md](CHANGELOG.md) lists each release.

Issues and receipts from other Sparks are welcome. There is no promise of response times.

## Credits

TensorFold is by ashhart and the TensorFold contributors ([upstream](https://github.com/ashhart/TensorFold),
Apache-2.0). This fork keeps upstream's [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md); NOTICE records that it is a modified version. Carried contributions:
philip-pentatonic (#315), BHCC2025 (#260, tall tiles), jschmied (#283's trellis ring), juliankang4 (#367), cshintov
(#343); scottleimroth's report on #315 shaped the fresh-request slot choice. EXL3 is turboderp's
format ([ExLlamaV3](https://github.com/turboderp-org/exllamav3)); the Flash Next EXL3 packs are turboderp's.

---

The rest of this page is upstream's README, trimmed to the CUDA engine.

## Models

Flash Next from EXL3 is what this fork tests on every change. The other rows are upstream's CUDA families as 0.6.5
left them.

| Model | Checkpoint | Backend | Drafting |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning | `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | CUDA | Included MTP head |
| Qwen3.8-27B | `TensorFold/Qwen3.8-27B-MLX-4bit` | CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` | CUDA | Included MTP head and context copies |
| GLM-5.3-Flash | `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` | CUDA, two ranks | MTP; optional DFlash2 |
| Qwen3.8-27B (NVFP4) | `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt: NVFP4 MLP, FP8 attention) | CUDA, one GPU | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8-27B (EXL3, experimental) | `turboderp/Qwen3.8-27B-exl3` (branches `3.00bpw`, `4.00bpw`; any codebook, 1 to 8 bits per weight) | CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (EXL3, experimental) | `turboderp/Qwen3.8-Flash-Next-exl3` (branch `3.05bpw_h5_ng5`; any codebook, a width per tensor) | CUDA | Included MTP head and context copies |
| Qwen3.8 Flash Next (NVFP4) | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (ModelOpt: NVFP4 experts, MXFP8 attention and DeltaNet); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (bf16 besides the experts) | CUDA, one GPU | Included MTP head and context copies |

`tensorfold models` lists families and checkpoints. `tensorfold info MODEL` checks configuration without
fetching weights. `serve` downloads a missing checkpoint; `pull` downloads it ahead of time.

Flash Next reads MLX affine 4-bit/group-32 checkpoints, NVFP4 exports and EXL3 packs (any codebook, a width per
tensor); without an MTP head pass `--no-drafts`. Qwen3.8-27B reads MLX affine 2- to 8-bit checkpoints in groups of
32, 64 and 128, EXL3 and NVFP4; pull DFlash2 before serving or pass `--no-drafts`. Nemotron requires 4-bit/group-64
weights and an MTP head unless `--no-drafts` is set. GLM-5.3-Flash needs two ranks and reads MLX 4-bit/group-64
weights or the experimental EXL3/TR3 checkpoint; its optional `incoai/GLM-5.3-Flash-DFlash2` has non-commercial
license terms ([third-party notices](THIRD_PARTY_NOTICES.md)). See the [recipes](docs/recipes/README.md).

## Exact decoding

A draft is accepted only when it equals the token the same engine would produce serially.
Sampling depends on the prompt or explicit seed, absolute position and token ID. Verify kernels keep each
row's arithmetic independent of the other rows in the call. Compare a request with the same request using
`"draft": false` to check drafted versus serial output.

Each stream keeps its own state and sampling key, with concurrent output required to match its solo output.
`--parallel N` with N greater than one enables shared rounds for Qwen3.8-27B and Flash Next on one or two ranks and for Qwen3.6-35B-A3B on
one rank. GLM and Nemotron CUDA serve one request at a time; CUDA `--parallel auto` also means one request at a
time.

Exactness is against the same engine, weights, runtime and settings. It does not imply identical output
between different quantizations or different tensor-parallel rank counts.

## Serve options

| Option | Meaning |
| --- | --- |
| `--host`, `--port` | Listen address, default `127.0.0.1:8080` |
| `--name` | Model ID advertised to clients |
| `--vision` | Opt-in image input: dense Qwen, and Flash Next with `--parallel` 2 or more |
| `--vision-max-images N` | With `--vision`, images across the full request history (default 4); other image limits still apply |
| `--vision-image-tokens N` | With `--vision`, the visual tokens a request's images share (default 4,096, up to 65,536); each image keeps at most 4,096 |
| `--alias` | Additional model IDs |
| `--context N` | Prompt plus reply capacity |
| `--max-tokens N` | Default reply limit, 4096 |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | Sampling defaults; temperature zero is greedy |
| `--thinking`, `--no-thinking` | Template thinking toggle |
| `--reasoning-effort` | Template effort when a request sets none; default: the template's own |
| `--thinking-budget N` | Token-count limit inside reasoning |
| `--backend auto`, `cuda` | Kept for scripts: this build serves on CUDA only |
| `--parallel N` | `auto` is 1; an explicit N enables shared rounds where the family supports them |
| `--no-drafts` | Decode serially |
| `--drafter auto`, `none`, or model ID | Select an optional draft model where the family supports it |
| `--mtp-drafts N` | Family-specific cap on MTP drafts |
| `--kv-dtype bf16`, `int8`, `int4` | Flash Next: `int8` or `int4` stores keys and values with one fp16 scale per 32 values. Other families refuse it |
| `--mtp-confidence P` | Flash Next: stop a draft chain before a later draft under this probability, 0 to 1 (default 0.70) |
| `--prefill-fp8` | Prompt matmuls take FP8 (e4m3) activations, one scale a row, where the checkpoint has an FP8 prompt kernel (Qwen3.8 27B and Qwen3.6 MLX 4-bit, FP8 and MXFP8 layers of NVFP4 checkpoints): faster prompts at lower precision ([measured](docs/recipes/cuda.md#prompt-precision)). Default: bf16 activations, as decode |
| `--precision checkpoint`, `full` | NVFP4 checkpoints: `checkpoint` (default) runs their own math, FP4 x FP4 on SM 12.x and FP8 x FP8 from 8.9, W4A16 elsewhere; `full` runs bf16 activations against the stored weights ([measured](docs/recipes/cuda.md#nvfp4-precision)) |
| `--tp 2 --rank R --master HOST` | Two-rank CUDA execution; `--master-port P` sets rank 0's rendezvous port (default 29551) |
| `--decode-share F` | Flash Next with `--parallel N`: replies decode inside each prompt pass, and the share sizes the passes so a round's decoding takes it (default 0: whole passes) |
| `--checkpoint-slots N` | The prompt states Qwen3.8-27B keeps under `--parallel` 2 or more (default 3) |
| `--no-update-check` | Disable the startup release check |

The default sampling settings come from `generation_config.json`. Requests can override sampling and reply
length. See [API fields](docs/api.md) for request scope.

In a terminal, `tensorfold serve` keeps one live throughput line under its log; it is off when output is
redirected, and `TENSORFOLD_NO_LIVE=1` turns it off.

<a id="memory"></a>

## Context and memory

Qwen defaults to the affordable native capacity. GLM targets a dense 2,051-token window, and Nemotron targets 16,384
tokens; the capacity estimate can lower these defaults. Flash Next's `--kv-dtype int8` or `int4` counts its smaller
cache, so the same memory admits a longer window. `--context 0` targets the affordable native capacity for every
family. A positive value must fit both the native window and the capacity estimate on every rank; otherwise startup
refuses it with fitting guidance. The CUDA budget grants a GPU its free memory less a floor of a tenth of that memory,
at least 4 GiB: a discrete card's own memory, or the host's available memory on a unified GPU such as GB10.
`TENSORFOLD_MEMORY_RESERVE_GIB` moves the floor (at least 2 GiB), and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps the grant
from above in GiB. A smaller floor can end requests with CUDA errors mid-reply. CUDA checks context before opening a
stream.

## Prompt caching

CUDA engines keep their own prompt and reply states. On Flash Next this fork keeps per-slot prompt ends (`TF_KEEP`),
a system-block checkpoint and, with `TF_WARM_STARTS`, re-prefills recorded system blocks after a restart; see
"What this fork adds" above.

## Image input

Install the vision extra, `python -m pip install 'tensorfold[vision] @ git+https://github.com/grearjake-star/TensorFold.git@spark-exl3'`,
and start a supported Qwen3.5/3.8 dense checkpoint with `--vision`; Flash Next also accepts images with
`--vision --parallel 2` or more. See [image input](docs/vision.md). Vision is not part of this fork's receipts.

## NVIDIA GPUs

Use NVIDIA's PyTorch container for CUDA, PyTorch, Triton and the extension compiler; the package has no
`cuda` installation extra. Install TensorFold inside the container without replacing that toolchain.

```bash
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install "git+https://github.com/grearjake-star/TensorFold.git@spark-exl3"
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --host 0.0.0.0
```

Qwen3.8-27B, Flash Next and Nemotron support one or two CUDA ranks; GLM requires two.
For two ranks, see the [CUDA runbook](RUNBOOK.md#nvidia-gpus). Each rank needs its checkpoint and any
optional drafter. Rank 0 serves HTTP. Unified GPU/host memory also holds runtime buffers and file-backed
model data; the startup estimate is not a measured maximum capacity.

## Updating

This fork's `tensorfold update` checks the fork's releases on GitHub (grearjake-star/TensorFold), never upstream's, so it
cannot replace this build with an upstream release. SparkFold versions name the TensorFold release they are based
on and a SparkFold revision: the package is `0.6.5+spark.1`, its release tag `v0.6.5-spark.1`, which update checks
order after 0.6.5 and before a release on a later base. Before the fork has published a release, `tensorfold update`
says so and changes nothing. `--no-update-check` or `TENSORFOLD_NO_UPDATE_CHECK=1` disables startup checks. [CHANGELOG.md](CHANGELOG.md) lists every release of the fork, then upstream's history.

## Development and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request. Family interfaces and verification requirements
are in the [recipe book](docs/recipes/README.md) and the [family map](src/tensorfold/families/README.md).
Apache-2.0 from upstream 0.6.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) (which records that this is a modified
version) and [third-party notices](THIRD_PARTY_NOTICES.md). Upstream releases up to 0.5.0 were MIT, and code written
before 0.6.0 keeps its [MIT notice](LICENSES/MIT.txt). Model weights keep their own licenses.
