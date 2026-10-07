# Qwen3.8-27B

Experimental [packed affine formats](../quantization.md) include 8-bit and mixed layer precision.

The `qwen3_5` family combines Gated DeltaNet and full attention. The standard recipe below uses its
4-bit/group-64 checkpoint; the quantization guide lists the other affine formats it reads.

```bash
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --name bench
```

DFlash2 is used automatically once pulled. On MLX, `--drafter none` disables that draft model;
`--no-drafts` disables all drafts on either backend. CUDA requires DFlash2 unless `--no-drafts` is set.
The target verifies every proposed token against its own serial sample.

## Weights other than 4-bit

The CUDA readers cover MLX affine 2/3/4/5/6/8-bit projections with groups of 32/64/128, including mixed layers
(for example `TensorFold/Qwen3.8-27B-oQ2` and `TensorFold/Qwen3.8-27B-oQ4`). CUDA also reads
[EXL3 packs](#exl3-checkpoints-experimental). A fused projection keeps its members separate where their bit width or
group size differs. Unsupported formats and tied embedding heads are refused from `config.json` before weight
downloads and again at load.
Lower weight precision does not guarantee faster decode or a fitting context. Release memory and
quality comparisons are TBD [release-0.3.5].

## CUDA

On CUDA the 27B serves NVFP4 and EXL3 checkpoints, and the MLX 4-bit checkpoint, the only format two ranks
read. `tensorfold serve` loads the checkpoint you name; it picks
none by itself. Use the [CUDA container setup](../../RUNBOOK.md#dgx-spark) for any of them. Prompts take bf16
activations by default; what that costs against the FP8 prompt path (`--prefill-fp8`) depends on the format
([prompt precision](cuda.md#prompt-precision)):

| Checkpoint | Weights | Ranks | bf16 prompts against `--prefill-fp8` |
| --- | --- | --- | --- |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 MLP and head, FP8 attention and DeltaNet | one | not measured yet |
| `turboderp/Qwen3.8-27B-exl3` (3.00bpw) | EXL3 | one | unchanged: EXL3 prompts never took FP8 activations |
| `TensorFold/Qwen3.8-27B-MLX-4bit` | MLX affine 4-bit, groups of 64 | one or two | 0.73-0.82x from 2k to 128k |

Mia-AiLab publishes EXL3 packs of the model (`Mia-AiLab/Qwen3.8-27B-EXL3`, `Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw`,
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`) and an EXL3 DFlash2 drafter (`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`); none
has been loaded here yet.

### NVFP4 checkpoints

The CUDA engine reads NVIDIA's ModelOpt export of the model (`nvidia/Qwen3.8-27B-NVFP4`: NVFP4 MLP and head, FP8
attention and DeltaNet projections, bf16 embedding and gates) as it ships. The reader also takes
compressed-tensors NVFP4 and FP8 exports, checked on synthetic tensors only. Each projection is read by its tensors: NVFP4 codes with their e4m3 block scales and
FP8 bytes go to the device unchanged, and the lane matmuls turn them into exact bf16 operands (an e2m1 code times
its block scale fits bf16), so drafted replies equal `"draft": false` ones and prompts keep their bits in any
chunking. `--parallel` serves concurrent requests as on the MLX checkpoint, each reply equal to the same request
alone. One GPU: `--tp 2` stops at startup (two ranks read the MLX checkpoint), and so does `--vision` until image
input is qualified on this checkpoint.

```bash
tensorfold pull nvidia/Qwen3.8-27B-NVFP4 z-lab/Qwen3.8-27B-DFlash2
tensorfold serve nvidia/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8080
```

Prompts take bf16 rows on a prompt GEMM that reads the stored bytes: each FP8 byte and each NVFP4 code times its
block scale is exact in bf16, summed in fp32 over the inputs, the tensor scale last. `--prefill-fp8` restores the FP8
prompt matmul (FP8 projections as stored, NVFP4 ones staged to e4m3 once a chunk, a step that rounds by 2^-4 at most,
and e4m3 activations). Decode, measured on one DGX Spark (GB10) through `tensorfold serve` against the MLX 4-bit
checkpoint on the same engine and box, alternating (MLX, NVFP4, NVFP4, MLX), 64 tokens, five seeds, medians:

| Cell | NVFP4 | MLX 4-bit | vLLM MTP=3 (NVFP4) |
| --- | ---: | ---: | ---: |
| Code, sampled | 47.1 tok/s | 57.8 tok/s | 23.4 tok/s |
| Chat, sampled | 38.2 tok/s | 50.0 tok/s | 25.4 tok/s |
| Code, greedy | 47.2 tok/s | 53.9 tok/s | 25.8 tok/s |
| Chat, greedy | 38.3 tok/s | 50.1 tok/s | 24.7 tok/s |

The table predates a fix to the FP8 lane matmul (its pipeline stages now start on 128-byte lines), which took a
12-row verify from 103 ms to 85 against the MLX checkpoint's 84, although the attention and DeltaNet weights are 8-bit
(16.3 GB read a token against 14.4). Tokens a round match (4.0-5.2). Cold prefill ran 1,834 / 1,872 / 1,779 / 1,563 /
1,240 tok/s at 2k / 8k / 16k / 32k / 64k on the FP8 prompt path (today's `--prefill-fp8`), level with the MLX
checkpoint's. The startup estimate is 62.5 GiB at the 262,144-token window.

### EXL3 checkpoints (experimental)

The CUDA engine also reads turboderp's EXL3 packs of the model (`turboderp/Qwen3.8-27B-exl3`, a branch per size,
`mul1` codebook, 6-bit head) through the shared EXL3 module ([EXL3 weights](exl3.md)) and drafts with
`z-lab/Qwen3.8-27B-DFlash2`. One GPU: two ranks read the MLX checkpoint. Download a size by its branch, then
serve the folder:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-27B-exl3', revision='3.00bpw', local_dir='qwen27b-exl3-3.00bpw')"
tensorfold pull z-lab/Qwen3.8-27B-DFlash2
tensorfold serve qwen27b-exl3-3.00bpw --host 0.0.0.0 --port 8080
```

Verify windows use the row-invariant EXL3 linear, so drafted replies equal `"draft": false` ones; prompts use the
EXL3 prompt path (W_q decoded once a chunk, a fixed-tile GEMM), whose bits do not depend on chunking, so the
engine keeps prompt ends as it does for the MLX checkpoint. The drafter reads the target's 6-bit head over its
draft vocabulary by slicing the head's 128-column strips as stored: its logits are the target's, bit for bit.

Measured on one DGX Spark (GB10) through `tensorfold serve`, the 3.00bpw pack against the MLX 4-bit checkpoint on
the same engine and box, the [public benchmark command](README.md#measurements), medians of 15 runs a cell:

| Cell | EXL3 3.00bpw | MLX 4-bit | vLLM MTP=3 |
| --- | ---: | ---: | ---: |
| Code, sampled | 83.4 tok/s | 57.5 tok/s | 23.4 tok/s |
| Chat, sampled | 44.2 tok/s | 49.7 tok/s | 25.4 tok/s |
| Code, greedy | 64.5 tok/s | 53.6 tok/s | 25.8 tok/s |
| Chat, greedy | 39.9 tok/s | 50.0 tok/s | 24.7 tok/s |

A 12-row round costs 75 ms on the pack against 84 on the MLX checkpoint (10.1 GB of weights a token against
14.4). The chat cells keep fewer drafted tokens a round (3.0-3.3 against 4.2), since DFlash2 was trained on the
unquantized model. Cold prefill runs 880-970 tok/s from 2k to 16k and 720-890 at 32k-64k, about two thirds of the
MLX checkpoint's bf16 prompts and half its `--prefill-fp8` ones. The engine and drafter take 13.2 GiB after loading;
a 64k prompt peaks at 28 GiB allocated. Other branches of the pack load through the same path; only 3.00bpw is
measured here.

### MLX 4-bit, one or two ranks

One or two ranks are supported. Pull the model and drafter on every rank, then start rank 1 before rank 0:

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The verify matmul fixes reduction order by weight shape. Tree attention reads only committed keys and the
node's own path; recurrent commits replay that path. Two-rank reductions gather fp32 partials and add in
rank order. Each rank count has its own serial reference. See the
[CUDA kernel map](../../src/tensorfold/families/qwen3_5/cuda/README.md).

### Concurrent requests

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --parallel 16 --context 8192 --name bench
```

`--parallel N` decodes up to N requests in shared rounds: each stream verifies its own DFlash2 tree in one
forward, and every reply equals the same request served alone. A new prompt prefills 1,024 tokens a round
while the other streams decode. States kept at message starts let prompts that share a system prompt resume
there, and each stream's caches are sized once, at admission.

On one DGX Spark, with the workload from issue #38 (48 chat requests of 3,000 to 5,100 tokens sharing a
2,900-token system prompt, up to 512 tokens each, greedy, all sent at once), TensorFold serves 161.7 tok/s at
16 in flight with a 25.4 GiB peak (nvidia-smi). vLLM with `nvidia/Qwen3.8-27B-NVFP4`, MTP=3, prefix caching
and `--max-num-seqs 16` serves 132.1 tok/s at 51.7 GiB on the same Spark. At 8 and 4 in flight TensorFold
serves 147.5 and 112.0 tok/s. `tools/shared_prefix_prompts.py` builds the workload and
`tools/shared_prefix_load.py` sends it (`--concurrency`, `--mem` for the memory peak).

### Structured output

With `pip install 'tensorfold[grammar]'` (xgrammar), serving on one or two GPUs enforces `response_format` and the
`guided_*` fields: drafts the grammar rejects are cut and rows are masked by their paths' grammar before sampling, so
drafted, serial and concurrent replies stay equal. See [the API reference](../api.md#structured-output).

### Historical public-fixture results

The earlier CUDA recipe reports these decode medians in NVIDIA's `pytorch:26.07-py3` container on GB10.
They are retained as historical results, not measurements of the merged 0.3.5 release.

| Ranks | Code sampled | Chat sampled | Code greedy | Chat greedy |
| --- | ---: | ---: | ---: | ---: |
| One | 49.6 tok/s | 45.8 tok/s | 49.2 tok/s | 45.9 tok/s |
| Two | 82.4 tok/s | 58.9 tok/s | 76.2 tok/s | 71.1 tok/s |

Reproduce the workload with the checkpoint above, default drafting and the
[public benchmark command](README.md#measurements). Its fixed prompts, 64-token replies, seeds 1234
through 1238 and sampling settings define these cells. For one rank, omit the tensor-parallel flags. Record the runtime
and model revision with any new result; these historical rates are not predictions for another runtime.

## Calibration and checks

The draft calibration metadata names public prompts. When regenerating it, start its server with
`--port 8473 --name qwen27` so the collection client reaches the named endpoint. Pass the saved full
provenance object with `--source` to `tools/fit_draft_calibration.py`; retain model and drafter revisions.

Kernel tests check rows alone and in windows, tree paths and committed state. Release checks must also
compare drafted/serial, resumed/fresh and concurrent/solo requests with thinking on and off and tools.
Decode rate, prefill, concurrency and peak-memory results are TBD [release-0.3.5].
