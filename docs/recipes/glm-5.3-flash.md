# GLM-5.3-Flash

The `glm5_next` family serves `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` on two-rank CUDA.
The checkpoint uses affine 4-bit weights in groups of 64 and includes its MTP layer.
Kimi delta attention, sparse MLA and MoE blocks mix four residual streams.

## CUDA

On CUDA GLM-5.3-Flash runs on two ranks from Brandon M. Music's EXL3/TR3 checkpoint
(`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, re-hosted as `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`; experimental;
[EXL3](#exl3)) or from the MLX 4-bit checkpoint.
No NVFP4 checkpoint of it is read. `tensorfold serve` loads the checkpoint you name; it picks none by itself. Prompt
precision does not change here: neither checkpoint has an FP8 prompt kernel, so `--prefill-fp8` is refused.

Use the [two-rank container setup](../../RUNBOOK.md#nvidia-gpus) and pull the same checkpoint on both ranks:

```bash
tensorfold pull TensorFold/GLM-5.3-Flash-MLX-4bit-MTP
tensorfold serve TensorFold/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve TensorFold/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

Rank 1 starts first and rank 0 serves HTTP. Use the same context and drafting settings on both ranks.
The optional `incoai/GLM-5.3-Flash-DFlash2` model has CC BY-NC-ND 4.0 terms; pull it on both ranks only
when those terms fit the intended use. The CLI uses it automatically once it has been pulled.
Without it, the engine uses MTP drafts; `--drafter none` explicitly selects MTP-only drafting.
Give both ranks the same drafter setting. `--no-drafts` disables all drafting for the serial reference.
A checkpoint with neither an MTP head nor a supplied DFlash2 model is refused unless drafts are disabled.
`TF_GLM_MTP` decides whether the CUDA engine loads the MTP head: `1` (the default) keeps it beside DFlash2, so MTP
policies and `auto`'s per-round choice stay available; `auto` leaves it out when DFlash2 is loaded or drafts are
disabled; `0` leaves it out. Left out, it saves each rank the head's weights (about 2 GiB for this checkpoint), its
cache rows and decode buffers, and prompts skip its absorb; MTP policies (and `--mtp-drafts N`) then draft with
DFlash2. Replies are the same either way. On two GB10s with DFlash2, `auto` read prompts about 3% faster and decoded
sampled code faster, but greedy chat about 4% slower, so the head stays by default. Give both ranks the same setting.

### EXL3

Brandon M. Music created this EXL3/TR3 checkpoint (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, ShapleyMCG License 1.0,
which asks for attribution); `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` is a byte-identical re-host of it. Either ID
serves. It is an experimental CUDA checkpoint. The reader supports 4-bit
mcg-codebook routed experts with BF16 weights elsewhere, not arbitrary EXL3 layouts. Start it with the
two-rank command above, substituting its checkpoint ID on both ranks. With DFlash2 available, the EXL3
`auto` policy uses DFlash2; without it, MTP remains available.

The expert decoder and BF16 target matmul keep row arithmetic fixed. A quantized copy of the head may
propose drafts, but target verification retains the BF16 head. EXL3 speed, capacity and long-context
qualification are TBD [release-0.3.5].

### Draft policies

For the affine checkpoint with its MTP head loaded, the default `auto` policy uses MTP for sampled requests.
For greedy requests with DFlash2 loaded beside the head (`TF_GLM_MTP=1`, the default),
it compares committed tokens per estimated round time and chooses a drafter. It periodically probes
the other drafter and discards its old rate after switching away, so later probes can change the choice.
Without the head (`TF_GLM_MTP=auto` beside DFlash2, or `0`), `auto` drafts with DFlash2.
Every policy verifies against the same target, and `"draft": false` selects the serial reference.
A request can select a policy after `@` in its model ID, such as `bench@c3:0.35`, or with `tf_policy`.
`--mtp-drafts N` selects a fixed depth at startup. With DFlash2 available, `--mtp-drafts 0` selects
`fc5:0.3`; without it, zero selects the serial reference.

| Policy | Meaning |
| --- | --- |
| `auto` | Default per-request selection |
| `0` | Serial |
| `N` | Fixed number of MTP drafts |
| `a:LOW:HIGH` | MTP depth from running acceptance |
| `cN:P` | MTP chain capped at N and a probability-product threshold |
| `fN`, `fcN:P`, `fa:...` | Corresponding DFlash2 policies, requiring its checkpoint |

The default context is 2,051 tokens, where attention stays dense. A larger positive `--context` enables
sparse attention beyond that boundary if the startup memory estimate admits it on both ranks.
`--context 0` instead targets the affordable native window. An explicit reply reservation beyond the allocated window receives HTTP 400 before streaming; an omitted reply limit
is capped to the remaining space. Larger-context restart advice appears only when the estimate allows it.

Prompt prefill uses the shared CUDA prefill kernels. Decode uses CUDA graphs, past the dense limit one per
pool bucket. The engine keeps up to 8 conversations' prompts (`TF_GLM_CACHE_ENTRIES`): when another
conversation takes the attention caches, a kept prompt's rows are saved. Kept states and saved rows together get
`TF_GLM_CACHE_GIB` (default 3), or less when the window leaves less memory on either Spark; the startup log says
when it is less, and the memory estimate includes it. Earlier turns keep their reasoning in the prompt
(`clear_thinking` false, zai-org's default), so an agent's next user message resumes from its previous tool loop
instead of filling it again; `TF_GLM_CLEAR_THINKING=1` drops it, as the TR3 checkpoint's template does by default,
and a request's `chat_template_kwargs.clear_thinking` wins over either. It serves one request at a time. Both ranks
finish a started reply after a client disconnects.

DFlash2 attends only its 2,048-row sliding window: a block pass reads only the window's tiles, and the drafter keeps
its context in a ring of that window, its block and a tile (2,176 rows, 21 MiB a rank whatever the window, instead
of 10 KiB a rank for every token of the window). A kept prompt DFlash2 can resume from holds a copy of the window
(20 MiB a rank, within `TF_GLM_CACHE_GIB`). The drafts are the same bits. `TF_GLM_DRAFT_RING=0` keeps the
whole-window buffer instead (give both ranks the same value). The memory estimate counts the draft model as it is
held (4-bit copies and its selector's codebooks, 0.63 GiB a rank), not at 4 bytes a checkpoint value.

### Long contexts: the latent cache

The DSA layers are NoPE MLA: head h's key is `Wk_h c` and its value `Wv_h c`, with `c` the token's 512-wide
normalized latent and `Wk_h`, `Wv_h` blocks of `kv_b_proj`. So `score_h = (Wk_h^T q_h) . c` and
`out_h = Wv_h (sum_j p_j c_j)`. The CUDA engine caches `c` only (bf16, 1 KB a token and layer, shared by the
heads and both ranks; `TF_GLM_LATENT=0` returns to per-head keys and values), absorbs the query once per row,
attends over latents and expands the attended latent once. Past 2,051 tokens a row attends to its top 512 index
pools (2,048 tokens) plus its incomplete pool; the pools are ranked by one radix-select kernel over the pools the
chunk can see, rounded up to a power of two. Every kernel computes a row alone in an order fixed by its own
position, so a decode window's rows keep the serial steps' bits; prompt chunks give the same bits for any
chunking. The memory estimate sizes the latent cache (`mla_geometry(latent=True)`).

Measured on two DGX Sparks (GB10, 128 GB each) with the MLX 4-bit checkpoint, MTP drafts only,
`--context 262144`, a synthetic codebase with one hidden fact, cold prompts:

| Prompt | Prompt reading | First token | Decode at that depth | Hidden fact |
| --- | ---: | ---: | ---: | :---: |
| 32,770 tokens | 1,138 tok/s | 29 s | 50.9 tok/s | found |
| 131,074 tokens | 898 tok/s | 146 s | 47.2 tok/s | found |
| 261,906 tokens | 849 tok/s | 309 s | 31.9 tok/s | found |

An earlier run of the same kernels read 131k and 256k prompts at 1,002 and 1,006 tok/s and decoded at 40.2 tok/s
at 256k; runs vary with page migration on the Sparks. Through the 256k prompt the torch peak stayed at 93.7 GiB
against a 96.1 GiB estimate, the kept conversations' 1.6 GiB included. MemAvailable never fell below 9.5 GiB;
the admission reserve (a tenth of RAM) also carries 3.5-4.8 GiB of CUDA context, NCCL and graph memory outside
the estimate. Drafted replies equaled serial ones (9/9, 2k to 16k), resumed prompts equaled fresh ones (6/6),
and four interleaved conversations each equaled their solo replies. `--context 0` allocated a 487,495-token
window, not measured that far.

Short prompts, with the recipe's default drafting (DFlash2, `auto`), decode at 53.8 / 43.4 / 85.0 / 50.0 tok/s
(code and chat, sampled and greedy, 64 tokens, median of 5 seeds) against 0.3.6's 53.4 / 44.1 / 63.7 / 45.5. The
latent path rounds attention differently, so some replies differ from 0.3.6 (the greedy cells' texts, hence their
speeds); `TF_GLM_LATENT=0` gives 0.3.6's replies exactly.

## Responses and exactness

When thinking is disabled, the server closes the template's open think block so the response reaches
`content`. Both servers turn GLM's native `arg_key`/`arg_value` tool calls into OpenAI `tool_calls`,
decoding each value by the tool's schema (a string keeps its exact text); the CUDA server keeps Qwen's XML
parameter values as text.

CUDA tests cover row arithmetic, recurrent rollback and synthetic model execution. Validate real
weights separately for drafted/serial and resumed/fresh output, with both thinking modes.
CUDA graphs and eager execution must agree under the same rank configuration.
Use a separate fp32 reference for quality checks, with TF32 disabled on that reference.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above. Retain model and
runtime revisions with every run. Decode rate, cold/resumed first-token latency and peak memory are
TBD [release-0.3.5]. Record the selected drafter policy with the result.
