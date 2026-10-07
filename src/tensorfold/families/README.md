# Model families

The CLI discovers family packages by `MODEL_TYPES`, matching the checkpoint configuration.
Each family provides its own CUDA engine (`cuda_engine`) in its `cuda/` package.

| Package | Model | CUDA |
| --- | --- | --- |
| `nemotron_h/` | Nemotron 3.5 Lightning | One or two ranks; MLX 4-bit weights, MTP |
| `qwen3_5/` | Qwen3.8-27B | One or two ranks; EXL3, NVFP4 or MLX affine weights, DFlash2 |
| `qwen3_5_moe/` | Qwen3.6-35B-A3B | One rank; MLX 4-bit weights, MTP |
| `qwen4_exp/` | Qwen3.8 Flash Next | One or two ranks; EXL3, NVFP4 or MLX 4-bit weights, MTP |
| `glm5_next/` | GLM-5.3-Flash | Two ranks, MTP and optional DFlash2 |

Each stream has independent state; shared execution must reproduce its solo output. Backend support is declared
by the package, not inferred from a model's name.

See [the recipe book](../../../docs/recipes/README.md) for checkpoints and limits and
[the CUDA interface](../../../docs/recipes/adding-a-cuda-family.md).
