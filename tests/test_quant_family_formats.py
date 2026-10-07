"""Family metadata gates and CLI routing are exercised without loading accelerator frameworks."""

from __future__ import annotations

import json
import sys

import pytest

from tensorfold import cli, families
from tensorfold.families import nemotron_h, qwen3_5, qwen4_exp


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def configuration(bits=4, group=64, **overrides):
    return {"quantization": {"bits": bits, "group_size": group, **overrides}}


def qwen_family():
    return families.Family("qwen3_5", qwen3_5.TITLE, qwen3_5.__name__)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_qwen_affine_metadata_accepts_all_declared_combinations(backend, bits, group):
    value = configuration(bits, group)
    qwen3_5.check_quantization(value, backend)
    families.require_readable(qwen_family(), value, backend)


def test_family_metadata_prioritizes_modern_quantization_and_skips_empty_overrides():
    value = configuration(8, 128, **{"model.layers.0.q_proj": {}, "model.layers.0.k_proj": False,
                                    "model.layers.0.v_proj": {"bits": 3}})
    value["quantization_config"] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    assert families.quant_method(value) == "mlx" and families.quantization(value) == (8, 128)
    assert families.layer_quantization(value) == {"model.layers.0.v_proj": (3, 64, "affine")}
    assert "8-bit" in families.describe_quantization(value)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("value", [configuration(7, 64), configuration(4, 16),
                                  configuration(mode="mxfp4"), configuration(mode="nvfp4"),
                                  configuration(mode="mxfp8"), configuration(quant_method="gptq")])
def test_qwen_family_refuses_unsupported_global_formats(backend, value):
    with pytest.raises(ValueError):
        families.require_readable(qwen_family(), value, backend)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_qwen_family_validates_per_module_overrides(backend):
    value = configuration(**{"model.layers.0.self_attn.q_proj": {"bits": 7}})
    with pytest.raises(ValueError, match="bits"):
        families.require_readable(qwen_family(), value, backend)


@pytest.mark.parametrize("package,group", [(nemotron_h, 64), (qwen4_exp, 32)])
@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_families_still_refuse_other_bit_widths_on_cuda(package, group, bits):
    family = families.Family(package.MODEL_TYPES[0], package.TITLE, package.__name__)
    with pytest.raises(ValueError, match="4-bit"):
        families.require_readable(family, configuration(bits, group), "cuda")


@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_family_metal_preflight_still_refuses_other_widths(tmp_path, bits):
    (tmp_path / "config.json").write_text(json.dumps(configuration(bits, 64)))
    with pytest.raises(ValueError, match="4-bit"):
        nemotron_h.check(tmp_path)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_flash_next_metal_preflight_takes_every_affine_width(tmp_path, bits, group):
    config = configuration(bits, group)
    config["quantization"]["language_model.model.layers.0.mlp.shared_expert.gate_proj"] = {"bits": 8, "group_size": 128}
    (tmp_path / "config.json").write_text(json.dumps(config))
    qwen4_exp.check(tmp_path)


def test_flash_next_refuses_unquantized_and_non_affine_checkpoints(tmp_path):
    for config in ({"model_type": "qwen4_exp"}, configuration(4, 32) | {"quantization": {"bits": 4, "group_size": 16}}):
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError):
            qwen4_exp.check(tmp_path)


