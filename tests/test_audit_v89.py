"""AUDIT-BUILD (v8.9): regressions found reviewing v8.5..v8.9; each test fails on speed-v8.9 (c396d6c).

1. The system-block checkpoint (TF_SYS_CHECKPOINT, default on) reached every CUDA engine through
   markers.snapshot_points, not only Flash Next: qwen3_5 / qwen3_5_moe kept an extra state at it.
2. Switch knobs read anything but "0" as on (TF_MOE_ROUTE_FUSED=off ran the fused route); TF_ROUTER_BE /
   TF_MOE_ROUTE_ROWS took any int (a router block that is not a power of two fails at the first decode round);
   TF_JOINT_RATE / TF_JOINT_VERIFY_MS took nan and inf.
3. A re-pin (TF_NGRAM_REPIN) whose mlock overlapped a release (a growing stream cache) kept the run pinned.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfold.cuda import markers
from tensorfold.cuda.markers import MIN_GAP, snapshot_points

OPEN, ASSISTANT = 900, 901
SRC = Path(__file__).resolve().parents[1] / "src"


def _prompt(system: int, user: int) -> list[int]:
    return [OPEN] + [5] * system + [OPEN] + [6] * user + [OPEN, ASSISTANT, 8]


# --- 1. the checkpoint belongs to Flash Next ------------------------------------------------------------------------

def test_snapshot_points_keep_no_checkpoint_unless_the_engine_asks(monkeypatch):
    monkeypatch.delenv("TF_SYS_CHECKPOINT", raising=False)
    ids = _prompt(13123, 600)                                    # a Hermes-sized system block (ends at 13124)
    plain = snapshot_points((OPEN,), (OPEN, ASSISTANT))          # what resume_points(model_dir) gives qwen3_5(_moe)
    assert plain(ids) == [13124, len(ids) - 3]
    assert plain.checkpoint(ids) is None
    flash = snapshot_points((OPEN,), (OPEN, ASSISTANT), markers.checkpoint_rows())
    assert flash(ids) == [12288, 13124, len(ids) - 3] and flash.checkpoint(ids) == 12288
    with pytest.raises(ValueError):
        snapshot_points((OPEN,), (OPEN, ASSISTANT), MIN_GAP - 1)


def test_only_the_flash_next_engine_asks_for_the_checkpoint():
    fam = SRC / "tensorfold" / "families"
    assert "resume_points(model_dir, checkpoint_rows())" in (fam / "qwen4_exp/cuda/engine.py").read_text()
    for other in ("qwen3_5/cuda/engine.py", "qwen3_5_moe/cuda/engine.py"):
        text = (fam / other).read_text()
        assert "resume_points(model_dir)" in text and "checkpoint_rows" not in text


# --- 2. knobs refuse values they would misread ----------------------------------------------------------------------

def _import(module: str, attr: str, **env) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if not k.startswith("TF_")}
    full.update(env, PYTHONPATH=str(SRC))
    code = f"import {module} as m; print(repr(m.{attr}))"
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("value, want", [("off", "False"), ("false", "False"), ("0", "False"), ("1", "True"),
                                         ("", "True"), ("single", "True")])
def test_route_fused_reads_off_as_off(value, want):
    got = _import("tensorfold.cuda.exl3.experts", "ROUTE_FUSED", TF_MOE_ROUTE_FUSED=value)
    assert got.returncode == 0, got.stderr[-400:]
    assert got.stdout.strip() == want


@pytest.mark.parametrize("name, value", [("TF_MOE_ROUTE_FUSED", "maybe"), ("TF_EXL3_PROMPT_SIDE", "sometimes"),
                                         ("TF_MOE_ROUTE_ROWS", "lots"), ("TF_MOE_ROUTE_ROWS", "-1")])
def test_experts_knobs_refuse_bad_values(name, value):
    got = _import("tensorfold.cuda.exl3.experts", "ROUTE_FUSED", **{name: value})
    assert got.returncode != 0 and name in got.stderr


def test_prompt_side_reads_off_as_off():
    got = _import("tensorfold.cuda.exl3.experts", "PROMPT_SIDE", TF_EXL3_PROMPT_SIDE="off")
    assert got.returncode == 0, got.stderr[-400:]
    assert got.stdout.strip() == "False"


@pytest.mark.parametrize("value", ["7", "0", "48", "big"])
def test_router_block_refuses_what_triton_cannot_take(value):
    pytest.importorskip("triton")
    got = _import("tensorfold.cuda.moe", "ROUTER_BE", TF_ROUTER_BE=value)
    assert got.returncode != 0 and "TF_ROUTER_BE" in got.stderr


def test_router_block_takes_the_measured_values():
    pytest.importorskip("triton")
    for value in ("16", "32", "64"):
        got = _import("tensorfold.cuda.moe", "ROUTER_BE", TF_ROUTER_BE=value)
        assert got.returncode == 0 and got.stdout.strip() == value, got.stderr[-400:]


@pytest.mark.parametrize("env", [{"TF_JOINT_RATE": "nan"}, {"TF_JOINT_RATE": "inf"}, {"TF_JOINT_RATE": "fast"},
                                 {"TF_JOINT_VERIFY_MS": "27,nan,40"}, {"TF_JOINT_VERIFY_MS": "27,inf"},
                                 {"TF_JOINT_VERIFY_MS": "27,x"}])
def test_joint_pricing_refuses_non_finite_values(env):
    from tensorfold.families.qwen4_exp.cuda.draft_cost import joint_settings

    with pytest.raises(ValueError):
        joint_settings({"TF_JOINT_PRICE": "1", **env})


def test_joint_pricing_takes_finite_values():
    from tensorfold.families.qwen4_exp.cuda.draft_cost import joint_settings

    got = joint_settings({"TF_JOINT_PRICE": "1", "TF_JOINT_RATE": "0.12", "TF_JOINT_VERIFY_MS": "27.2,32.7,36.4"})
    assert got == {"rate": 0.12, "verify": (27.2, 32.7, 36.4)}


# --- 3. a release during a re-pin's mlock wins ----------------------------------------------------------------------

def test_a_release_during_the_repin_mlock_cancels_the_run(tmp_path, monkeypatch):
    pytest.importorskip("safetensors.numpy")
    if not sys.platform.startswith("linux"):
        pytest.skip("mlock/proc")
    from tensorfold.families.qwen4_exp import host_residency

    from .test_flashnext_repin import _start

    res, table, ids, spans, multi, host, floor, clock = _start(tmp_path, monkeypatch)
    assert res.release(1) == spans[3]                            # a growth unpinned the last run
    clock.step(31)                                               # past the cooldown
    real, calls = host_residency._libc(), []

    class Libc:                                                  # the real libc; a cache grows during the mlock
        def mlock(self, at, size):
            rc = real.mlock(at, size)
            calls.append(("mlock", at, size))
            res.release(1)
            return rc

        def munlock(self, at, size):
            calls.append(("munlock", at, size))
            return real.munlock(at, size)

        def __getattr__(self, name):
            return getattr(real, name)

    monkeypatch.setattr(host_residency, "_libc", lambda: Libc())
    assert res.repin_step(clock.step(1)) == 0
    pick = calls[0][1:]
    assert calls[0][0] == "mlock" and ("munlock",) + pick in calls      # pinned, then given back
    assert pick not in table._locked and table.locked_bytes() == sum(spans[:2])
    assert res.repinned == 0
    monkeypatch.setattr(host_residency, "_libc", lambda: real)
    clock.step(31)                                              # quiet again: the runs come back one a period
    assert res.repin_step(clock.step(1)) == spans[2]
