"""AUDIT: the house env knobs (TF_KEEP, TF_PASS_MIN, TF_FIRST_PASS, TF_SOLO_ROWS) refuse values that break serving.

TF_PASS_MIN=0 or TF_FIRST_PASS=0 sized live passes at 0 rows (a prompt beside a decoding stream never filled until
every stream ended); TF_KEEP=-1 popped from an empty kept list (IndexError at every prompt end)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _import(module: str, **env) -> subprocess.CompletedProcess:
    full = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", **env}
    return subprocess.run([sys.executable, "-c", f"import {module}"], env=full, capture_output=True, text=True,
                          timeout=300)


@pytest.mark.parametrize("name,value", [("TF_PASS_MIN", "0"), ("TF_FIRST_PASS", "0"), ("TF_PASS_MIN", "-64"),
                                        ("TF_FIRST_PASS", "lots")])
def test_a_pass_size_knob_below_one_row_is_refused(name, value):
    pytest.importorskip("torch")
    r = _import("tensorfold.families.qwen4_exp.cuda.multi_fill", **{name: value})
    assert r.returncode != 0 and name in r.stderr


def test_a_negative_keep_is_refused():
    r = _import("tensorfold.families.qwen4_exp.cuda.engine", TF_KEEP="-1")
    assert r.returncode != 0 and "TF_KEEP" in r.stderr


def test_the_house_values_and_empty_values_still_load():
    pytest.importorskip("torch")
    r = _import("tensorfold.families.qwen4_exp.cuda.multi_fill", TF_PASS_MIN="512", TF_FIRST_PASS="", TF_KEEP="16")
    assert r.returncode == 0, r.stderr


def test_env_int():
    from tensorfold.families.qwen4_exp.cuda import env_int

    os.environ.pop("TF_AUDIT_X", None)
    assert env_int("TF_AUDIT_X", 7, 0) == 7
    for raw, want in (("", 7), (" 16 ", 16), ("0", 0)):
        os.environ["TF_AUDIT_X"] = raw
        assert env_int("TF_AUDIT_X", 7, 0) == want
    for raw in ("-1", "1.5", "x"):
        os.environ["TF_AUDIT_X"] = raw
        with pytest.raises(ValueError, match="TF_AUDIT_X"):
            env_int("TF_AUDIT_X", 7, 0)
    os.environ.pop("TF_AUDIT_X")
