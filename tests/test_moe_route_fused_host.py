"""S1-ROUTE (TF_MOE_ROUTE_FUSED): which decode windows take the fused route launch (host side: the gate)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _args(R=3, dims=2560, count=513, slots=11, rows=64, dtype=torch.bfloat16):
    x = torch.empty((1, 8), dtype=dtype)
    ex = SimpleNamespace(dims=dims, count=count)
    s = SimpleNamespace(rows=rows, slots=slots, route_done=None)
    return R, x, ex, s, 10, 512


def test_flash_next_decode_windows_take_the_fused_route():
    for R in (1, 2, 3, 11, 16, 33, 64):
        assert experts.route_ok(*_args(R=R), fused=True), R
    assert experts.route_ok(*_args(dtype=torch.float16), fused=True)


def test_the_switch_and_the_limits_keep_the_three_launches():
    assert not experts.route_ok(*_args(), fused=False)
    assert not experts.route_ok(*_args(R=0), fused=True)
    assert not experts.route_ok(*_args(R=65, rows=128), fused=True)      # a prompt window: the plan path
    assert not experts.route_ok(*_args(R=40, rows=32), fused=True)       # more rows than the scratch holds
    assert not experts.route_ok(*_args(dims=2560 + 128), fused=True)     # K not a multiple of 512
    assert not experts.route_ok(*_args(count=1025), fused=True)
    assert not experts.route_ok(*_args(slots=12), fused=True)            # slots != top_k + 1
    assert not experts.route_ok(*_args(dtype=torch.float32), fused=True)
    R, x, ex, s, k, e = _args()
    del s.route_done
    assert not experts.route_ok(R, x, ex, s, k, e, fused=True)           # a scratch without the counter


def _flag(**env):
    full = {k: v for k, v in os.environ.items() if not k.startswith("TF_MOE_ROUTE")}
    full.update(PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", **env)
    r = subprocess.run([sys.executable, "-c", "from tensorfold.cuda.exl3 import experts as x; "
                        "print(x.ROUTE_FUSED, x.ROUTE_ROWS)"], env=full, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_env_default_is_on_and_zero_turns_it_off():
    assert _flag() == ["True", "64"]
    assert _flag(TF_MOE_ROUTE_FUSED="0") == ["False", "64"]
    assert _flag(TF_MOE_ROUTE_FUSED="1", TF_MOE_ROUTE_ROWS="16") == ["True", "16"]
