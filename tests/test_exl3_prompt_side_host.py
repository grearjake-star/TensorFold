"""TF_EXL3_PROMPT_SIDE (v8.8): which prompt-kernel instances run on the side stream (host side: the launch mask)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts  # noqa: E402

FOUR, SIX = 1 << 8, 1 << 12                       # K2 8 (4 bits, the routed experts) and 12 (6 bits, the shared one)


def test_main_width_is_the_most_common():
    assert experts.main_width([8] * 512 + [12]) == FOUR      # Flash Next: 512 routed 4-bit, the shared expert 6-bit
    assert experts.main_width([12, 8, 8]) == FOUR
    assert experts.main_width([12, 8]) == FOUR               # a tie: the narrowest
    assert experts.main_width([]) == 0


def test_the_shared_width_goes_to_the_side_stream():
    widths = FOUR | SIX
    got = experts.launch_widths(widths, FOUR, side=True)
    assert got & 0xFFFFFFFF == widths and got >> 32 == SIX   # every instance still launches; only 6-bit on the side
    assert experts.launch_widths(widths, FOUR, side=False) == widths      # TF_EXL3_PROMPT_SIDE=0: one stream


def test_one_width_or_no_main_never_forks():
    assert experts.launch_widths(FOUR, FOUR, side=True) == FOUR
    assert experts.launch_widths(FOUR | SIX, 0, side=True) == FOUR | SIX
    assert experts.launch_widths(SIX, FOUR, side=True) == SIX          # main absent from the layer: one stream


def test_three_widths_keep_the_main_on_the_current_stream():
    widths = (1 << 4) | FOUR | SIX
    got = experts.launch_widths(widths, FOUR, side=True)
    assert got >> 32 == (1 << 4) | SIX and got & 0xFFFFFFFF == widths


def test_env_default_is_on():
    import os
    import subprocess
    import sys

    code = "from tensorfold.cuda.exl3 import experts; print(experts.PROMPT_SIDE)"
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if k != "TF_EXL3_PROMPT_SIDE"}
    src = str(Path(__file__).resolve().parents[1] / "src")             # this tree, not an installed tensorfold
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    run = lambda e: subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,  # noqa: E731
                                   check=True).stdout.strip()
    assert run(env) == "True" and run({**env, "TF_EXL3_PROMPT_SIDE": "0"}) == "False"
