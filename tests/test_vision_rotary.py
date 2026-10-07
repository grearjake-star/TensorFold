"""Multimodal rotary frequency axes are computed without loading Torch."""

from __future__ import annotations

import sys

import pytest

from tensorfold.vision import rotary


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def test_frequency_axes_interleave_height_and_width_with_temporal_remainder():
    axes = rotary.frequency_axes(16, (4, 2, 2))
    assert axes == [0, 1, 2, 0, 1, 2, 0, 0]
    assert [axes.count(axis) for axis in range(3)] == [4, 2, 2]


@pytest.mark.parametrize("dims,sections", [(0, (0, 0, 0)), (3, (1, 0, 0)), (8, (1, 1)),
                                           (8, (2, 1, 0)), (8, (4, 1, -1)), (8, (2, True, 1))])
def test_invalid_frequency_sections_are_rejected(dims, sections):
    with pytest.raises(ValueError):
        rotary.frequency_axes(dims, sections)
