"""Request-local multimodal rotary positions without changing text-only arithmetic."""
from __future__ import annotations

from typing import Sequence


def frequency_axes(dims: int, sections: Sequence[int]) -> list[int]:
    if dims <= 0 or dims % 2 or len(sections) != 3:
        raise ValueError('multimodal rotary dimensions require three valid sections')
    if any(type(n) is not int or n < 0 for n in sections) or sum(sections) != dims // 2:
        raise ValueError('multimodal rotary sections must cover the rotary frequencies')
    return [1 if i % 3 == 1 and i < 3 * sections[1] else
            2 if i % 3 == 2 and i < 3 * sections[2] else 0 for i in range(dims // 2)]


