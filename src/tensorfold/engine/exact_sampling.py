"""Key Gumbel draws by seed, position and token id so verification matches serial top-k/top-p/min-p sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Sequence

import numpy as np

MARGIN = 8     # candidates beyond top_k read from the GPU, so tied values resolve by id on the CPU


@dataclass(frozen=True)
class Sampling:
    seed: int
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0          # keep tokens at least min_p times as likely as the likeliest (after temperature)

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_k", max(0, int(self.top_k)))

    @property
    def min_log(self) -> float:
        """ln(min_p), -inf when off: every rule adds it to the row's top scaled logit, one float64 add."""

        return math.log(self.min_p) if self.min_p > 0.0 else -math.inf


def _salt_from_env() -> int:
    """``TENSORFOLD_SEED_SALT``: an integer mixed into every prompt-derived seed (0, the default, changes nothing)."""

    value = os.environ.get("TENSORFOLD_SEED_SALT", "").strip()
    try:
        return int(value) if value else 0
    except ValueError:
        raise ValueError(f"TENSORFOLD_SEED_SALT={value}: an integer") from None


SEED_SALT = _salt_from_env()


def seed_for(tokens: Sequence[int], salt: int | None = None) -> int:
    """A reproducible seed from the prompt: the same conversation samples the same reply (for one salt)."""

    salt = SEED_SALT if salt is None else salt
    digest = hashlib.sha256((",".join(str(int(t)) for t in tokens) + f"|{salt}").encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def uniform(seed: int, position: int, ids: np.ndarray) -> np.ndarray:
    """Uniform (0, 1) doubles from a splitmix64 hash of (seed, position, token id)."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.uint64(position) * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def uniform_rows(seed: int, positions: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``uniform`` for many positions at once: row r is ``uniform(seed, positions[r], ids[r])``, same bits."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.asarray(positions).astype(np.uint64)[:, None] * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def choose(values: np.ndarray, ids: np.ndarray, position: int, s: Sampling) -> int:
    """One row: candidate logits ``values`` for token ``ids`` -> the sampled token id."""

    order = np.lexsort((ids, -values))
    k = max(1, min(int(s.top_k) if s.top_k else len(ids), len(ids)))
    ids = ids[order][:k]
    scaled = values[order][:k].astype(np.float64) / max(float(s.temperature), 1e-6)
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    if 0.0 < s.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), s.top_p) + 1)
        ids, scaled = ids[:keep], scaled[:keep]
    if s.min_p > 0.0:            # a prefix of the order: the tokens within ln(min_p) of the top
        keep = int((scaled >= scaled[0] + s.min_log).sum())
        ids, scaled = ids[:keep], scaled[:keep]
    gumbel = -np.log(-np.log(uniform(s.seed, position, ids)))
    return int(ids[int(np.argmax(scaled + gumbel))])


def choose_rows(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], s: Sampling) -> list[int]:
    """Choose each row with the same operation order and bits as an independent ``choose`` call."""

    rows, width = ids.shape
    order = np.lexsort((ids, -values), axis=-1)
    k = max(1, min(int(s.top_k) if s.top_k else width, width))
    ids = np.take_along_axis(ids, order, axis=-1)[:, :k]
    scaled = np.take_along_axis(values, order, axis=-1)[:, :k].astype(np.float64) / max(float(s.temperature), 1e-6)
    score = scaled - np.log(-np.log(uniform_rows(s.seed, np.asarray(positions), ids)))
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        keep = (np.cumsum(probs, axis=-1) < s.top_p).sum(axis=-1) + 1    # searchsorted(cumsum, top_p) + 1
        score[np.arange(k)[None, :] >= keep[:, None]] = -np.inf
    if s.min_p > 0.0:
        score[scaled < scaled[:, :1] + s.min_log] = -np.inf               # ``choose``'s min_p prefix
    return [int(t) for t in ids[np.arange(rows), np.argmax(score, axis=-1)]]


__all__ = ["MARGIN", "Sampling", "choose", "choose_rows", "seed_for", "uniform",
           "uniform_rows"]
