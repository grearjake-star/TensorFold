"""House: two MTP heads, one picked per conversation by its prompt length (``TF_HEAD_SWITCH_ROWS``).

The house head (``TF_MTP_HEAD``, trained on contexts up to 2048 entries) drafts short prompts; a second head (the
checkpoint's own MTP head by default, or ``TF_MTP_HEAD_LONG``) drafts prompts of ``TF_HEAD_SWITCH_ROWS`` tokens or
more. Unset: one head, exactly today's behaviour.

- **Exact either way.** Drafts are verified against the same keyed samples serial decoding draws, so either head
  changes only how many drafts are kept, never a token.
- **A head belongs to a sequence, not a round.** The MTP layer keeps its own attention cache (its k/v projections
  write it), so a sequence's head is fixed when its MTP cache starts: a fresh prompt picks by its length
  (``State.mtp_head``); a kept prompt end carries its head in its snapshot, and a request that resumes it keeps that
  head (switching would draft against the other head's keys). Prefix copies carry the source's head.
- **One launch, one head.** An MTP launch reads one head's weights; --parallel's multi-stream steps run one launch a
  head when streams differ (``groups``). CUDA graphs bake weight pointers in, so graph keys name the head (head 0
  keeps today's keys).
- **Memory.** The second head shares the MTP layer's routed and shared experts (one EXL3 table): it adds only fc_e,
  fc_h, the attention projections, norms, the three hyper-connection mixers and the router (``head_bytes``).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

SWITCH_ENV = "TF_HEAD_SWITCH_ROWS"
LONG_ENV = "TF_MTP_HEAD_LONG"


def switch_rows(env=os.environ) -> int | None:
    """``TF_HEAD_SWITCH_ROWS``: prompts of at least this many tokens draft with the long head; unset or 0: off."""

    raw = env.get(SWITCH_ENV, "").strip()
    if not raw:
        return None
    try:
        rows = int(raw)
    except ValueError:
        raise ValueError(f"{SWITCH_ENV}: a prompt length in tokens, not {raw!r}") from None
    if rows < 0:
        raise ValueError(f"{SWITCH_ENV}: a prompt length in tokens, 0 or more, not {rows}")
    return rows or None


def long_head(env=os.environ) -> str | None:
    """The long head's file (``TF_MTP_HEAD_LONG``); unset or ``own``: None, the checkpoint's own MTP tensors."""

    raw = env.get(LONG_ENV, "").strip()
    return None if raw in ("", "own") else raw


def pick(w, prompt_len: int) -> int:
    """The head a fresh prompt of ``prompt_len`` tokens drafts with: 0 (the primary) or 1 (the long head)."""

    heads = getattr(w, "mtp_heads", None)
    rows = w.meta.get("head_switch_rows") if hasattr(w, "meta") else None
    return 1 if heads and len(heads) > 1 and rows and prompt_len >= rows else 0


def head_of(st) -> int:
    return getattr(st, "mtp_head", 0)


def of(w, segs: Sequence) -> Any:
    """The MTP weights a launch over ``segs`` ((state, a0, a1), ...) reads: its states' one head."""

    heads = getattr(w, "mtp_heads", None)
    if not heads:
        return w.mtp
    h = head_of(segs[0][0])
    if any(head_of(st) != h for st, _, _ in segs):
        raise ValueError("an MTP launch reads one head: group the streams by head first")
    return heads[h]


def launch_key(segs: Sequence) -> tuple:
    """The graph-key suffix of a launch's head: () for head 0 (today's keys), (head,) otherwise."""

    h = head_of(segs[0][0]) if segs else 0
    return (h,) if h else ()


def groups(states: Sequence) -> list[list[int]]:
    """Indices of ``states`` by head, each group in order, groups by first appearance (one group: one launch)."""

    out: dict[int, list[int]] = {}
    for i, st in enumerate(states):
        out.setdefault(head_of(st), []).append(i)
    return list(out.values())


def _storages(x: Any, seen: dict[int, int], skip: set[int], depth: int = 0) -> None:
    if depth > 8 or id(x) in skip:
        return
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu":
            s = x.untyped_storage()
            seen[s.data_ptr()] = s.nbytes()
        return
    if isinstance(x, (list, tuple)):
        for y in x:
            _storages(y, seen, skip, depth + 1)
        return
    if isinstance(x, (str, int, float, bool, type(None))):
        return
    fields = getattr(x, "__dataclass_fields__", None)
    names = list(fields) if fields else list(getattr(x, "__dict__", {}))
    for name in names:
        if name in ("sc", "users"):                      # the model's shared scratch, not the head's
            continue
        _storages(getattr(x, name, None), seen, skip, depth + 1)


def head_bytes(w) -> int:
    """Device bytes the second head adds: its storages that the primary head does not hold (the experts are one)."""

    heads = getattr(w, "mtp_heads", None)
    if not heads or len(heads) < 2:
        return 0
    first: dict[int, int] = {}
    second: dict[int, int] = {}
    _storages(heads[0], first, set())
    _storages(heads[1], second, set())
    return sum(n for p, n in second.items() if p not in first)
