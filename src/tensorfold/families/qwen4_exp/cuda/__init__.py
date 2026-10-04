"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 10           # most MTP drafts a round (upstream 6; decode-suite A/B with the trained head, decode-merge 89640f3)
CONFIDENCE = 0.5     # a chain ends before a later draft the MTP head gives less than this (upstream 0.7; one stream or many)
COST = 0.06          # tokens per ms: a later draft only while the chain's probability repays its verify row
                     # (decode.cost_bars; README code greedy +14%, no category down: the fork's GREEDY measurements)
CONTEXT = 8192       # prompt plus reply tokens the caches hold


def env_int(name: str, default: int, low: int) -> int:
    """House knobs (TF_KEEP, TF_PASS_MIN, TF_FIRST_PASS, TF_SOLO_ROWS): unset or empty is ``default``; anything else
    must be a whole number of at least ``low``, else the server refuses to start (a 0-row pass floor stalled prompts
    beside live streams, a negative TF_KEEP failed every prompt end)."""

    import os

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or value < low:
        raise ValueError(f"{name}: a whole number of at least {low}, not {raw!r}")
    return value
