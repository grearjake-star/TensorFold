"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 10           # most MTP drafts a round (upstream 6; decode-suite A/B with the trained head, decode-merge 89640f3)
CONFIDENCE = 0.5     # a chain ends before a later draft the MTP head gives less than this (upstream 0.7; one stream or many)
COST = 0.06          # tokens per ms: a later draft only while the chain's probability repays its verify row
                     # (decode.cost_bars; README code greedy +14%, no category down: the fork's GREEDY measurements)
CONTEXT = 8192       # prompt plus reply tokens the caches hold
