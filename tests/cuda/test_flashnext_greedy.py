"""The expected-time draft stop (``decode.draft`` with ``cost`` > 0, the shipped ``COST``; the fork's GREEDY measurements):
a later draft is verified only while the product of the head's probabilities of drafts 1..j reaches
``cost x (VERIFY_MS[j+1] - VERIFY_MS[j] + DRAFT_MS)``; the first draft is always verified (0.3.4+'s rule), a chain
still ends before a later draft under ``confidence``, and the MTP step of a draft the product already fails is
skipped. It changes which drafts are verified, never the output: drafted == serial, greedy and sampled."""

import inspect
import os

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import COST, decode
from tensorfold.families.qwen4_exp.cuda.decode import DRAFT_MS, VERIFY_MS, cost_bars, draft


class _St:
    def __init__(self):
        self.mtp_len, self.mtp_drafted = 100, 0

    def set_mtp_len(self, n):
        self.mtp_len = n


class _Buf:
    streams = torch.zeros(4, 8)


class _FakeEngine:
    """Scripted head probabilities: the j-th MTP step's draft is token 1000 + j with probability probs[j]."""

    def __init__(self, probs):
        self.probs, self.st, self.mbuf, self.steps = list(probs), _St(), _Buf(), 0

    def mtp_forward(self, next_tokens, streams):
        self.steps += 1
        return torch.tensor([self.steps - 1])

    def sample_draft(self, logits, position, sampling):
        j = int(logits[0])
        return 1000 + j, self.probs[j]


def _run(probs, count, confidence, cost):
    e = _FakeEngine(probs)
    out = draft(e, _Buf.streams[:1], [7], 50, count, None, confidence, cost=cost)
    return len(out), e.steps


def test_cost_bars_follow_the_verify_table():
    bars = cost_bars(0.1, 4)
    assert len(bars) == 5
    for j in range(5):
        assert bars[j] == pytest.approx(0.1 * (VERIFY_MS[j + 1] - VERIFY_MS[j] + DRAFT_MS))
    far = cost_bars(0.1, len(VERIFY_MS) + 3)             # past the table: its last step again
    assert far[-1] == pytest.approx(0.1 * (VERIFY_MS[-1] - VERIFY_MS[-2] + DRAFT_MS))
    assert 0 < COST < 0.2
    # the engine applies COST; mtp_decode's own default stays the plain rule, which direct callers (tests that swap
    # in their own draft(), tools) rely on
    assert inspect.signature(decode.mtp_decode).parameters["cost"].default == 0.0


def test_the_rule_keeps_the_first_draft_and_stops_on_the_product():
    bar = 0.06 * (VERIFY_MS[1] - VERIFY_MS[0] + DRAFT_MS)          # ~0.3
    # confident chain: all drafts, one MTP step each after the absorb
    assert _run([0.99] * 6, 5, 0.5, 0.06) == (5, 5)
    # a weak first draft is still verified (the first-draft rule), and nothing chains after it
    assert _run([0.2, 0.99, 0.99], 3, 0.5, 0.06) == (1, 1)
    # a later draft under the cutoff ends the chain before it
    assert _run([0.9, 0.4, 0.99], 3, 0.5, 0.06)[0] == 1
    # every draft passes the 0.5 cutoff, but the running product 0.6^j falls under the bar: the product stops it
    n, steps = _run([0.6] * 8, 8, 0.5, 0.06)
    products = [0.6 ** (j + 1) for j in range(8)]
    bars = cost_bars(0.06, 8)
    want = next(j for j in range(8) if products[j] < bars[j])
    assert n == want and bar > 0 and steps == n + 1     # the absorb, then a step for each draft up to the failed one
    # the product passes draft 4's bar but already misses draft 5's: draft 5's MTP step is skipped
    bars = cost_bars(0.06, 6)
    assert bars[3] <= 0.28 < bars[4]
    assert _run([1.0, 1.0, 1.0, 0.28, 1.0, 1.0], 6, 0.0, 0.06) == (4, 4)
    # cost 0 is the plain rule: the same chain runs to the cutoff only
    assert _run([0.6] * 8, 8, 0.5, 0.0) == (8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("sampling", ["greedy", "sampled"])
def test_cost_stopped_drafts_give_serial_tokens(tmp_path, sampling):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_flashnext_tp import PROMPT, _checkpoint

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    s = None if sampling == "greedy" else Sampling(seed=23, top_k=20, top_p=0.95)
    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    os.environ["TF_DRAFT_COST"] = "0.02"
    try:
        eng = FlashNextEngine(model, depth=5, confidence=0.0, draft_vocab=None, max_len=512, prefetch=False)
    finally:
        del os.environ["TF_DRAFT_COST"]
    assert eng.cost == 0.02
    assert FlashNextEngine(model, depth=2, draft_vocab=None, max_len=512, prefetch=False).cost == COST
    for value, want in (("0", 0.0), ("", 0.0)):          # TF_DRAFT_COST=0 (or empty): the plain rule
        os.environ["TF_DRAFT_COST"] = value
        try:
            assert FlashNextEngine(model, depth=2, draft_vocab=None, max_len=512, prefetch=False).cost == want
        finally:
            del os.environ["TF_DRAFT_COST"]
    os.environ["TF_DRAFT_COST"] = "-1"
    try:
        with pytest.raises(ValueError, match="TF_DRAFT_COST"):
            FlashNextEngine(model, depth=2, draft_vocab=None, max_len=512, prefetch=False)
    finally:
        del os.environ["TF_DRAFT_COST"]
    first = decode.prefill(eng.e, PROMPT, s)
    ref = decode.serial_decode(eng.e, first, 40, s).tokens
    widths = set()
    for depth in (1, 3, 5):
        for confidence in (0.0, 0.3):
            for cost in (0.0, 1e-6, 0.002, 0.02, 0.06, 0.5):
                decode.prefill(eng.e, PROMPT, s)
                got = decode.mtp_decode(eng.e, first, 40, s, depth=depth, confidence=confidence, cost=cost)
                assert got.tokens == ref, (depth, confidence, cost)
                widths.add(tuple(got.widths))
    assert len(widths) > 3                 # the rules really verified different windows
    # the engine's own path (generate) with its cost equals serial too
    got = []
    eng.generate(PROMPT, 40, s, lambda new: got.extend(new) and False)
    assert got and got == ref[:len(got)]
