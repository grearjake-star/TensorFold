"""Flash Next's expected-time draft stops: ``--mtp-cost`` verifies a later draft only while its chance of being kept,
the head's chain product calibrated by live rounds, repays the ms it adds to the round; ``--mtp-lookahead`` drafts
toward the depth with the most expected tokens per ms of the round. Drafts change speed only: drafted output equals
serial output, greedy and sampled, at any price. ``--mtp-live-cost`` moves either stop's prices toward the rounds'
measured times."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.draft_price import PRIOR, RATE, DraftPrice


def test_each_draft_is_priced_by_the_ms_it_adds() -> None:
    price = DraftPrice(0.1, (30.0, 34.0, 38.5, 41.0), 1.5, 5)
    assert [round(price.adds(j), 9) for j in range(5)] == [5.5, 6.0, 4.0, 4.0, 4.0]   # past the table: its last step
    assert price.pays(0, 0.6) and not price.pays(0, 0.5)                              # 0.1 x 5.5 ms = 0.55 tokens


def test_live_rounds_calibrate_the_chain_product_per_depth_and_mode() -> None:
    price = DraftPrice(0.05, (30.0, 34.0, 38.0), 1.0, 2)
    assert price.chance(1, 0.4) == 0.4                         # unseen: the head's product
    assert price.pays(1, 0.4)
    for _ in range(400):
        price.begin(False)
        price.products = [0.8, 0.4]
        price.observe(2, 1)                                    # depth 1 always kept, depth 2 never
    assert price.chance(0, 0.8) == pytest.approx(1.0, abs=1e-6)
    assert price.chance(1, 0.4) < 0.01 and not price.pays(1, 0.4)
    price.begin(True)
    assert price.chance(1, 0.4) == 0.4                         # sampled rounds keep their own counts


def test_unverified_drafts_teach_nothing() -> None:
    price = DraftPrice(0.05, (30.0, 34.0, 38.0), 1.0, 2)
    price.begin(False)
    price.products = [0.9, 0.5]
    price.observe(1, 0)                                        # a window cut the second draft before verifying it
    assert price.kept[False][1] is None and price.kept[False][0] == pytest.approx(0.9 * (1 - RATE))



def test_lookahead_drafts_through_a_dear_row_when_cheap_rows_follow() -> None:
    table = (30.0, 31.0, 40.0, 41.0, 42.0)                     # the second draft's row costs 9 ms, the rest 1 ms
    marginal, ahead = DraftPrice(0.1, table, 0.0, 4), DraftPrice(0.0, table, 0.0, 4, lookahead=True)
    for price in (marginal, ahead):
        price.begin(False)
        price.products = [0.85]
    assert not marginal.more(0, 0.85)                          # 0.85 < 0.1 x 9 ms: the marginal bar stops at one
    assert ahead.best([0.85], 1) == 4 and ahead.more(0, 0.85)  # four drafts give the most tokens per ms
    assert not ahead.more(0, 0.2)                              # a weak chain: one draft is the best depth


def test_lookahead_drops_a_draft_its_depth_does_not_repay_and_respects_the_chain_limit() -> None:
    price = DraftPrice(0.0, (30.0, 31.0, 40.0, 41.0, 42.0), 0.5, 4, lookahead=True)
    price.begin(False)
    price.products = [0.9]
    assert price.keeps(0, 0.01)                                # the first draft is always verified
    assert not price.keeps(1, 0.05) and price.keeps(1, 0.85)
    price.begin(False, 1)                                      # one token left in the reply
    assert not price.more(0, 0.99)


def test_lookahead_prices_deeper_drafts_at_the_rate_live_rounds_kept_them() -> None:
    price = DraftPrice(0.0, (30.0, 31.0, 32.0, 33.0), 0.0, 3, lookahead=True)
    price.begin(False)
    assert price.more(0, 0.9)                                  # no rounds yet: the head's own rate carries on
    for _ in range(400):
        price.begin(False)
        price.products = [0.9, 0.85, 0.8]
        price.observe(3, 1)                                    # depth 1 always kept, depth 2 never
    price.begin(False)
    assert price.best([0.9], 1) == 1 and not price.more(0, 0.9)
    price.begin(True)
    assert price.more(0, 0.9)                                  # sampled rounds keep their own counts


def test_live_cost_fits_the_rounds_measured_times_from_the_startup_table() -> None:
    table = (30.0, 33.0, 36.0, 39.0, 42.0)
    fixed, live = DraftPrice(0.0, table, 1.0, 4, lookahead=True), DraftPrice(0.0, table, 1.0, 4, lookahead=True, live=True)
    assert live.verify == fixed.verify and live.line() == (0.0, 0.0)          # no rounds yet: the startup table
    for i in range(400):                                   # live rounds: 2 ms more, and 1.5 ms more a draft
        d = 1 + i % 3
        fixed.timed(d, fixed.table_ms(d) + 2.0 + 1.5 * d)
        live.timed(d, live.table_ms(d) + 2.0 + 1.5 * d)
    assert fixed.verify[:5] == list(table)                 # off unless asked
    a, b = live.line()
    assert a == pytest.approx(2.0, abs=0.1) and b == pytest.approx(1.5, abs=0.1)     # PRIOR pulls a little to 0
    assert live.verify[4] == pytest.approx(42.0 + a + 4 * b)
    (at, bt), (an, bn) = live.fits()
    assert (at, bt) == pytest.approx((30.0, 4.0)) and bn == pytest.approx(4.0 + b)
    live.timed(2, 1e6)                                     # one stalled round moves the line by a bounded step
    assert live.line()[1] < b + 2.0 and PRIOR > 0


def test_live_cost_stops_the_lookahead_where_dear_live_drafts_no_longer_pay() -> None:
    table = (30.0, 31.0, 32.0, 33.0, 34.0)                 # the table: drafts nearly free
    price = DraftPrice(0.0, table, 0.2, 4, lookahead=True, live=True)
    price.begin(False)
    assert price.best([0.9], 1) == 4                       # priced on the table: draft all four
    for _ in range(200):
        price.timed(2, price.table_ms(2) + 60.0)           # live: every draft costs 30 ms more
        price.timed(1, price.table_ms(1) + 30.0)
    price.begin(False)
    assert price.best([0.9], 1) < 4 and not price.more(0, 0.6)
    marginal = DraftPrice(0.02, table, 0.2, 4, live=True)
    assert marginal.pays(1, 0.25)                          # 0.02 x 1.2 ms
    for _ in range(200):
        marginal.timed(2, marginal.table_ms(2) + 30.0)
        marginal.timed(1, marginal.table_ms(1) + 15.0)
    assert not marginal.pays(1, 0.25)                      # 0.02 x ~15 ms


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@cuda
def test_measured_round_costs_are_positive_ordered_and_leave_the_state_empty() -> None:
    from test_flashnext_forward import _model

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.draft_price import measure_round_costs

    w = _model()
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    prompt = [5, 17, 99, 250, 7, 64]
    first = prefill(e, prompt, None)
    verify, step = measure_round_costs(e, 7, reps=2)
    assert len(verify) == 7 and all(v > 0 for v in verify) and step >= 0
    assert all(b >= a for a, b in zip(verify, verify[1:]))           # a wider window never prices below a narrower
    assert e.st.pos == 0 and e.st.mtp_len == 0                       # emptied, as warm leaves it
    assert prefill(e, prompt, None) == first


@cuda
@pytest.mark.parametrize("sampling_seed", [None, 1234])
def test_drafts_under_the_cost_stop_give_the_serial_tokens(sampling_seed) -> None:
    from test_flashnext_forward import _model

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = _model()
    sampling = None if sampling_seed is None else Sampling(seed=sampling_seed, temperature=0.7, top_k=20, top_p=0.8)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 32, sampling).tokens
    for depth in (1, 3, 6):
        for confidence in (0.0, 0.7):
            for cost in (0.0, 0.02, 0.2, 2.0, None):                 # any table: the stop moves speed only
                price = DraftPrice(cost or 0.0, (24.0, 28.0, 32.0, 35.0, 38.0, 42.0, 44.0), 1.2, depth,
                                   lookahead=cost is None)              # None: --mtp-lookahead
                for _ in range(2):                                   # the second reply runs on calibrated counts
                    assert prefill(e, prompt, sampling) == first
                    got = mtp_decode(e, first, 32, sampling, depth=depth, confidence=confidence, price=price)
                    assert got.tokens == ref, (depth, confidence, cost)
                    if (cost or 0) >= 2.0:          # an unpayable bar (> 1): every round verifies its first draft only
                        assert max(got.widths) <= 2


@cuda
@pytest.mark.parametrize("sampling_seed", [None, 1234])
def test_drafts_priced_on_live_round_times_give_the_serial_tokens(sampling_seed) -> None:
    from test_flashnext_forward import _model

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = _model()
    sampling = None if sampling_seed is None else Sampling(seed=sampling_seed, temperature=0.7, top_k=20, top_p=0.8)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 32, sampling).tokens
    table = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)             # far below the tiny model's real rounds: live moves it
    for depth in (1, 3, 6):
        for confidence in (0.0, 0.7):
            for cost in (0.0, 0.2):                         # 0.0: --mtp-lookahead
                price = DraftPrice(cost, table, 0.05, depth, lookahead=cost == 0.0, live=True)
                for _ in range(3):                          # later replies run on live prices
                    assert prefill(e, prompt, sampling) == first
                    got = mtp_decode(e, first, 32, sampling, depth=depth, confidence=confidence, price=price)
                    assert got.tokens == ref, (depth, confidence, cost)
                assert price.rounds > 0 and price.verify[0] > table[0]   # rounds were timed and re-priced
