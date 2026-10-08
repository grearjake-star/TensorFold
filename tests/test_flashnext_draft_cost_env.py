"""TF_DRAFT_COST is a finite number of tokens per ms, 0 or more, or off.

It was read with a bare float(): "off" failed with Python's generic float message, "nan" passed the "0 or more"
check and silently turned the expected-time stop off, and "inf" stopped every chain after its first draft (and
both broke the two ranks' settings check, which rounds the cost to an integer). Drafts change speed only."""

import pytest

from tensorfold.families.qwen4_exp.cuda import COST
from tensorfold.families.qwen4_exp.cuda.draft_cost import cost_setting


@pytest.mark.parametrize("raw,want", [(None, COST), ("0.02", 0.02), (" 0.5 ", 0.5), ("0", 0.0), ("", 0.0),
                                      ("off", 0.0), ("OFF", 0.0), ("1e-6", 1e-6)])
def test_accepted_values(raw, want):
    env = {} if raw is None else {"TF_DRAFT_COST": raw}
    assert cost_setting(COST, env) == want


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-1", "fast", "0.1.2"])
def test_refused_values_name_the_knob(raw):
    with pytest.raises(ValueError, match="TF_DRAFT_COST: tokens per ms, a finite number"):
        cost_setting(COST, {"TF_DRAFT_COST": raw})


def test_the_old_reading_let_nan_and_inf_through():
    """The engine's former expression, for the record: no error for either."""

    for raw in ("nan", "inf"):
        cost = float(raw or 0)
        assert not cost < 0
