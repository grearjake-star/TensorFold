"""A reply names the model id its request asked for when the server answers to it, else the served name, whichever server shape answers."""

from types import SimpleNamespace

import pytest

from tensorfold.server.http import reply_model


@pytest.mark.parametrize("asked, named", [("alias-a", "alias-a"), ("fake-27b", "fake-27b"), ("other", "fake-27b"),
                                          (None, "fake-27b"), (7, "fake-27b")])
def test_reply_model_picks_the_asked_id_only_when_served(asked, named):
    mlx = SimpleNamespace(served_name="fake-27b", model_ids=["fake-27b", "alias-a"])
    cuda = SimpleNamespace(served="fake-27b", model_ids=["fake-27b", "alias-a"])
    body = {} if asked is None else {"model": asked}
    assert reply_model(mlx, body) == reply_model(cuda, body) == named


