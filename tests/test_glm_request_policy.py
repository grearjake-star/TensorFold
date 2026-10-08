"""A GLM request's own draft policy (tf_policy or model@policy) that the engine cannot parse is a client error (400),
refused before streaming, not a 500 from inside the run."""

from types import SimpleNamespace

from tensorfold.families.glm5_next.cuda.app import policy_problem, request_policy
from tensorfold.families.glm5_next.cuda.engine import GlmEngine

ENGINE = SimpleNamespace(check_policy=lambda spec: GlmEngine.check_policy(None, spec))


def test_the_request_policy_comes_from_the_field_or_the_model_suffix():
    assert request_policy({"model": "glm-5.3-flash"}) is None
    assert request_policy({"model": "glm-5.3-flash@fc5:0.3"}) == "fc5:0.3"
    assert request_policy({"model": "glm-5.3-flash@3", "tf_policy": "0"}) == "0"


def test_valid_policies_pass():
    for spec in ("auto", "0", "3", "c4:0.3", "fc5:0.3", "a:0.8:0.9", "f3"):
        assert policy_problem({"model": f"glm@{spec}"}, ENGINE) is None
    assert policy_problem({"model": "glm"}, ENGINE) is None


def test_malformed_policies_name_the_policy():
    for spec in ("c9:0.3", "fc12:0.3", "x", "c3", "auto:1:2", "-1"):
        problem = policy_problem({"model": f"glm@{spec}"}, ENGINE)
        assert problem and f"draft policy {spec!r}" in problem, spec
    assert "draft policy 'nope'" in policy_problem({"model": "glm", "tf_policy": "nope"}, ENGINE)


def test_an_engine_without_a_parser_takes_the_policy_as_it_is():
    assert policy_problem({"model": "glm@anything"}, SimpleNamespace()) is None
