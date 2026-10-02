"""Sampled requests without a seed are independent draws (OpenAI semantics); a seed, or TF_PROMPT_SEED=1, reproduces.

Guard for the 2026-09-29 bakeoff finding (the fork's TFLIMIT measurements): the prompt-derived default seed made all three
sampled repeats of every task byte-identical, so the sampled phase measured one draw per task.
"""

from tensorfold.cuda.server import App
from tensorfold.engine.exact_sampling import seed_for


def _app() -> App:
    app = object.__new__(App)
    app.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    return app


BODY = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
PROMPT = [11, 22, 33, 44]


def test_no_seed_draws_a_fresh_seed_per_request(monkeypatch):
    monkeypatch.delenv("TF_PROMPT_SEED", raising=False)
    seeds = {_app().sampling_for(dict(BODY), PROMPT).seed for _ in range(8)}
    assert len(seeds) == 8
    assert seed_for(PROMPT) not in seeds


def test_explicit_seed_is_used(monkeypatch):
    monkeypatch.delenv("TF_PROMPT_SEED", raising=False)
    a = _app().sampling_for({**BODY, "seed": 1234}, PROMPT)
    b = _app().sampling_for({**BODY, "seed": 1234}, PROMPT)
    assert a == b and a.seed == 1234 and (a.temperature, a.top_k, a.top_p) == (1.0, 20, 0.95)


def test_prompt_seed_opt_in_restores_the_old_rule(monkeypatch):
    monkeypatch.setenv("TF_PROMPT_SEED", "1")
    assert _app().sampling_for(dict(BODY), PROMPT).seed == seed_for(PROMPT)


def test_greedy_has_no_sampling(monkeypatch):
    monkeypatch.delenv("TF_PROMPT_SEED", raising=False)
    assert _app().sampling_for({"temperature": 0}, PROMPT) is None


def test_repeated_requests_reach_the_engine_with_distinct_seeds_and_report_them(tmp_path, monkeypatch):
    """Through the HTTP server: identical unseeded requests are separate draws, drawn once per request (the
    prepared sampling is what the engine gets), and the reply's tensorfold block names the seed used."""

    import json

    from tests.test_cuda_admission import http_server
    from tests.test_cuda_server_errors import HI, app_for, request

    monkeypatch.delenv("TF_PROMPT_SEED", raising=False)
    app = app_for(tmp_path)
    body = {"messages": HI, "temperature": 1.0}
    reported = []
    with http_server(app) as port:
        for _ in range(3):
            status, _, text = request(port, body)
            assert status == 200
            reported.append(json.loads(text)["tensorfold"]["seed"])
    seeds = [call["sampling"].seed for call in app.engine.calls]
    assert len(set(seeds)) == 3 and seeds == reported
    greedy = app_for(tmp_path)
    with http_server(greedy) as port:
        status, _, text = request(port, {"messages": HI, "temperature": 0})
    assert status == 200 and "seed" not in json.loads(text).get("tensorfold", {})
