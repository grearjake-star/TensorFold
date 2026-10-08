"""TF_WARM_STARTS under --tp 2 --parallel: only rank 0 replays the recorded system blocks.

Rank 0's replay is an ordinary admission: its plan goes to rank 1 over the link, and rank 1 admits the same prompt
when it follows. A replay of rank 1's own, from its constructor, would run its admission's plan collective while
rank 0 waits for rank 1 to connect the link (a hang), or admit every block a second time once it follows.
"""

from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda import engine as engine_mod
from tensorfold.families.qwen4_exp.cuda import warm_starts as ws


class FakeWarm:
    user = [9, 9]
    entries = [{"ids": [1, 2, 3]}]

    def replays(self, n):
        return [[1, 2, 3] + self.user]


def make(monkeypatch, tp, rank):
    monkeypatch.setattr(ws.WarmStarts, "from_env", classmethod(lambda cls, model_dir, vocab: FakeWarm()))
    e = object.__new__(engine_mod.FlashNextEngine)
    e.tp, e.rank, e.warm_replay = tp, rank, None
    e.multi = SimpleNamespace(warm_starts=None)
    e.submitted = []
    e.scheduler = SimpleNamespace(submit=lambda prompt, *a, **k: e.submitted.append(list(prompt)))
    e._warm_starts("unused", 100)
    return e


def test_rank_one_neither_replays_nor_records(monkeypatch):
    e = make(monkeypatch, tp=2, rank=1)
    assert e.submitted == [] and e.multi.warm_starts is None and e.warm_replay is None


def test_rank_zero_replays_for_both_ranks(monkeypatch):
    e = make(monkeypatch, tp=2, rank=0)
    assert e.submitted == [[1, 2, 3, 9, 9]] and isinstance(e.multi.warm_starts, FakeWarm) and e.warm_replay == 1


def test_one_gpu_replays(monkeypatch):
    e = make(monkeypatch, tp=1, rank=0)
    assert e.submitted == [[1, 2, 3, 9, 9]] and e.warm_replay == 1
