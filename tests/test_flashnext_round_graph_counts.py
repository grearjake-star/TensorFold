"""--parallel round graphs count replays, eager rounds, captures and drops per kind, and the done line carries them."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs


class _Graph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _graphs(limit: int, after: int) -> RoundGraphs:
    g = object.__new__(RoundGraphs)                 # no CUDA: the bookkeeping only
    g.limit, g.after = limit, after
    g.graphs, g.seen, g.tables = OrderedDict(), {}, {}
    g.captures = g.replays = g.eager = g.dropped = 0
    g.kinds = {}

    def capture(fn):
        g.captures += 1
        return _Graph(), "out"

    g._capture = capture
    return g


def test_counts_by_kind() -> None:
    g = _graphs(limit=2, after=2)
    ran = []
    fn = lambda: ran.append(1) or "eager"           # noqa: E731
    assert g.run(("main", 2, 6), fn) == "eager"      # first sighting: eager
    assert g.run(("main", 2, 6), fn) == "out"        # second: captured, its replay is the round
    assert g.run(("main", 2, 6), fn) == "out"        # replayed
    g.run(("mtp", 2, 2), fn)
    g.run(("mtp", 2, 2), fn)                         # captured: 2 graphs kept
    g.run(("main", 4, 12), fn)
    g.run(("main", 4, 12), fn)                       # captured: the least recent (main 2/6) is dropped
    assert g.kinds == {"main": [1, 2, 2, 1], "mtp": [0, 1, 1, 0]}
    assert (g.captures, g.eager, g.dropped) == (3, 3, 1)
    assert g.summary() == "main=1/2/2/1 mtp=0/1/1/0 kept=2"


def test_the_done_line_appends_the_counts(capsys) -> None:
    from tensorfold.cuda.server import print_done

    request = SimpleNamespace(first=1.0, started=0.5)
    print_done(10, 0, False, [1, 2, 3], "stop", {"prefill_s": 0.25, "rounds": 2, "accepted": 1, "drafted": 3},
               request, graphs="main=5/1/1/0 mtp=9/2/2/0 kept=3")
    line = capsys.readouterr().out.strip()
    assert line.endswith("accepted=1/3 graphs main=5/1/1/0 mtp=9/2/2/0 kept=3"), line
    print_done(10, 0, False, [1, 2, 3], "stop", {}, request)
    assert "graphs" not in capsys.readouterr().out
