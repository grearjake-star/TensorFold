"""--parallel's prompt passes read the n-gram table ahead (MADV_WILLNEED) for the rows after each pass: the windows
asked for, no window twice, and the ids those rows' own gather will look up. Residency only: nothing is returned."""

from types import SimpleNamespace

import numpy as np

from tensorfold.families.qwen4_exp.cuda import decode, multi_fill
from tensorfold.families.qwen4_exp.cuda.ngram import NGram


class Pass(multi_fill.PromptPasses):
    def __init__(self, streams):
        self.w, self.filling = object(), streams


def _stream(sid, n):
    return SimpleNamespace(sid=sid, prompt=list(range(n)))


def test_each_pass_asks_for_the_next_rows_once(monkeypatch):
    asked = []
    monkeypatch.setattr(multi_fill, "ngram_rows_ahead", lambda w, prompt, a, b: asked.append((a, b)) or True)
    s = _stream("a", 10_000)
    f = Pass([s])
    f._read_ahead([(s, 0, 2048)])
    f._read_ahead([(s, 2048, 2048)])
    f._read_ahead([(s, 4096, 256)])               # a short pass beside decoding streams: only the rows not asked for
    f._read_ahead([(s, 4352, 5648)])              # the prompt's last pass: nothing after it
    assert asked == [(2048, 6144), (6144, 8192), (8192, 8448)]


def test_finished_prompts_are_forgotten_and_nothing_asked_without_a_table(monkeypatch):
    asked = []
    monkeypatch.setattr(multi_fill, "ngram_rows_ahead", lambda w, prompt, a, b: asked.append((a, b)) and False)
    s, t = _stream("a", 5000), _stream("b", 3000)
    f = Pass([s, t])
    f._read_ahead([(s, 0, 1000), (t, 0, 1000)])
    assert asked == [(1000, 5000), (1000, 3000)]
    assert f._ahead_to == {}                      # no table read ahead: nothing recorded, the next pass asks again
    f.filling = [t]
    f._read_ahead([(t, 1000, 1000)])
    assert asked[-1] == (2000, 3000)


class Table:
    rows = 1 << 40

    def __init__(self):
        self.asked = []

    def willneed(self, ids):
        self.asked.append(np.asarray(ids).copy())
        return 0


def test_the_rows_ids_are_what_their_gather_looks_up(monkeypatch):
    ngram = NGram(vocab=1000, ngram_size=3, heads_per_ngram=2, vocab_base=5000, divisor=128, shards=1, seed=0,
                  eos=999, embed_dim=64)
    table = Table()
    ple = SimpleNamespace(table=table, ngram=ngram)
    w = SimpleNamespace(x3=object(), layers=[SimpleNamespace(ple=None), SimpleNamespace(ple=ple)])
    monkeypatch.setattr(decode, "_ahead_submit", lambda job: job())
    prompt = [int(x) for x in np.random.default_rng(0).integers(0, 998, 600)]
    assert decode.ngram_rows_ahead(w, prompt, 256, 512)
    want = ngram.ids(np.asarray(prompt[254:256]), np.asarray(prompt[256:512]))      # stage's history: n - 1 rows
    assert len(table.asked) == 1 and np.array_equal(table.asked[0], want)
    assert not decode.ngram_rows_ahead(w, prompt, 1, 100)        # inside the history: not read ahead
    monkeypatch.setattr(decode, "NGRAM_AHEAD", False)
    assert not decode.ngram_rows_ahead(w, prompt, 256, 512)      # TF_NGRAM_AHEAD=0
