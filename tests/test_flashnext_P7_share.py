"""W5-3 guard: with --decode-share, a round's first pass beside decoding streams is short (it measures a row's time);
later passes follow the share; without a share, or a round estimate once a row is timed, passes stay whole."""

from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda import multi


def _rows(share, round_s, row_s, prefill_rows=2048, live=True):
    d = SimpleNamespace(share=share, round_s=round_s, row_s=row_s, prefill_rows=prefill_rows,
                        streams={0: SimpleNamespace(done=not live)})
    return multi.MultiDecoder._pass_rows(d)


def test_first_pass_is_short_then_share_sized():
    assert _rows(0.25, 0.05, None) == multi.FIRST_PASS
    assert _rows(0.25, 0.05, None, prefill_rows=128) == 128
    assert _rows(0.25, 0.05, 0.0005) == 384                   # 0.05 / (0.25 * 0.0005) = 400 -> 384 (64-row steps)
    assert _rows(0.25, 0.05, 1e-6) == 2048
    assert _rows(0.25, 0.05, 1.0) == multi.PASS_MIN


def test_whole_passes_without_share_or_round_time():
    assert _rows(0.0, 0.05, None) == 2048
    assert _rows(0.25, None, 0.0005) == 2048                  # no round time to size by
    assert _rows(0.25, None, None) == multi.FIRST_PASS       # the first pass beside a stream is short regardless
    assert _rows(0.25, 0.05, None, live=False) == 2048       # no live stream: nothing to stall
