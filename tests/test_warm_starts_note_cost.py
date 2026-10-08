"""TF_WARM_STARTS: noting a resumed system block again does not rescan, copy and digest it on the decode thread.

Every resume of a kept system-block state calls ``WarmStarts.note`` from the decode thread: it scanned the block for
a second message token by token, copied it and took a SHA-256 of its JSON (about 1.3 ms for a 12K-token block, per
admission). The kept state's ids are the same list on every resume, so that work is done once per list; the
recency update (and the file write it triggers) is unchanged."""

from tensorfold.families.qwen4_exp.cuda import warm_starts as ws

HEAD, OPEN, USER = [1, 3], 1, [1, 4]


def make(tmp_path):
    return ws.WarmStarts(tmp_path / "warm.json", head=HEAD, opener=OPEN, user=USER, fingerprint="tok", vocab=10**6)


def block(n, seed):
    return HEAD + [100 + (seed * 13 + i) % 5000 for i in range(n)]


def test_a_resumed_block_is_scanned_and_digested_once(tmp_path, monkeypatch):
    warm = make(tmp_path)
    keys, scans = [], []
    real_key, real_scan = ws._key, warm.system_only
    monkeypatch.setattr(ws, "_key", lambda ids: keys.append(1) or real_key(ids))
    monkeypatch.setattr(warm, "system_only", lambda ids: scans.append(1) or real_scan(ids))
    a, b = block(3000, 1), block(2000, 2)
    for ids in (a, b, a, a, b):
        warm.note(ids)
    assert len(keys) == 2 and len(scans) == 2
    assert [e["ids"] for e in warm.entries] == [b, a] and [e["uses"] for e in warm.entries] == [2, 3]


def test_a_conversation_is_rejected_once(tmp_path, monkeypatch):
    warm = make(tmp_path)
    scans = []
    real_scan = warm.system_only
    monkeypatch.setattr(warm, "system_only", lambda ids: scans.append(1) or real_scan(ids))
    conversation = block(600, 3) + USER + [7] * 300
    for _ in range(3):
        warm.note(conversation)
    assert len(scans) == 1 and warm.entries == []


def test_an_equal_new_list_is_the_same_block(tmp_path):
    warm = make(tmp_path)
    a = block(1000, 4)
    warm.note(a)
    warm.note(list(a))
    assert len(warm.entries) == 1 and warm.entries[0]["uses"] == 2


def test_the_cache_is_bounded(tmp_path):
    warm = make(tmp_path)
    blocks = [block(400, i) for i in range(10 * warm.limit)]
    for ids in blocks:
        warm.note(ids)
    assert len(warm.known) <= 4 * warm.limit and len(warm.entries) == warm.limit
