"""Host tests (no GPU) for TF_PIN_BASE (H2): the most used system block's family (Hermes' base prompt, whose memory
and environment lines at its end change from session to session) gets its checkpoint inside the head its sessions
share, stays in the warm-start file and is replayed last, is copied out of on a resume, and is freed last under
memory pressure."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from tensorfold.cuda.markers import snapshot_points  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefixes  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import warm_starts as ws  # noqa: E402

OPENER, SYS, USER, NL, ASSISTANT = 248045, 8678, 846, 198, 77091
HEAD, TAIL = [OPENER, SYS, NL], [OPENER, USER, NL]
SHARED = 14604                                    # the measured head two Hermes sessions' blocks share (2026-10-12)


def block(tail_seed: int, shared: int = SHARED, tail: int = 2500) -> list[int]:
    """A Hermes-shaped system block: a shared head (tools, then system text), then a session's own tail."""

    return HEAD + [1000 + i % 5000 for i in range(shared - len(HEAD))] + \
        [60000 + (tail_seed * 31 + i) % 900 for i in range(tail)]


def chat(sys_block: list[int], user: int = 300) -> list[int]:
    return sys_block + TAIL + [7] * user + [OPENER, ASSISTANT, NL]


class Recorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.made = []

    def tearDown(self):
        for w in self.made:                       # let the writer thread finish before the directory goes
            with w.saving:
                pass
        self.tmp.cleanup()

    def make(self, limit=ws.LIMIT):
        w = ws.WarmStarts(Path(self.tmp.name) / "w.json", head=HEAD, opener=OPENER, user=TAIL, fingerprint="t",
                          vocab=250000, limit=limit)
        self.made.append(w)
        return w


class StableHead(Recorder):
    def test_one_block_says_nothing_two_say_where_sessions_differ(self):
        w = self.make()
        a, b = block(1), block(2)
        w.note(a)
        self.assertIsNone(w.stable(chat(a)))                 # one block: no sibling, no shared length yet
        w.note(b)
        self.assertEqual(w.stable(chat(a)), SHARED)
        self.assertEqual(w.stable(chat(block(3))), SHARED)   # a new session of the family
        self.assertIsNone(w.stable(chat([OPENER, SYS, NL] + [9] * 5000)))      # another client
        self.assertTrue(w.family(a[:14336]) and w.family(a) and not w.family(a[:2000]))

    def test_an_unrelated_block_is_not_a_sibling(self):
        w = self.make()
        a = block(1)
        for _ in range(3):
            w.note(a)                                        # the top block
        w.note(HEAD + [1000 + i % 5000 for i in range(1927)] + [3] * 12000)   # shares 1,930 tokens (another profile)
        self.assertIsNone(w.stable(chat(a)))

    def test_the_most_used_block_stays_in_the_file(self):
        w = self.make(limit=4)
        top = block(1)
        for _ in range(5):
            w.note(top)
        w.note(block(2))
        for seed in range(10, 14):                           # four other clients' blocks, each newer
            w.note(HEAD + [seed] * (3000 + seed))
        keys = [e["key"] for e in w.entries]
        self.assertEqual(len(keys), 4)
        self.assertIn(ws._key(top), keys)
        self.assertEqual(w.replays(2, pin=True)[-1], top + TAIL)                # replayed, last
        self.assertNotIn(top + TAIL, w.replays(2))                               # the old rule: most recent only
        self.assertEqual(len(w.replays(4, pin=True)), 4)
        self.assertEqual(w.replays(1, pin=True), [top + TAIL])

    def test_reload_keeps_the_shared_length(self):
        w = self.make()
        w.note(block(1))
        w.note(block(2))
        w.save()
        again = self.make()
        self.assertEqual(again.stable(chat(block(5))), SHARED)

    def test_resuming_the_top_block_does_not_rescan(self):
        w = self.make()
        w.note(block(1))
        w.note(block(2))
        calls = []
        real = w._refresh
        w._refresh = lambda: (calls.append(1), real())
        ids = block(1)
        for _ in range(20):
            w.note(ids)                                      # a resumed start is noted on every resume
        self.assertLessEqual(len(calls), 1)                  # at most once, when it became the top


class Checkpoint(Recorder):
    def test_the_checkpoint_falls_inside_the_shared_head(self):
        os.environ.pop("TF_SYS_CHECKPOINT", None)
        points = snapshot_points((OPENER,), (OPENER, ASSISTANT), 2048)
        a, b = chat(block(1)), chat(block(2))
        self.assertEqual(points.checkpoint(a), 16384)        # today: past the 14,604 shared tokens
        self.assertNotEqual(a[:16384], b[:16384])
        w = self.make()
        w.note(block(1))
        w.note(block(2))
        points.stable = w.stable
        self.assertEqual(points.checkpoint(a), points.checkpoint(b))
        self.assertEqual(points.checkpoint(a), 14336)
        self.assertEqual(a[:14336], b[:14336])
        self.assertIn(14336, points(a))
        other = chat(HEAD + [9] * 6000)                      # another client's block: as before
        self.assertEqual(points.checkpoint(other), 4096)
        short = chat(block(1, shared=5000, tail=200))         # not the family: as before
        self.assertEqual(points.checkpoint(short), 4096)


class Slot:
    def __init__(self, n):
        self.name, self.copied = n, None

    def copy_prefix(self, other, rows, mtp_len):
        self.copied = (other, rows)


class Owner:
    def __init__(self, warm, pin=True, free=()):
        self.kept, self.free, self.depth, self.keep, self.busy = [], list(free), 3, 16, set()
        self.warm_starts, self.pin_base, self.solo, self.shrunk = warm, pin, None, []

    def _busy(self):
        return {id(s) for s in self.busy}

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, protect=None):
        return True

    def _shrink(self, st, release=False, force=False):
        self.shrunk.append(st)


def kept(ids, st):
    return (list(ids), st, {"pos": len(ids), "mtp_len": len(ids) - 1}, None)


class Pin(Recorder):
    def family(self):
        w = self.make()
        w.note(block(1))
        w.note(block(2))
        return w

    def base_slot(self, owner):
        base = Slot("base")
        a = block(1)
        prefixes.remember(owner, a[:14336], base, {"pos": 14336, "mtp_len": 14335}, None, checkpoint=True)
        prefixes.remember(owner, a, base, {"pos": len(a), "mtp_len": len(a) - 1}, None, start=True)
        return base

    def test_a_new_session_copies_out_of_the_base(self):
        spare = Slot("spare")
        owner = Owner(self.family(), free=[spare])
        base = self.base_slot(owner)
        st, resume, n = prefixes.slot_for(owner, chat(block(3)), True)
        self.assertIs(st, spare)
        self.assertEqual((n, spare.copied), (14336, (base, 14336)))
        self.assertEqual(sum(k[1] is base for k in owner.kept), 2)          # the base slot keeps both states

    def test_without_the_pin_a_session_resumes_in_place(self):
        spare = Slot("spare")
        owner = Owner(self.family(), pin=False, free=[spare])
        base = self.base_slot(owner)
        st, _, n = prefixes.slot_for(owner, chat(block(1)), True)           # the exact block: in place, as today
        self.assertIs(st, base)
        self.assertEqual(n, len(block(1)))

    def test_with_the_pin_the_exact_block_copies_out_too(self):
        spare = Slot("spare")
        owner = Owner(self.family(), free=[spare])
        base = self.base_slot(owner)
        st, _, n = prefixes.slot_for(owner, chat(block(1)), True)
        self.assertIs(st, spare)                              # the base slot stays the base
        self.assertEqual((n, spare.copied), (len(block(1)), (base, len(block(1)))))

    def test_no_spare_slot_resumes_in_place(self):
        owner = Owner(self.family())
        base = self.base_slot(owner)
        st, _, n = prefixes.slot_for(owner, chat(block(1)), True)
        self.assertIs(st, base)
        self.assertEqual(n, len(block(1)))

    def test_memory_pressure_frees_the_base_last(self):
        from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

        owner = Owner(self.family())
        base = self.base_slot(owner)                          # oldest: the first one freed today
        conv, grower = Slot("conv"), Slot("grower")
        prefixes.remember(owner, [5] * 30000, conv, {"pos": 30000, "mtp_len": 29999}, None)
        self.assertTrue(MultiDecoder._evict_kept(owner, grower))
        self.assertEqual([k[1] for k in owner.kept], [base, base])         # the conversation went first
        self.assertTrue(MultiDecoder._evict_kept(owner, grower))           # then the base, rather than fail
        self.assertEqual(owner.kept, [])
        owner = Owner(self.family(), pin=False)
        base = self.base_slot(owner)
        prefixes.remember(owner, [5] * 30000, conv, {"pos": 30000, "mtp_len": 29999}, None)
        MultiDecoder._evict_kept(owner, grower)
        self.assertEqual([k[1] for k in owner.kept], [conv])               # without the pin: oldest first


if __name__ == "__main__":
    unittest.main()
