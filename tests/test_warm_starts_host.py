"""Host tests (no GPU) for TF_WARM_STARTS: which prefixes are recorded (system blocks only), the 0600 file, its
invalidation, the replay prompts, and that a replay's kept point lands exactly where a chat's message start does."""

import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from tensorfold.families.qwen4_exp.cuda import prefixes  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import warm_starts as ws  # noqa: E402

OPENER, SYS, USER, NL = 248045, 8678, 846, 198
HEAD, TAIL = [OPENER, SYS, NL], [OPENER, USER, NL]
MODEL = Path(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "models/qwen38-flash-next-exl3-4.05bpw"))


def system(n=1200, seed=5):
    return HEAD + [1000 + (seed * 7 + i) % 5000 for i in range(n)]


MADE = []


def settle():
    """Wait until every recorder's writer thread is idle (nothing dirty, no write in progress)."""

    import time
    for w in MADE:
        for _ in range(200):
            with w.saving:
                if not w.dirty:
                    break
            time.sleep(0.01)
    MADE.clear()


def make(path, limit=ws.LIMIT, fingerprint="tok-a"):
    MADE.append(None)
    MADE[-1] = ws.WarmStarts(path, head=HEAD, opener=OPENER, user=TAIL, fingerprint=fingerprint, vocab=250000,
                         limit=limit)
    return MADE[-1]


class Slot:
    def __init__(self, n):
        self.n = n

    def copy_prefix(self, other, rows, mtp_len):
        pass


class Owner:
    def __init__(self, warm, slots=4, keep=8):
        self.kept, self.keep, self.depth, self.warm_starts = [], keep, 6, warm
        self.free = [Slot(i) for i in range(slots)]

    def _busy(self):
        return set()

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, protect=None):
        return True

    def _shrink(self, st, force=False):
        pass


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "state" / "warm-starts.json"

    def tearDown(self):
        settle()
        self.dir.cleanup()

    def test_only_system_blocks_are_recorded(self):
        w = make(self.path)
        self.assertTrue(w.system_only(system()))
        self.assertFalse(w.system_only(TAIL + system()[3:]))               # a prompt that opens with a user turn
        self.assertFalse(w.system_only(system() + TAIL + [5] * 300))       # a later message start: holds a user turn
        self.assertFalse(w.system_only(system(100)))                       # under MIN_GAP: never a kept point
        w.note(TAIL + system()[3:])
        w.note(system() + TAIL + [5] * 300)
        self.assertEqual(w.entries, [])

    def test_file_is_private_and_reloads(self):
        w = make(self.path)
        a, b = system(seed=1), system(seed=2)
        w.note(a)
        w.note(b)
        w.save()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        data = json.loads(self.path.read_text())
        self.assertEqual(set(data), {"version", "tokenizer", "entries"})
        self.assertEqual([e["ids"] for e in data["entries"]], [b, a])    # most recently used first
        again = make(self.path)
        self.assertEqual([e["ids"] for e in again.entries], [b, a])
        os.chmod(self.path, 0o644)                                        # loosened by hand: tightened on load
        make(self.path)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_writer_thread_saves(self):
        w = make(self.path)
        w.note(system())
        for _ in range(100):
            if self.path.exists():
                break
            import time
            time.sleep(0.05)
        self.assertEqual(json.loads(self.path.read_text())["entries"][0]["ids"], system())
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_recency_and_limit(self):
        w = make(self.path, limit=3)
        blocks = [system(seed=i) for i in range(5)]
        for b in blocks:
            w.note(b)
        w.note(blocks[2])                                                 # resumed again: most recent
        self.assertEqual([e["ids"] for e in w.entries], [blocks[2], blocks[4], blocks[3]])
        self.assertEqual(w.entries[0]["uses"], 2)

    def test_invalidation(self):
        w = make(self.path)
        w.note(system())
        w.save()
        self.assertEqual(make(self.path, fingerprint="tok-b").entries, [])   # another tokenizer
        data = json.loads(self.path.read_text())
        data["entries"][0]["ids"][5] += 1                                  # edited ids no longer match their key
        self.path.write_text(json.dumps(data))
        self.assertEqual(make(self.path).entries, [])
        self.path.write_text("not json")
        self.assertEqual(make(self.path).entries, [])
        data["entries"][0]["ids"] = HEAD + [10**9] * 300
        self.path.write_text(json.dumps(data))
        self.assertEqual(make(self.path).entries, [])

    def test_replays_least_recent_first(self):
        w = make(self.path)
        a, b, c = (system(seed=i) for i in range(3))
        for x in (a, b, c):
            w.note(x)
        self.assertEqual(w.replays(2), [b + TAIL, c + TAIL])
        self.assertEqual(w.replays(0), [])
        self.assertEqual(len(w.replays(9)), 3)

    def test_replay_thread_submits_and_survives_errors(self):
        w = make(self.path)
        a, b = system(seed=1), system(seed=2)
        w.note(a)
        w.note(b)
        seen = []

        def submit(p):
            seen.append(p)
            if len(seen) == 1:
                raise RuntimeError("no free stream slot")

        self.assertEqual(ws.replay(w, submit, 2), 1)                      # synchronous: done before serving
        self.assertEqual(seen, [a + TAIL, b + TAIL])
        self.assertEqual(ws.replay(make(Path(self.dir.name) / "none.json"), submit, 2), 0)

    def test_replay_count_env(self):
        os.environ.pop(ws.REPLAY_ENV, None)
        self.assertEqual(ws.replay_count(), ws.REPLAY)
        os.environ[ws.REPLAY_ENV] = "3"
        self.assertEqual(ws.replay_count(), 3)
        os.environ[ws.REPLAY_ENV] = "x"
        self.assertRaises(ValueError, ws.replay_count)
        os.environ.pop(ws.REPLAY_ENV)

    def test_prefix_store_notes_starts_and_resumes(self):
        w = make(self.path)
        owner = Owner(w)
        sys_ids, chat = system(), system() + TAIL + [7] * 40
        st, resume, n = prefixes.slot_for(owner, chat, True)
        self.assertIsNone(resume)
        prefixes.remember(owner, sys_ids, st, {"pos": len(sys_ids), "mtp_len": len(sys_ids) - 1}, None, start=True)
        prefixes.remember(owner, chat[:-1], st, {"pos": len(chat) - 1, "mtp_len": len(chat) - 2}, None)
        self.assertEqual([e["ids"] for e in w.entries], [sys_ids])          # the chat's own end: not recorded
        later = system(seed=9)
        st2, _, _ = prefixes.slot_for(owner, later + TAIL, True)
        prefixes.remember(owner, later, st2, {"pos": len(later), "mtp_len": len(later) - 1}, None, start=True)
        self.assertEqual(w.entries[0]["ids"], later)
        _, resume, n = prefixes.slot_for(owner, sys_ids + TAIL + [8] * 30, True)   # a new chat on the first block
        self.assertEqual(n, len(sys_ids))
        self.assertEqual(w.entries[0]["ids"], sys_ids)                     # resumed: most recent again
        turn3 = chat + [OPENER, 77, NL] + [9] * 400                        # a later start holds the user's turn
        st3, _, _ = prefixes.slot_for(owner, turn3 + TAIL, True)
        prefixes.remember(owner, turn3, st3, {"pos": len(turn3), "mtp_len": len(turn3) - 1}, None, start=True)
        self.assertTrue(all(e["ids"] in (sys_ids, later) for e in w.entries))

    def test_no_recorder_no_change(self):
        owner = Owner(None)
        st, _, _ = prefixes.slot_for(owner, system() + TAIL, True)
        prefixes.remember(owner, system(), st, {}, None, start=True)
        self.assertEqual(len(owner.kept), 1)


@unittest.skipUnless((MODEL / "tokenizer.json").is_file(), "the house checkpoint's tokenizer")
class RealTemplateTests(unittest.TestCase):
    """With the house tokenizer and chat template: the recorded prefix is the system block, and the replay's kept
    point (its only message-start stop) is exactly the position a real chat's first stop is at."""

    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer

        from tensorfold.cuda.chat_template import ChatTemplate
        from tensorfold.cuda.markers import TemplateTokens, resume_points

        cls.t = TemplateTokens(Tokenizer.from_file(str(MODEL / "tokenizer.json")), ChatTemplate(MODEL))
        cls.points = staticmethod(resume_points(MODEL))
        cls.dir = tempfile.TemporaryDirectory()
        os.environ[ws.FILE_ENV] = str(Path(cls.dir.name) / "w.json")
        cls.w = ws.WarmStarts.from_env(MODEL, 250000)
        os.environ.pop(ws.FILE_ENV)

    @classmethod
    def tearDownClass(cls):
        import time
        time.sleep(1.5)                                                     # the writer thread's last save
        cls.dir.cleanup()

    def test_from_env_tokens(self):
        self.assertEqual(self.w.head, HEAD)
        self.assertEqual(self.w.user, TAIL)
        self.assertEqual(self.w.opener, OPENER)
        self.assertIsNone(ws.WarmStarts.from_env(MODEL, 250000))          # TF_WARM_STARTS unset: off

    def test_replay_stop_matches_chat_stop(self):
        tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                                   "parameters": {"type": "object", "properties": {
                                                       "path": {"type": "string"}}}}}]
        sys_msg = {"role": "system", "content": "You are Hermes, a household agent. " * 120}
        for kwargs in ({}, {"tools": tools}):
            chats = [self.t.apply_chat_template([sys_msg, {"role": "user", "content": q}],
                                                add_generation_prompt=True, **kwargs)
                     for q in ("hello there", "What is on my calendar today?")]
            s = self.points(chats[0])[0]
            self.assertEqual(s, self.points(chats[1])[0])
            block = chats[0][:s]
            self.assertEqual(block, chats[1][:s])
            self.assertTrue(self.w.system_only(block))
            self.w.note(block)
            prompt = self.w.replays(1)[0]
            self.assertEqual(prompt[:s], block)
            stops = [p for p in self.points(prompt) if 0 + 256 <= p < max(1, len(prompt) - 1)]
            self.assertEqual(stops, [s])                                    # the replay keeps a state exactly at s
            multi = self.t.apply_chat_template([sys_msg, {"role": "user", "content": "hi " * 300},
                                                {"role": "assistant", "content": "ok " * 300},
                                                {"role": "user", "content": "more"}],
                                               add_generation_prompt=True, **kwargs)
            for p in self.points(multi)[1:]:                                # later stops hold the user's turns
                self.assertFalse(self.w.system_only(multi[:p]))

    def test_user_first_prompt_not_recorded(self):
        ids = self.t.apply_chat_template([{"role": "user", "content": "secret " * 400},
                                          {"role": "assistant", "content": "ok"},
                                          {"role": "user", "content": "and?"}], add_generation_prompt=True)
        for p in self.points(ids):
            self.assertFalse(self.w.system_only(ids[:p]))


if __name__ == "__main__":
    unittest.main()
