"""Host tests (no GPU) for Flash Next's kept-prefix policy under --parallel: a shared system block (the second
message's start, kept as a message-start state since 0.6.3) must outlive the agent sessions that fork from it.

Run against a tree with TF_PREFIX_TEST_SRC=<tree>/src (default: this repo's src)."""

import ast
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(os.environ.get('TF_PREFIX_TEST_SRC', Path(__file__).resolve().parents[1] / 'src'))


def load_source(name, path):
    module = types.ModuleType(name)
    sys.modules[name] = module
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not node.level]
    exec(compile(tree, str(ROOT / path), 'exec'), module.__dict__)
    return module


prefixes = load_source('prefix_keep_host_prefixes', 'tensorfold/families/qwen4_exp/cuda/prefixes.py')

MIN_GAP = 256
SYSTEM = list(range(1000, 1000 + 11200))          # tools + Hermes's system prompt, identical across sessions


class Slot:
    def __init__(self, n):
        self.n = n

    def copy_prefix(self, other, rows, mtp_len):
        pass

    def __repr__(self):
        return f'slot{self.n}'


class Owner:
    """The decoder surface prefixes.py reads; requests run one at a time, as a single household user's do."""

    def __init__(self, slots=4, keep=8):
        self.kept, self.keep, self.depth = [], keep, 6
        self.free = [Slot(i) for i in range(slots)]
        self.busy = set()

    def _busy(self):
        return set(self.busy)

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, protect=None):
        return True

    def _shrink(self, st, force=False):
        pass

    def request(self, prompt, points=()):
        """Admit, keep message-start states past the resume point and the end state (one token early), finish."""

        st, resume, cached = prefixes.slot_for(self, list(prompt), True)
        self.busy.add(id(st))
        for p in points:
            if cached + MIN_GAP <= p < len(prompt) - 1:
                prefixes.remember(self, list(prompt[:p]), st, {'pos': p, 'mtp_len': p - 1}, None)
        end = len(prompt) - 1
        prefixes.remember(self, list(prompt[:end]), st, {'pos': end, 'mtp_len': end - 1}, None)
        self.busy.discard(id(st))
        if all(k[1] is not st for k in self.kept) and all(f is not st for f in self.free):
            self.free.append(st)
        return cached


def session(owner, tag, calls):
    """One Hermes chat: a first request (system + user), then ``calls`` tool-call turns, and the title request
    (a short separate prompt). Returns the first request's cached tokens."""

    user = [7, tag, tag, tag] + [tag] * 40
    prompt = SYSTEM + user
    first = owner.request(prompt, points=(len(SYSTEM),))
    for i in range(calls):
        prompt = prompt + [8, tag, i] + [tag + i] * 120          # assistant tool call + tool result
        owner.request(prompt, points=(len(SYSTEM),))
    owner.request([9] * 40 + [tag] * 230)                          # session title: no message-start state
    return first


class PrefixKeepTest(unittest.TestCase):
    def test_second_chat_resumes_the_system_block(self):
        owner = Owner()
        self.assertEqual(session(owner, 1, 0), 0)
        self.assertEqual(session(owner, 2, 0), len(SYSTEM))

    def test_system_block_outlives_a_long_agent_session(self):
        """A 12-call agent session must not push the shared system block out of the 8 kept states."""

        owner = Owner()
        session(owner, 1, 12)
        self.assertEqual(session(owner, 2, 3), len(SYSTEM))
        self.assertEqual(session(owner, 3, 20), len(SYSTEM))
        self.assertEqual(session(owner, 4, 0), len(SYSTEM))

    def test_a_resumed_conversation_keeps_its_newest_end(self):
        owner = Owner()
        session(owner, 1, 12)
        ends = [k for k in owner.kept if len(k[0]) > len(SYSTEM)]
        newest = max(len(k[0]) for k in ends)
        self.assertTrue(any(len(k[0]) == newest for k in owner.kept))
        self.assertLessEqual(len(owner.kept), owner.keep)

    def test_fork_source_becomes_newest(self):
        owner = Owner()
        session(owner, 1, 2)
        session(owner, 2, 0)
        system = next(k for k in owner.kept if len(k[0]) == len(SYSTEM))
        position = owner.kept.index(system)
        self.assertGreater(position, 0)

    def test_without_middles_the_oldest_goes(self):
        owner = Owner(slots=8, keep=3)
        for tag in (1, 2, 3, 4):
            owner.request([tag] * 400)
        self.assertEqual([k[0][0] for k in owner.kept], [2, 3, 4])

    def test_displaced_idle_slot_returns_to_the_free_list(self):
        owner = Owner(slots=2, keep=1)
        owner.request([1] * 400)
        owner.request([2] * 400)
        self.assertEqual(len(owner.kept), 1)
        self.assertEqual(len(owner.free), 1)


if __name__ == '__main__':
    unittest.main()
