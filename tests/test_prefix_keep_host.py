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
                prefixes.remember(self, list(prompt[:p]), st, {'pos': p, 'mtp_len': p - 1}, None, start=True)
        end = len(prompt) - 1
        if cached < end:                                  # a resend resumes at its own end: nothing new to keep
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

    def test_resends_and_next_turns_keep_every_prompts_end(self):
        """tests/cuda/test_flashnext_multi.py's packed-passes case on the host: five prompts on five slots, then
        each resent and continued; every resend and next turn resumes one token early, none restarts from 0."""

        owner = Owner(slots=5, keep=8)
        prompts = [[p] * n for p, n in ((1, 20), (2, 30), (3, 4), (4, 70), (5, 12))]
        for p in prompts:
            owner.request(p)
        for p in prompts:
            for q in (p, p[:-1] + [271, 77]):
                self.assertEqual(owner.request(q), len(p) - 1, (p[0], q[-2:]))

    def test_resending_an_end_does_not_age_the_others(self):
        owner = Owner(slots=4, keep=8)
        for tag in (1, 2, 3):
            owner.request([tag] * 300)
        before = [k[0][0] for k in owner.kept]
        owner.request([1] * 300)
        self.assertEqual([k[0][0] for k in owner.kept], before)

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


def conversation(owner, system, user, turns, step=480, tag=0):
    """A conversation's turns on ``system``: each resends the last prompt plus ``step`` tokens; returns cached counts."""

    prompt = list(system) + [tag * 100000 + 50000 + i for i in range(user)]
    cached = []
    for t in range(turns):
        cached.append(owner.request(prompt, points=(len(system),)))
        prompt = prompt + [tag * 100000 + 90000 + t * step + i for i in range(step)]
    return cached, prompt


class PrefixClaimTest(unittest.TestCase):
    """#315 with the house rules: a fork with no free slot claims the idle slot that loses the fewest kept tokens."""

    def interleaved(self, slots):
        owner = Owner(slots=slots)
        other = list(range(500000, 504700))                        # another system block, used first
        conversation(owner, other, 300, 1, tag=3)
        prompts, cached = {}, {1: [], 2: []}
        for t in range(4):
            for tag, user in ((1, 8000), (2, 12000)):             # each longer than the other block
                if t == 0:
                    prompts[tag] = SYSTEM + [tag * 100000 + i for i in range(user)]
                else:
                    prompts[tag] = prompts[tag] + [tag * 100000 + 90000 + t * 480 + i for i in range(480)]
                cached[tag].append((owner.request(prompts[tag], points=(len(SYSTEM),)), len(prompts[tag])))
        return cached

    def test_two_long_conversations_on_one_system_block_resume_their_own_turns(self):
        for slots in (2, 4):
            cached = self.interleaved(slots)
            for tag in (1, 2):
                for (c, _), (_, before) in zip(cached[tag][1:], cached[tag]):
                    self.assertEqual(c, before - 1, (slots, tag))

    def test_a_claim_skips_a_slot_that_keeps_a_system_block(self):
        owner = Owner(slots=3)
        conversation(owner, list(range(600000, 600400)), 100, 1, tag=4)    # slot: its system block + a short end
        conversation(owner, [], 700, 1, tag=5)                             # slot: one end, no system block
        conversation(owner, SYSTEM, 3000, 2)                               # slot: SYSTEM + a long chain
        held = {id(k[1]) for k in owner.kept if prefixes._is_start(owner, k)}
        plain = next(k[1] for k in owner.kept if len(k[0]) == 699)
        st, _, cached = prefixes.slot_for(owner, SYSTEM + [1, 2, 3], True)
        self.assertEqual(cached, len(SYSTEM))
        self.assertIs(st, plain)
        self.assertNotIn(id(st), held)

    def test_a_refused_claim_keeps_the_victims_chains_in_their_places(self):
        owner = Owner(slots=2)
        conversation(owner, list(range(600000, 600400)), 100, 1, tag=4)
        conversation(owner, SYSTEM, 3000, 2)
        before = list(owner.kept)
        owner._grow = lambda st, rows, protect=None: False
        st, _, cached = prefixes.slot_for(owner, SYSTEM + [1, 2, 3], True)
        self.assertEqual(cached, len(SYSTEM))
        self.assertEqual([k for k in owner.kept if k[1] is not st], [k for k in before if k[1] is not st])

    def test_a_fresh_prompt_without_a_free_slot_spares_the_system_blocks_slot(self):
        owner = Owner(slots=2)
        conversation(owner, SYSTEM, 300, 1)                                # oldest: SYSTEM's slot
        conversation(owner, [], 700, 1, tag=5)
        system_slot = next(k[1] for k in owner.kept if len(k[0]) == len(SYSTEM))
        st, resume, cached = prefixes.slot_for(owner, [7] * 500, True)
        self.assertIsNone(resume)
        self.assertIsNot(st, system_slot)
        self.assertTrue(any(len(k[0]) == len(SYSTEM) for k in owner.kept))

    def test_a_claimed_fork_records_its_system_block_for_warm_starts(self):
        noted = []
        owner = Owner(slots=2)
        owner.warm_starts = type('W', (), {'note': lambda self, ids: noted.append(len(ids))})()
        conversation(owner, [], 700, 1, tag=5)
        conversation(owner, SYSTEM, 3000, 2)
        noted.clear()
        st, _, cached = prefixes.slot_for(owner, SYSTEM + [1, 2, 3], True)
        self.assertEqual((cached, noted), (len(SYSTEM), [len(SYSTEM)]))
        self.assertTrue(all(k[1] is not st for k in owner.kept))


if __name__ == '__main__':
    unittest.main()
