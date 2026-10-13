"""Host tests (no GPU) for TF_PIN_REWARM (H2c): once memory pressure evicted the pinned base, the idle scheduler
prefills it again, alone, as the start-up replay does; only when a free slot fits the memory gate (nothing evicted),
once per idle period, never noted as a use, and a request admitted meanwhile takes it out between passes."""

import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from tensorfold.cuda.markers import snapshot_points  # noqa: E402
from tensorfold.cuda.memory_gate import NoRoom  # noqa: E402
from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefixes, rewarm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_fill import PromptPasses  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pin_base_host import ASSISTANT, OPENER, TAIL, Recorder, block, chat  # noqa: E402


class Gate:
    def __init__(self, room):
        self.room, self.asked = room, []

    def fits(self, extra):
        self.asked.append(extra)
        return extra <= self.room


class Slot:
    limit, layers = 262144, 2

    def __init__(self, name, capacity=256):
        self.name, self.capacity = name, capacity

    def layer_bytes(self, rows=None):
        return (self.capacity if rows is None else rows) * 1000

    def cache_bytes(self, rows=None):
        return self.layers * self.layer_bytes(rows)


class Owner:
    """The decoder surface rewarm.py reads (MultiDecoder's attributes of the same names)."""

    def __init__(self, warm, free=(), room=1 << 40):
        self.kept, self.free, self.depth, self.keep = [], list(free), 3, 16
        self.warm_starts, self.pin_base, self.solo = warm, True, None
        self.points = snapshot_points((OPENER,), (OPENER, ASSISTANT), 2048)
        self.points.stable = warm.stable
        self.memory_gate = Gate(room)
        self.streams, self.filling, self.fills, self.passed, self.held = {}, [], {}, {}, {}
        self.rewarm = rewarm.Rewarm(warm, sampling="greedy", idle_s=30)
        self.admitted, self.during, self.shrunk, self.refuse = [], [], [], None
        self.quiet_notes = self.no_evict = False
        self.arrived = lambda: False

    def live(self):
        return len(self.streams) + len(self.filling)

    def _busy(self):
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _shrink(self, st, release=False, force=False):
        self.shrunk.append(st)

    def admit(self, s):
        again = rewarm.before_admit(self, s)
        self.during.append((again, self.quiet_notes, self.no_evict))
        if self.refuse is not None:
            raise self.refuse
        s.sid, s.st = len(self.admitted), self.free.pop()
        self.admitted.append(s)
        self.fills[s.sid] = [None, True, 0, None]
        self.filling.append(s)


def keep(owner, ids, st, checkpoint=False, start=False):
    prefixes.remember(owner, list(ids), st, {"pos": len(ids), "mtp_len": len(ids) - 1}, None,
                      start=start, checkpoint=checkpoint)


class Base(Recorder):
    def family(self):
        w = self.make()
        w.note(block(1))
        w.note(block(1))
        w.note(block(2))
        return w

    def owner(self, **kw):
        return Owner(self.family(), **kw)


class Missing(Base):
    def test_the_base_prompt_is_the_replayed_one(self):
        w = self.family()
        self.assertEqual(w.base_replay(), block(1) + TAIL)
        self.assertEqual(w.base_replay(), w.replays(1, pin=True)[0])
        self.assertIsNone(self.make().base_replay())                       # no family yet

    def test_present_while_a_kept_state_covers_the_checkpoint(self):
        o = self.owner()
        a = block(1)
        keep(o, a[:14336], Slot("base"), checkpoint=True)
        self.assertIsNone(o.rewarm.missing(o))
        o.kept = []
        self.assertEqual(o.rewarm.missing(o), a + TAIL)                    # evicted: the base's replay prompt

    def test_a_sessions_own_checkpoint_counts_its_block_end_does_not(self):
        o = self.owner()
        keep(o, block(4)[:14336], Slot("s4"), checkpoint=True)             # a later session's checkpoint: the head
        self.assertIsNone(o.rewarm.missing(o))
        o.kept = []
        keep(o, block(1), Slot("end"), start=True)                         # past the shared head: no fresh session
        keep(o, block(1)[:12288], Slot("short"))                           # shorter than the checkpoint
        self.assertEqual(o.rewarm.missing(o), block(1) + TAIL)

    def test_no_family_nothing_to_do(self):
        w = self.make()
        w.note(block(1))
        o = Owner(w)
        self.assertIsNone(o.rewarm.missing(o))


class Check(Base):
    def test_waits_for_a_request_then_the_idle_period(self):
        o = self.owner(free=[Slot("a")])
        self.assertIsNone(o.rewarm.wait())                                 # start-up: nothing until a request
        self.assertFalse(rewarm.check(o))
        o.rewarm.arm()
        self.assertEqual(o.rewarm.wait(), 30)

    def test_admits_the_base_alone_without_noting_or_evicting(self):
        o = self.owner(free=[Slot("a")])
        o.rewarm.arm()
        self.assertTrue(rewarm.check(o))
        (s,) = o.admitted
        self.assertEqual(s.prompt, block(1) + TAIL)
        self.assertEqual((s.count, s.draft, s.sampling, s.background), (1, True, "greedy", False))
        self.assertEqual(o.during, [(True, True, True)])                   # itself, quiet, no eviction
        self.assertTrue(o.quiet_notes)
        self.assertFalse(o.no_evict)
        self.assertIs(o.rewarm.stream, s)
        self.assertIsNone(o.rewarm.wait())                                 # once per idle period

    def test_memory_without_room_polls_and_evicts_nothing(self):
        big = Slot("a")
        need = big.cache_bytes(24576) - big.cache_bytes() + big.layer_bytes(24576)
        o = self.owner(free=[big], room=need + rewarm.MARGIN - 1)
        o.rewarm.arm()
        self.assertFalse(rewarm.check(o))
        self.assertEqual(o.memory_gate.asked, [need + rewarm.MARGIN])
        self.assertEqual((o.admitted, o.rewarm.state, o.rewarm.wait()), ([], "poll", rewarm.POLL_S))
        o.memory_gate.room = need + rewarm.MARGIN                          # memory came back
        self.assertTrue(rewarm.check(o))

    def test_a_grown_free_slot_needs_nothing(self):
        o = self.owner(free=[Slot("grown", capacity=32768)], room=rewarm.MARGIN)
        o.rewarm.arm()
        self.assertTrue(rewarm.check(o))

    def test_no_free_slot_polls(self):
        o = self.owner()
        o.rewarm.arm()
        self.assertFalse(rewarm.check(o))
        self.assertEqual(o.rewarm.state, "poll")

    def test_present_disarms(self):
        o = self.owner(free=[Slot("a")])
        keep(o, block(1)[:14336], Slot("base"), checkpoint=True)
        o.rewarm.arm()
        self.assertFalse(rewarm.check(o))
        self.assertIsNone(o.rewarm.state)

    def test_busy_or_a_queued_request_does_nothing(self):
        o = self.owner(free=[Slot("a")])
        o.rewarm.arm()
        o.arrived = lambda: True
        self.assertFalse(rewarm.check(o))
        o.arrived = lambda: False
        o.filling.append(Stream([1, 2], 1))
        self.assertFalse(rewarm.check(o))
        self.assertEqual((o.admitted, o.rewarm.state), ([], "idle"))

    def test_a_refused_growth_polls_again(self):
        o = self.owner(free=[Slot("a")])
        o.rewarm.arm()
        o.refuse = NoRoom("no")
        self.assertFalse(rewarm.check(o))
        self.assertEqual((o.rewarm.stream, o.rewarm.state, o.quiet_notes, o.no_evict), (None, "poll", False, False))

    def test_the_rewarm_block_is_not_noted_as_a_use(self):
        o = self.owner(free=[Slot("a")])
        uses = {e["key"]: e["uses"] for e in o.warm_starts.entries}
        o.quiet_notes = True
        keep(o, block(1), Slot("b"), start=True)
        self.assertEqual({e["key"]: e["uses"] for e in o.warm_starts.entries}, uses)
        o.quiet_notes = False
        keep(o, block(2), Slot("c"), start=True)
        self.assertNotEqual({e["key"]: e["uses"] for e in o.warm_starts.entries}, uses)


class GiveWay(Base):
    def started(self):
        o = self.owner(free=[Slot("spare"), Slot("a")])
        o.rewarm.arm()
        self.assertTrue(rewarm.check(o))
        return o, o.admitted[0]

    def test_a_request_takes_it_out_and_rearms(self):
        o, s = self.started()
        o.fills[s.sid][2] = 8192                                         # two passes in
        req = Stream(chat(block(5)), 8)
        self.assertFalse(rewarm.before_admit(o, req))
        self.assertTrue(s.done)
        self.assertEqual((o.filling, o.fills, o.rewarm.stream, o.quiet_notes), ([], {}, None, False))
        self.assertIn(s.st, o.free)                                      # nothing kept: its slot is free again
        self.assertEqual(o.shrunk, [s.st])
        self.assertEqual((o.rewarm.yielded, o.rewarm.state), (1, "idle"))

    def test_the_checkpoint_it_reached_stays(self):
        o, s = self.started()
        o.fills[s.sid][2] = 14336
        keep(o, s.prompt[:14336], s.st, checkpoint=True)                 # prefilled alone up to here
        rewarm.before_admit(o, Stream(chat(block(5)), 8))
        self.assertNotIn(s.st, o.free)
        self.assertEqual([len(k[0]) for k in o.kept], [14336])
        self.assertIsNone(o.rewarm.missing(o))

    def test_the_fill_loop_gives_way_before_a_pass(self):
        o, s = self.started()
        passes = []
        o._pass = lambda: passes.append(1) or []
        o._waiting_request = lambda: True
        self.assertEqual(PromptPasses._fill(o), [])
        self.assertEqual(passes, [])                                     # not one more pass of the re-warm
        o.rewarm.stream = None                                           # an ordinary prompt: the house loop
        o.streams = {}
        PromptPasses._fill(o)
        self.assertEqual(passes, [1])

    def test_finishing_logs_and_clears(self):
        o, s = self.started()
        o.filling.remove(s)
        keep(o, s.prompt[:14336], s.st, checkpoint=True)
        o.link = None
        s.done = True
        MultiDecoder.finish(o, [s])
        self.assertEqual((o.rewarm.stream, o.rewarm.done, o.quiet_notes), (None, 1, False))

    def test_no_evict_refuses_eviction(self):
        o = self.owner()
        keep(o, [5] * 3000, Slot("conv"))
        o.no_evict = True
        self.assertFalse(MultiDecoder._evict_kept(o, Slot("grower")))
        self.assertEqual(len(o.kept), 1)


class Idle:
    """A decoder whose idle work admits one stream that ends on its next round."""

    def __init__(self, wait):
        self.wait, self.calls, self.streams, self.rounds, self.admitted = wait, [], [], 0, []

    def idle_wait(self):
        return self.wait

    def idle_work(self):
        self.calls.append(time.monotonic())
        if len(self.calls) == 2:
            raise RuntimeError("boom")                   # the worker survives a failing hook
        if len(self.calls) == 3:
            self.streams.append(Stream([1], 1))
            return True
        return False

    def live(self):
        return len(self.streams)

    def admit(self, s):
        self.admitted.append(s)
        s.done = True

    def round(self):
        self.rounds += 1
        done, self.streams = [s for s in self.streams], []
        for s in done:
            s.done = True
        return done

    def finish(self, done):
        pass

    def drop(self):
        return []


class SchedulerHook(unittest.TestCase):
    def test_idle_work_runs_after_the_wait_and_a_request_still_goes_through(self):
        d = Idle(0.02)
        sch = Scheduler(d)
        try:
            deadline = time.monotonic() + 5
            while d.rounds < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(len(d.calls), 3)
            self.assertEqual(d.rounds, 1)                # the admitted idle work ran a round
            out = sch.submit([1, 2, 3], 1, None, True, lambda new: None)
            self.assertIsInstance(out, dict)
            self.assertEqual(len(d.admitted), 1)
        finally:
            sch.close()
        self.assertFalse(sch.thread.is_alive())

    def test_without_a_wait_the_worker_blocks_as_before(self):
        d = Idle(None)
        sch = Scheduler(d)
        time.sleep(0.1)
        self.assertEqual(d.calls, [])
        sch.close()
        self.assertFalse(sch.thread.is_alive())


class Env(unittest.TestCase):
    def test_knobs(self):
        old = {k: os.environ.pop(k, None) for k in (rewarm.ENV, rewarm.IDLE_ENV)}
        try:
            self.assertTrue(rewarm.enabled())
            self.assertEqual(rewarm.idle_seconds(), rewarm.IDLE_S)
            os.environ[rewarm.ENV] = "0"
            self.assertFalse(rewarm.enabled())
            os.environ[rewarm.IDLE_ENV] = "5"
            self.assertEqual(rewarm.idle_seconds(), 5.0)
            for bad in ("0", "x", "nan"):
                os.environ[rewarm.IDLE_ENV] = bad
                with self.assertRaises(ValueError):
                    rewarm.idle_seconds()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
