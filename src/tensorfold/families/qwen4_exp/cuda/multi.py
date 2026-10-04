"""Flash Next's shared rounds on one GPU or two ranks, with exact prompt pieces and growing caches."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.capacity import LIMIT_ENV, available_bytes, cuda_limit_bytes
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import (PREFILL_ROWS, WARM_TAIL, Engine, _gathered_fits, cost_bars, choose_gathered_streams,
                     entry_end, prefill_begin, tp_sample_rows)
from . import attn_multi, gdn_multi, heads, image_rows, prefixes
from .draft_cost import joint_bars, joint_settings
from .multi_graphs import RoundGraphs, bucket, fold
from .forward import commit, compute, compute_mixed, converges, stage
from .mtp import mtp_compute, mtp_stage
from .state import Buffers, State
from .multi_solo import Alone, solo
from .multi_fill import FILL_GUARD, FIRST_PASS, PASS_MIN, PromptPasses  # noqa: F401 (re-exported)
from .multi_tp import Link as Link
from .multi_tp import OutOfStep, TwoRanks
from ..cuda import CONFIDENCE, DEPTH, env_int

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.stops = ()
    return e


class MultiDecoder(TwoRanks, Alone, PromptPasses):
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    warm_starts = None               # TF_WARM_STARTS: records kept system blocks (the engine sets it)
    release = None                   # TF_NGRAM_LOCK: munlocks table runs for growing caches (the engine sets it)
    released = 0

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, graphs: bool = True, vision=None, workspace_bytes: int = 0,
                 cost: float = 0.0, timing=None, round_graphs: bool = False) -> None:
        self.link = self.follower = None
        self.planning, self.pass_plan, self.mixed_plan = False, None, None
        self.pass_index, self.pass_width = 0, prefill_rows
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.cost, self.timing = cost, timing            # the expected-time stop (decode.cost_bars), as one stream's
        self.joint = joint_settings(os.environ)          # house: TF_JOINT_PRICE=1 prices shared rounds' drafts jointly
        if self.joint is not None and cost > 0:
            rate = self.joint["rate"]
            print(f"[tensorfold] joint draft pricing: on (shared rounds; aggregate rate "
                  f"{f'{rate} tokens/ms' if rate else 'from the verify table'}, "
                  f"{'TF_JOINT_VERIFY_MS' if self.joint['verify'] else 'the one-stream'} row costs)", flush=True)
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        # slots start small and grow with their stream's context, up to the window, while the gate has room
        self.free = [State(w, min(capacity, FIRST), depth + 1, kv_dtype, limit=capacity) for _ in range(slots)]
        self.slots = list(self.free)
        self.solo = (solo(w, self.free[0], capacity, depth, self.pbuf, timing)
                     if graphs and depth > 0 and self.mbuf is not None else None)
        self.solo_on = self.solo is not None
        self.solo_rounds = self.solo_moves = 0          # house: graph-slot rounds and moves (tests count them)
        # ``round_graphs``: shared rounds (any streams, nothing filling beside them) and draft steps replay CUDA graphs
        self.rounds = (RoundGraphs(w, depth) if round_graphs and w.comm is None and torch.cuda.is_available() and
                       all(getattr(layer.moe.experts, "capturable", True) is not False for layer in w.layers)
                       else None)
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        self.memory_gate = MemoryGate(
            live() if live is not None else 1 << 62, reserve=max(2 * GIB, workspace_bytes), live=live)
        # pinned n-gram pages a growing cache may take back (TF_NGRAM_LOCK): ``release(bytes) -> bytes`` munlocks
        # table runs (the engine sets it); a stream never waits or ends while pinned pages could be released
        self.release = None
        self.released = 0
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        self.fills: dict[int, list] = {}                 # stream id -> [its engine, drafts?, next row, kept state]
        self.passed = {}
        self.arrived = lambda: False
        self.fill_yield = False
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches while the gate has room; one stream may use its startup allowance within an explicit cap."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        if self._is_solo(st):                    # each resize recaptures its graphs: double, so they rarely do
            size = min(st.limit, max(size, 1 << (st.capacity - 1).bit_length() + 1))
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if self._release_pinned(grow + st.layer_bytes(size)):
                continue
            if not self._evict_kept(st, protect=protect):
                if alone:
                    limit = cuda_limit_bytes() if torch.cuda.is_available() else None
                    if limit is not None:
                        # A lone stream may use its startup reserve without exceeding the copy peak cap.
                        peak = int(torch.cuda.memory_allocated()) + grow + st.layer_bytes(st.capacity)
                        if peak > limit:
                            if self.filling:
                                return False             # a filling request can finish and release its slot
                            raise NoRoom(
                                f"Growing this request's attention caches to {size} tokens would exceed "
                                f"{LIMIT_ENV}={limit / GIB:g}: the estimated copy peak is {peak / GIB:.2f} GiB. "
                                "Shorten the prompt or max_tokens, reduce --context or --parallel, or raise "
                                f"{LIMIT_ENV} if more GPU memory is available.")
                    break
                return False
        self._state_changed(st)
        try:
            added = st.resize(size)
        except Exception:
            self.memory_gate.take(st.cache_bytes() - before)
            raise
        self.memory_gate.take(added)
        if self.planning and alone:
            self.actions[-1].append("alone")
        if not self.planning and torch.cuda.is_available():
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _is_solo(self, st: State) -> bool:
        if self.solo is None:
            return False
        solo = self.solo.st                      # while planning both are Shadows of the real slots: compare those
        return getattr(st, "source", st) is getattr(solo, "source", solo)

    def planned_growth(self) -> int:
        """What the slots' caches take from their first rows to one growth step each (TF_NGRAM_LOCK=auto leaves it)."""

        st = self.free[0] if self.free else None
        if st is None:
            return 0
        size = min(st.limit, STEP)
        return len(self.free) * max(0, st.cache_bytes(size) - st.cache_bytes(min(st.limit, FIRST))) + st.layer_bytes(size)

    def _release_pinned(self, extra: int) -> bool:
        """Unpin n-gram table memory so ``extra`` cache bytes fit the host check; False when nothing was released or
        when the gate's own room or the explicit cap (not the host) is what refuses."""

        gate = self.memory_gate
        if self.release is None or gate.held + int(extra) > gate.room - gate.reserve:
            return False
        limit = cuda_limit_bytes() if torch.cuda.is_available() else None
        if limit is not None and limit - int(torch.cuda.memory_allocated()) < int(extra) + gate.reserve:
            return False
        short = int(extra) + gate.reserve - (gate.live() if gate.live is not None else 0)
        freed = self.release(max(short, 1))
        if freed > 0:
            self.released += freed
            print(f"[tensorfold] memory: unpinned {freed / GIB:.2f} GiB of the n-gram table for stream caches "
                  f"({self.released / GIB:.2f} GiB so far)", flush=True)
        return freed > 0

    def _shrink(self, st: State, *, release: bool = False, force: bool = False) -> None:
        """Return idle caches to the gate, retaining the graph slot unless memory or cleanup requires its release."""

        st.reset(self.w)
        if force or st.capacity > FIRST and (release or not self._is_solo(st)):
            self._state_changed(st)
            self.memory_gate.give(-st.resize(min(FIRST, st.limit)))

    def _evict_kept(self, keep: State, *, protect: State | None = None) -> bool:
        """Free the oldest idle kept prompt end (never ``keep``); False when none is left."""

        busy = self._busy()
        for ids, st, _, _ in self.kept:
            if st is not keep and st is not protect and id(st) not in busy:
                self._drop_kept(st)
                self._shrink(st, release=True)
                if all(f is not st for f in self.free):
                    self.free.append(st)
                return True
        solo = None if self.solo is None else self.solo.st
        if (solo is not None and solo is not keep and solo is not protect
                and id(solo) not in busy and solo.capacity > FIRST):
            self._shrink(solo, release=True)
            return True
        return False

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        blocked, ended = False, []
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            try:
                s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=len(live) == 1)
            except NoRoom as exc:
                s.error, s.done, s.waiting = exc, True, False
                self.held.pop(s.sid, None)
                self._drop_kept(s.st)
                self.memory_gate.ends += 1
                ended.append(s)                          # finish() releases only this request's slot
                continue
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st, release=True)
            self.free.append(newest.st)
            return [*ended, newest, *self._make_room()]
        for s in live:
            if s.waiting:
                self._flush(s)                 # shared scratch will be reused while this stream waits
        self.memory_gate.waits += any(s.waiting for s in live)
        return ended

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse kept prefixes, preferring the graph slot among otherwise free destinations."""

        if self.solo is not None and any(st is self.solo.st for st in self.free):
            self.free = [st for st in self.free if st is not self.solo.st] + [self.solo.st]
        st, resume, n = prefixes.slot_for(self, prompt, reuse)
        if resume is None and self.solo is not None and self.w.comm is None and not self.planning:
            st = self._fresh_slot(st)            # house: a fresh prompt decodes in the graph slot when it can take it
        return st, resume, n

    def _remember(self, ids: list[int], st: State, snap: dict, tail, start: bool = False) -> None:
        prefixes.remember(self, ids, st, snap, tail, start=start)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def warm(self) -> None:
        """Warm a full prompt chunk, a cut partial chunk, drafts and a shared round, then discard their state."""

        self.solo_on = False
        try:
            s = Stream([0] * min(self.prefill_rows + WARM_TAIL + 1, self.capacity - self.depth - 2), 2)
            self.admit(s)
            if not s.done:
                self.round()                                 # the whole prompt (nothing else decodes), then a round
            self.streams.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            if all(f is not s.st for f in self.free):
                self.free.append(s.st)
        finally:
            self.solo_on = self.solo is not None
        if self.solo is not None:                    # last: the graph slot's rows are the ones requests will find
            st = self.solo.st
            # house (W2): TF_SOLO_ROWS pre-sizes the graph slot so a growing conversation never resizes it (a resize
            # drops every graph) below that many rows, and captures every context bucket up to it now, not mid-reply
            want = env_int("TF_SOLO_ROWS", 0, 0)
            if want > st.capacity and self.w.comm is None:
                self._grow(st, min(want, st.limit), alone=True)
            starts = [0]
            if want:
                b = 16384
                while b <= st.capacity:              # a window ending in (b/2, b] uses bucket b (graphs._bucket)
                    starts.append(b // 2)
                    b *= 2
            captured = 0
            for pos in starts:
                st.set_pos(pos)
                st.set_mtp_len(pos)
                captured += self.solo.graphs.warm(self.depth + 1)
            st.reset(self.w)                         # the captures wrote its state
            if want:
                print(f"[tensorfold] lone-stream graph slot: {st.capacity} rows, {captured} graphs captured over "
                      f"{len(starts)} context buckets", flush=True)

    @torch.no_grad()
    def admit(self, s: Stream, told=None) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        if self.w.comm is not None and any(x is not None for x in (s.constraint, s.vision, s.probabilities)):
            raise ValueError("concurrent Flash Next on two ranks serves text without grammars or logprobs")
        t0 = time.perf_counter()
        if self.w.comm is not None:
            st, resume, s.cached = self._prepare_admission(s, told)
        else:
            st, resume, s.cached = self._slot_for(list(s.prompt), s.draft and s.vision is None)
        if self.w.comm is None:
            try:
                if not self._grow(st, len(s.prompt) + self.depth + 2,
                                  alone=not self.streams and not self.filling):
                    raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
            except NoRoom:
                if resume is None:
                    self.free.append(st)
                else:
                    self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
                raise
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
            image_rows.begin(e, s, self.vision)
        except Exception:
            self._drop_kept(st)
            self.free.append(st)
            raise
        if self.w.comm is not None:
            e.stops = list(s.stops)
        else:
            e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
                if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        s.prefill_s = time.perf_counter() - t0
        same = resume is not None and self._keep_at(s) == begin         # the same prompt again: its own point
        self.fills[s.sid] = [e, mtp, begin, (resume["state"], resume["tail"]) if same else None]
        self.filling.append(s)

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self, told=None) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        if self.w.comm is not None:
            ended, solo_sid = self._prepare_round(told)
        else:
            ended, solo_sid = self._make_room(), None
            self.pass_plan = self.mixed_plan = None
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if not live:
            return ended
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed + ended
        use_solo = solo_sid is not None if self.w.comm is not None else (
            self.solo_on and not ended and not self.filling and len(self.streams) == 1
            and len(live) == 1 and live[0].draft and live[0].constraint is None
            and live[0].st.image_positions is None)
        if use_solo:
            s = live[0]
            if self.w.comm is None:
                self._move_to_solo(s) if s.st is not self.solo.st else self._flush(s)
            return failed + ended + self._solo_round(s)
        t0 = time.perf_counter()
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        width = self.pass_width if self.w.comm is not None else self._pass_rows()
        pieces, psegs = (self._pieces(width) if self.filling and self.converged else []), None
        if pieces and self.mixed_plan is not None:
            if [[s.sid, a, n] for s, a, n in pieces] != self.mixed_plan[self.pass_index]:
                raise OutOfStep("mixed prompt pieces differ from the agreed round")
        cuts = []
        if pieces:
            self._note_passed(pieces)
            try:
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                self._read_ahead(pieces)
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        graphed = self.rounds is not None and not pieces and all(s.st.image_positions is None for s in live)
        if graphed:                                    # persistent tables, launches bounded by the context bucket
            n, rows = len(segs), segs[-1][2]
            ctx = bucket(max(st.pos + a1 - a0 for st, a0, a1 in segs))
            # last round's kept rows folded first (multi_solo._flush's replay), so the graph's forward is idempotent
            fold(self.w, self.gdn, [s.st for s in live], held)
            tables = self.buf.gdn_tables = gdn_multi.Tables(self.w, self.gdn, segs, [[] for _ in live],
                                                            out=self.rounds.gdn_out(n, rows), width=self.depth + 1)
            self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False, out=self.rounds.attn_out(False, n, rows),
                                                 bucket=ctx)
        else:
            tables = self.buf.gdn_tables = gdn_multi.Tables(self.w, self.gdn, segs, held)
            self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False)
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
                candidates = self._prompt_candidates(len(pends))
            elif graphed:
                logits = self.rounds.run(("main", n, rows, tables.cur, ctx),
                                         lambda: compute(self.w, segs, self.buf))
            else:
                logits = compute(self.w, segs, self.buf)
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        samplings = [s.sampling for s in live]
        if self.w.comm is None:
            sampled = sample_streams(logits, starts, positions, samplings)
        elif all(_gathered_fits(smp) for smp in samplings):
            sampled = choose_gathered_streams(self.w, self.buf.cand_all, starts[-1], starts, positions, samplings)
        else:
            sampled = [tp_sample_rows(self.w, logits[a0:a1], pos, smp, offset=int(self.w.meta["vocab_offset"]))
                       for (_, a0, a1), pos, smp in zip(segs, positions, samplings)]
        paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                 for s, (_, tokens), rows in zip(live, windows, sampled)]
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None:
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
            self.held[s.sid] = rows                      # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last],
                        rows=sum(1 for *_, last in kept if not last))
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces), candidates)
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _draft_all(self, streams: list, rows: int | None = None) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth.

        ``rows``: the streams the next round verifies (each its pending row), drafting or not; with joint pricing
        (TF_JOINT_PRICE=1) a stream's next draft is priced at that round's marginal row and aggregate rate
        (draft_cost.JointBars), else at one stream's bars. Speed only: drafts never change the output."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        logits, prevs = self._mtp_windows([(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo])
        for s, _, keep in todo:
            s.st.set_mtp_len(s.st.mtp_len + len(keep))
        active = [(s, prev) for (s, _, _), prev in zip(todo, prevs)]
        bars = joint = None
        if self.cost > 0 and getattr(self, "joint", None) is not None:
            n = max(len(streams), rows or 0)
            joint = joint_bars(self.joint, self.cost, self.timing, n, n)
        elif self.cost > 0:
            bars = cost_bars(self.cost, self.depth, self.timing)
        chain = {s.sid: 1.0 for s, _, _ in todo}
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt = []
            for (s, prev), (d, p) in zip(active, picks):
                chain[s.sid] *= p
                c = chain[s.sid]
                bar = joint.bar() if joint is not None else bars[j] if bars is not None else None
                low = (self.confidence > 0 and p < self.confidence) or (bar is not None and c < bar)
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if joint is not None:
                    joint.take()                     # the round verifies one more row
                after = joint.bar() if joint is not None else bars[j + 1] if bars is not None else None
                if not low and j + 1 < room[s.sid] and not (after is not None and c < after):
                    nxt.append((s, prev, d))
            if not nxt:
                return
            logits, prevs = self._mtp_windows([(s.st, [d], prev) for s, prev, d in nxt])
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, prev) for (s, _, _), prev in zip(nxt, prevs)]

    def _mtp_windows(self, windows: list) -> tuple[torch.Tensor, list]:
        """One MTP step over ``windows`` ((state, tokens, input streams), ...): each window's last-row draft logits
        (one row a window, in order) and its output stream row, the next step's input. House: streams on different
        MTP heads (TF_HEAD_SWITCH_ROWS) take one launch a head; one head is today's single launch."""

        parts = heads.groups([st for st, _, _ in windows])
        if len(parts) == 1:
            segs = mtp_stage(self.w, self.mbuf, windows)
            logits = self._mtp(segs)
            return logits, [self.mbuf.streams[a1 - 1:a1] for _, _, a1 in segs]
        # each launch overwrites the MTP buffers: inputs and results are copied out of them
        windows = [(st, tokens, x.clone()) for st, tokens, x in windows]
        rows: list = [None] * len(windows)
        prevs: list = [None] * len(windows)
        for idx in parts:
            segs = mtp_stage(self.w, self.mbuf, [windows[i] for i in idx])
            out = self._mtp(segs)
            for k, (i, (_, _, a1)) in enumerate(zip(idx, segs)):
                rows[i] = out[k:k + 1].clone()
                prevs[i] = self.mbuf.streams[a1 - 1:a1].clone()
        return torch.cat(rows), prevs

    def _mtp(self, segs: list) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them."""

        if self.rounds is not None and all(st.image_positions is None for st, _, _ in segs):
            n, rows = len(segs), segs[-1][2]
            ctx = bucket(max(st.mtp_len + a1 - a0 for st, a0, a1 in segs))
            self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True, out=self.rounds.attn_out(True, n, rows),
                                                  bucket=ctx)
            try:
                return self.rounds.run(("mtp", n, rows, ctx) + heads.launch_key(segs),
                                       lambda: mtp_compute(self.w, segs, self.mbuf))
            finally:
                self.mbuf.attn_step = None
        self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        try:
            return mtp_compute(self.w, segs, self.mbuf)
        finally:
            self.mbuf.attn_step = None

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        if self.w.comm is not None:
            return self._picks_tp(logits, positions, samplings)
        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def _picks_tp(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Two ranks: each row's keyed draft and probability from the draft head's gathered candidates."""

        w, n = self.w, len(positions)
        if all(_gathered_fits(s) for s in samplings):
            chosen, probs = choose_gathered_streams(w, self.mbuf.cand_all, n, list(range(n + 1)),
                                                    [[p] for p in positions], samplings, with_prob=True)
            return [(c[0], p[0]) for c, p in zip(chosen, probs)]
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            toks, probs = tp_sample_rows(w, logits[i:i + 1], [pos], smp, offset=int(w.meta["vocab_offset"]),
                                         id_map=w.draft_ids, with_prob=True)
            out.append((toks[0], probs[0]))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        if self.link is not None and done:
            self.link.send(["finish", [s.sid for s in done]])
        for s in done:
            self.held.pop(s.sid, None)
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        if self.link is not None:
            self.link.send(["drop"])
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
