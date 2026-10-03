"""A lone stream's graph slot, with cache-pointer invalidation and deferred recurrence commits."""

from __future__ import annotations

import time

import torch

from tensorfold.cuda.kernels import gdn
from tensorfold.cuda.logprobs import capture
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream, accept

from .decode import Engine, draft
from .forward import commit
from .state import Buffers


def solo(w, st, capacity, depth, pbuf, timing=None):
    from .graphs import Graphs

    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return None
    rows = max(8, depth + 1)
    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, rows, pbuf.rows
    e.kv_dtype, e.st = st.kv_dtype, st
    e.buf = Buffers(w, rows, capacity, moe_prefill=True)       # the serial engine and shared rounds use these bits
    e.mbuf, e.pbuf = Buffers(w, rows, capacity), pbuf
    e.graphs = Graphs(e, max_rows=rows)
    if timing is not None:
        e.timing = timing                                      # decode.cost_bars: the same table as shared rounds
    return e


class Alone:
    def _state_changed(self, st) -> None:
        """Drop graphs before reallocating a slot: graph pointers must never outlive its cache geometry."""

        if not self.planning and self.solo is not None and st is self.solo.st:
            from .graphs import Graphs

            self.solo.graphs = Graphs(self.solo, max_rows=self.solo.rows)

    def _flush(self, s) -> None:
        """Materialize the shared round's deferred recurrent rows before a graph reads or copies the slot."""

        rows = self.held.pop(s.sid, [])
        if not rows:
            return
        if self.planning:
            self.actions.append(["flush", s.sid])
            return
        sc, st = self.gdn, s.st
        parity = 1 - sc.parity
        if not sc.lin:
            return
        k, v, g, beta = [[getattr(sc, name)[parity, li] for li in range(sc.lin)]
                         for name in ("k", "v", "g", "beta")]
        ptrs = gdn.replay_table(k, v, g, beta, [[st.rec[st.cur[li], li] for li in range(sc.lin)]])
        table = gdn.to_device(ptrs, torch.int64, self.w.device)
        kept = gdn.to_device(rows, torch.int32, self.w.device).view(1, -1)
        counts = gdn.to_device([len(rows)], torch.int32, self.w.device)
        gdn.replay(table, sc.lin, 1, kept, counts, k[0], v[0], in_place=True)

    def _relocate_kept(self, target, avoid, evict: bool = False) -> bool:
        """Move all graph-slot keeps into spare rows without eviction; plans record the same copy and ownership move."""

        spare = next((f for f in self.free if f is not target and f is not avoid), None)
        if spare is None and evict and not self.planning and self.w.comm is None:
            # house (speed-v8.1): a resumed lone turn with every slot kept drops the oldest other kept end (LRU), not
            # the graphs; 0.6.4 moves the graphs instead and each turn recaptures them. A fresh stream left alone
            # keeps 0.6.4's rule (no kept end is lost for it; fresh prompts take the graph slot at admission)
            busy = self._busy()
            spare = next((k[1] for k in self.kept if k[1] is not target and k[1] is not avoid
                          and id(k[1]) not in busy), None)
            if spare is not None:
                self._drop_kept(spare)
                self._shrink(spare, release=True)
                self.free.append(spare)
        if spare is None:
            return False
        size = target.capacity
        if spare.capacity != size:
            self._shrink(spare, release=True)    # back to its first rows (a no-op when already there)
            need = spare.cache_bytes(size) - spare.cache_bytes() + spare.layer_bytes(size)
            if not self.memory_gate.fits(need):
                return False
            self.memory_gate.take(spare.resize(size))
        spare.copy_from(target)
        self.free = [f for f in self.free if f is not spare]
        self.kept = [(ids, spare if st is target else st, snap, tail) for ids, st, snap, tail in self.kept]
        if self.planning:
            self.actions.append(["kept", self._index(target), self._index(spare)])
        return True

    def _hand_over(self, st) -> bool:
        """Admission of a fresh prompt: the idle graph slot's kept ends move into ``st`` (the slot the prompt would
        take), and the prompt takes the graph slot, so the stream decodes there without a later copy or recapture.
        (House, from speed-v8.1: 0.6.4 otherwise moves the graphs, and every new lone request on a full pool
        recaptures them.)"""

        slot = self.solo.st
        if st is slot or id(slot) in self._busy() or any(f is slot for f in self.free):
            return False
        if not any(k[1] is slot for k in self.kept):
            return False
        size = slot.capacity
        if st.capacity < size:
            need = st.cache_bytes(size) - st.cache_bytes() + st.layer_bytes(size)
            if not self.memory_gate.fits(need):
                return False
            self._state_changed(st)
            self.memory_gate.take(st.resize(size))
        st.copy_from(slot)
        self.kept = [(ids, st if k is slot else k, snap, tail) for ids, k, snap, tail in self.kept]
        self.solo_moves = getattr(self, "solo_moves", 0) + 1
        return True

    def _fresh_slot(self, st):
        """The slot a fresh prompt decodes in: the graph slot when it is free or its kept ends can move into ``st``."""

        slot = self.solo.st
        if st is slot:
            return st
        if any(f is slot for f in self.free):
            self.free = [f for f in self.free if f is not slot] + [st]
            return slot
        return slot if self._hand_over(st) else st

    def _move_to_solo(self, s) -> None:
        """Copy a lone stream into the graph slot after committing its pending rows and matching cache sizes."""

        target, old = self.solo.st, s.st
        self._flush(s)
        resumed = getattr(s, "cached", 0) > 0          # house: only a resumed turn may evict another kept end
        if any(k[1] is target for k in self.kept) and not self._relocate_kept(target, old, evict=resumed):
            self.solo.st = old                   # preserve both prefix chains instead of evicting a kept slot
            if self.planning:
                self.actions.append(["solo", self._index(old)])
            else:
                self._state_changed(old)
            return
        self.solo_moves = getattr(self, "solo_moves", 0) + 1
        self._drop_kept(target)
        self.free = [f for f in self.free if f is not target]
        self._shrink(target)
        try:
            grown = self._grow(target, old.capacity, alone=True)
        except NoRoom:
            grown = False
        if not grown:                            # house: no room for the copy's peak, the graphs move to this slot
            self.solo.st = old                   # instead (0.6.4 raised here and failed the round)
            if self.planning:
                self.actions.append(["solo", self._index(old)])
            else:
                self._state_changed(old)
            self._shrink(target)
            self.free.append(target)
            return
        target.copy_from(old)
        s.st = target
        if not any(k[1] is old for k in self.kept) and all(f is not old for f in self.free):
            self._shrink(old)
            self.free.append(old)

    def _solo_round(self, s: Stream) -> list[Stream]:
        """Verify a lone stream through its graphs, commit accepted rows and draft the next chain."""

        e, st = self.solo, s.st
        self.solo_rounds = getattr(self, "solo_rounds", 0) + 1      # house: tests count graph-slot rounds
        t0 = time.perf_counter()
        tokens = [s.out[-1]] + list(s.drafts)
        R = len(tokens)
        logits = e.forward(tokens)
        rows = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], s.sampling)
        path, end = accept(tokens, list(range(-1, R - 1)), rows, s.count - len(s.out), self._ends(s))
        if s.probabilities is not None:
            capture(logits, [tokens[r] for r in path[1:]] + [end],
                    [st.pos + 1 + r for r in path], s.probabilities, rows=path)
        commit(self.w, st, e.buf, R, len(path))
        s.committed.extend(tokens[:len(path)])
        s.counted(R)
        new = [tokens[r] for r in path[1:]] + [end]
        last = len(s.out) + len(new) >= s.count or end in self._ends(s)
        s.drafts = []
        room = min(self.depth, s.count - len(s.out) - len(path))
        if not last and room > 0:
            more = {"cost": self.cost} if self.cost > 0 else {}
            s.drafts = draft(e, e.buf.streams[:len(path)], rows[:len(path)], st.pos + 1, room, s.sampling,
                             self.confidence, **more)
        s.take(new, self._ends(s))
        self._timed(time.perf_counter() - t0, 0)    # a prompt arriving next sizes its passes by this round's time
        return [s] if s.done else []
