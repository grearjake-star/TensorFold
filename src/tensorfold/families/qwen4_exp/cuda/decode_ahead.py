"""D4: decode rounds ask for their next window's n-gram table pages while the draft chain runs (TF_DECODE_AHEAD).

A round's ``stage`` gathers each window row's n-gram table rows on the host, on the GPU's critical path: nothing is
queued on the GPU between the last draft pick and the next verify launch. A row whose page is outside the pinned part
of the table (and not in the page cache) costs a major fault there, ~0.44 ms with the GIL held (C2). A window's rows
are known one at a time well before ``stage``: row 0 (the round's end token) once verify's sampling has run, and each
draft once its pick is read back, while the next MTP step still runs. Here each row's table ids go to a background
thread as soon as the row is known, and it asks the kernel for their pages (MADV_WILLNEED through ctypes: the GIL is
released and the reads run asynchronously), so ``stage``'s gather finds them read. Residency only: ``stage`` and
every value it gathers are unchanged, and nothing here touches the GPU. TF_DECODE_AHEAD=0: nothing is asked for.
"""

from __future__ import annotations

import os
import resource
from typing import Sequence

import numpy as np

ENV = "TF_DECODE_AHEAD"
ON = os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")

_POOL = None                     # one worker thread: decode rounds' jobs, apart from the prompt passes' read-ahead


def _submit(job) -> None:
    from concurrent.futures import ThreadPoolExecutor

    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode-ahead")
    _POOL.submit(job)


def _ple(w):
    """The PLE layer whose host table can read ahead (an EXL3 pack's, with ``willneed``), or None."""

    if not ON or getattr(w, "x3", None) is None:
        return None
    p = next((lay.ple for lay in getattr(w, "layers", ()) if getattr(lay, "ple", None) is not None), None)
    return p if p is not None and hasattr(p.table, "willneed") else None


class Rows:
    """The next windows of a round's streams as their rows become known; ``send`` asks for the rows not yet asked."""

    def __init__(self, p) -> None:
        self.p = p
        self.win: dict = {}          # stream key -> [history (n - 1 tokens before the window), window tokens, sent]
        self.sent = 0                # rows asked for (tests)

    def start(self, key, history, token: int) -> None:
        """A window's row 0 (``token``, after ``history``: the stream's n-gram history once its round committed)."""

        if history is None:
            return
        self.win[key] = [np.asarray(history, dtype=np.int64), [int(token)], 0]

    def add(self, key, token: int) -> None:
        """The window's next row (a draft just picked)."""

        got = self.win.get(key)
        if got is not None:
            got[1].append(int(token))

    def send(self) -> None:
        """Ask for the rows known since the last ``send`` (one background job for every stream)."""

        jobs = []
        for got in self.win.values():
            history, tokens, sent = got
            if len(tokens) > sent:
                jobs.append((history, np.asarray(tokens, dtype=np.int64), sent))
                got[2] = len(tokens)
                self.sent += len(tokens) - sent
        if not jobs:
            return
        p = self.p

        def run() -> None:
            # stage hashes the whole window after its history: a row's ids depend only on the rows before it
            p.table.willneed(np.concatenate([p.ngram.ids(h, t)[s:].reshape(-1) for h, t, s in jobs]))

        _submit(run)


def rows(w) -> Rows | None:
    """A round's collector, or None when nothing reads ahead (TF_DECODE_AHEAD=0, no EXL3 host table)."""

    p = _ple(w)
    return Rows(p) if p is not None else None


def start(ahead: Rows | None, items: Sequence) -> None:
    """``items``: (key, state, token) for each stream that runs a next round: its window's row 0."""

    if ahead is None:
        return
    for key, st, token in items:
        ahead.start(key, getattr(st, "ple_history", None), token)


def faults() -> int:
    """This thread's major page faults so far (the done line's stage faults)."""

    try:
        return resource.getrusage(resource.RUSAGE_THREAD).ru_majflt
    except (AttributeError, OSError):              # no per-thread usage on this OS
        return 0
