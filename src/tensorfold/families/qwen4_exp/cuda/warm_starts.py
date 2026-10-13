"""Warm starts: the system blocks a --parallel server kept states for, prefilled again after a restart.

A kept message-start state (the end of a shared system block, ``prefixes.remember(start=True)``) lives in memory only,
so the first chat after every restart prefills its whole system block again. With ``TF_WARM_STARTS=<file>`` the
server records the token ids of those system blocks (only prompts that open with a system message, cut at the second
message's start: system and tool text, never a user or assistant message) in a local file it creates with mode 0600,
and at its next start, before it takes traffic, prefills each of the ``TF_WARM_REPLAY`` (default 2) most recently used
blocks as a request of its own: the block plus a user turn's opening, one greedy token. The replay is an ordinary request through the
ordinary prefill, so the state it keeps is the one a first chat would have kept (exact by construction), and a later
chat resumes from it as a second chat does. Nothing is restored from disk: a new build, settings or weights recompute
it; a recorded block whose tokenizer changed is ignored.

TF_PIN_BASE (H2): one client's base prompt (Hermes: tools, then a system text whose memory and environment lines at
its end change from session to session) is recorded as a family of blocks that share a long head. ``stable`` gives
the family's shared length, so the system-block checkpoint falls inside it rather than past it (a checkpoint at the
last granule before a 17,136-token block's end, 16,384, is useless to a session whose block differs from token
14,604 on); the most used block never leaves the file and is always replayed, last (the newest kept state).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from tensorfold.cuda.markers import MIN_GAP

FILE_ENV, REPLAY_ENV = "TF_WARM_STARTS", "TF_WARM_REPLAY"
LIMIT = 4                        # system blocks the file remembers (most recently used first)
REPLAY = 2                       # system blocks a start prefills again (each keeps one slot's rows, as a chat would)
VERSION = 1
FAMILY = 4096                    # two recorded blocks sharing at least this many tokens: one client's base prompt


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    """How many leading tokens two blocks share."""

    import numpy as np

    m = min(len(a), len(b))
    diff = np.flatnonzero(np.asarray(a[:m], dtype=np.int64) != np.asarray(b[:m], dtype=np.int64))
    return int(diff[0]) if len(diff) else m


def _top(entries: list[dict]) -> dict | None:
    """The most used block (ties: the most recent): the one a pinned base is made from."""

    return max(entries, key=lambda e: (e["uses"], e["last"]), default=None)


def _key(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()[:32]


class WarmStarts:
    """The recorded system blocks of one checkpoint's tokenizer; ``note`` from the decode thread, writes on a thread."""

    def __init__(self, path: str | Path, *, head: Sequence[int], opener: int, user: Sequence[int],
                 fingerprint: str, vocab: int, limit: int = LIMIT) -> None:
        self.path, self.head, self.opener, self.user = Path(path), list(head), int(opener), list(user)
        self.fingerprint, self.vocab, self.limit = fingerprint, int(vocab), limit
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.saving = threading.Lock()                # one write at a time (the writer thread, or ``save``)
        self.dirty = False
        self.writer: threading.Thread | None = None
        # id(kept ids) -> (those ids, their key or None when not a system block): a resumed kept state is noted
        # again on every resume, on the decode thread; its scan, copy and digest are done once
        self.known: dict[int, tuple[Sequence[int], int, list[int] | None, str | None]] = {}
        self.entries: list[dict] = self._load()      # most recently used first: {"ids", "last", "uses"}
        self._stable: tuple[int, list[int]] | None = None      # (shared length, the top block's head) or None
        self._refresh()

    @classmethod
    def from_env(cls, model_dir: str | Path, vocab: int) -> "WarmStarts | None":
        """``TF_WARM_STARTS`` set: the recorder for this checkpoint's tokenizer (None when off or not a ChatML one)."""

        path = os.environ.get(FILE_ENV, "").strip()
        if not path:
            return None
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        tok_file = model_dir / "tokenizer.json"
        if not tok_file.is_file():
            return None
        tok = Tokenizer.from_file(str(tok_file))
        opener = tok.token_to_id("<|im_start|>")
        if opener is None:
            print(f"[tensorfold] warm starts: off ({FILE_ENV} is set, but this tokenizer has no <|im_start|>)",
                  flush=True)
            return None
        head = tok.encode("<|im_start|>system\n", add_special_tokens=False).ids
        user = tok.encode("<|im_start|>user\n", add_special_tokens=False).ids
        return cls(path, head=head, opener=opener, user=user, fingerprint=_digest(tok_file), vocab=vocab)

    def system_only(self, ids: Sequence[int]) -> bool:
        """A prompt that opens with a system message, cut where the second message starts: nothing else in it."""

        n = len(self.head)
        return (len(ids) >= max(MIN_GAP, n + 1) and list(ids[:n]) == self.head
                and not any(int(t) == self.opener for t in ids[1:]))

    def note(self, ids: Sequence[int]) -> None:
        """A system block's state was kept or resumed: it becomes the most recently used (other prefixes: ignored)."""

        hit = self.known.get(id(ids))
        if hit is not None and hit[0] is ids and hit[1] == len(ids):
            _, _, flat, key = hit
        else:
            flat = [int(t) for t in ids] if self.system_only(ids) else None
            key = _key(flat) if flat is not None else None
            if len(self.known) >= 4 * self.limit:              # kept states come and go: forget the oldest
                self.known.pop(next(iter(self.known)))
            self.known[id(ids)] = (ids, len(ids), flat, key)   # holds ``ids``, so its id is not reused meanwhile
        if flat is None:
            return
        ids = flat
        with self.lock:
            old = next((e for e in self.entries if e["key"] == key), None)
            entry = {"key": key, "ids": ids, "last": round(time.time()), "uses": (old["uses"] if old else 0) + 1}
            before = _top(self.entries)
            rest = [e for e in self.entries if e["key"] != key]
            kept = rest[:self.limit - 1]
            if before is not None and self.limit > 1 and before["key"] != key and all(e is not before for e in kept) \
                    and before["uses"] > entry["uses"]:
                kept = kept[:-1] + [before]               # the most used block stays: it is the pinned base
            self.entries = [entry] + kept
            if old is None or (before or {}).get("key") != _top(self.entries)["key"]:
                self._refresh()                           # a new block or a new top: the shared length again
            self.dirty = True
            if self.writer is None:
                self.writer = threading.Thread(target=self._write_loop, name="tf-warm-starts", daemon=True)
                self.writer.start()
            self.wake.notify()

    def _refresh(self) -> None:
        """The top block's family length: the shortest head it shares with another recorded block of at least
        FAMILY tokens (None while no other block shares one: nothing says yet where its sessions differ)."""

        top = _top(self.entries)
        shared = [] if top is None else [n for e in self.entries if e is not top
                                         if (n := _common(top["ids"], e["ids"])) >= FAMILY]
        self._stable = (min(shared), list(top["ids"][:min(shared)])) if shared else None

    def stable(self, ids: Sequence[int]) -> int | None:
        """How many of this prompt's tokens are the pinned family's shared head (None: not that family, or none yet)."""

        got = self._stable
        if got is None:
            return None
        n, head = got
        return n if len(ids) > n and list(ids[:n]) == head else None

    def family(self, ids: Sequence[int]) -> bool:
        """A kept state of the pinned family: at least FAMILY tokens, all of them its shared head or past it."""

        got = self._stable
        if got is None or len(ids) < FAMILY:
            return False
        n, head = got
        m = min(len(ids), n)
        return list(ids[:m]) == head[:m]

    def replays(self, n: int, pin: bool = False) -> list[list[int]]:
        """The ``n`` most recently used blocks as prompts, least recent first (the most recent is kept newest).
        ``pin``: the most used block is among them, and last (the newest kept state)."""

        with self.lock:
            chosen = self.entries[:max(0, n)]
            top = _top(self.entries)
        order = list(reversed(chosen))
        if pin and top is not None and n > 0:
            order = [e for e in order if e is not top][-(n - 1):] if n > 1 else []
            order.append(top)
        return [e["ids"] + self.user for e in order]

    def _load(self) -> list[dict]:
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return []
        except OSError as exc:
            print(f"[tensorfold] warm starts: {self.path} unreadable ({type(exc).__name__}); starting empty",
                  flush=True)
            return []
        try:
            os.chmod(self.path, 0o600)                    # only this user reads the system blocks
            data = json.loads(raw)
            if data.get("version") != VERSION or data.get("tokenizer") != self.fingerprint:
                return []                                 # another tokenizer: its token ids mean other text
            out = []
            for e in data.get("entries", [])[:self.limit]:
                ids = [int(t) for t in e["ids"]]
                if self.system_only(ids) and all(0 <= t < self.vocab for t in ids) and _key(ids) == e["key"]:
                    out.append({"key": e["key"], "ids": ids, "last": int(e.get("last", 0)),
                                "uses": int(e.get("uses", 0))})
            return out
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            print(f"[tensorfold] warm starts: {self.path} is not a warm-start file; starting empty", flush=True)
            return []

    def _write_loop(self) -> None:
        while True:
            with self.lock:
                while not self.dirty:
                    self.wake.wait()
                self.dirty = False
            try:
                self.save()                               # the newest entries, taken under the write lock
            except OSError as exc:
                print(f"[tensorfold] warm starts: could not write {self.path} ({type(exc).__name__})", flush=True)
            time.sleep(1.0)                               # a burst of chats: one write a second at most

    def save(self, data: dict | None = None) -> None:
        """Write the file atomically, created 0600 in a 0700 directory it makes."""

        with self.saving:
            if data is None:
                with self.lock:
                    data = {"version": VERSION, "tokenizer": self.fingerprint, "entries": list(self.entries)}
            self._save(data)

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, self.path)


def replay_count() -> int:
    value = os.environ.get(REPLAY_ENV, "").strip()
    if not value:
        return REPLAY
    if not value.isdecimal() or int(value) > 8:
        raise ValueError(f"{REPLAY_ENV}: 0 to 8 system blocks, not {value!r}")
    return int(value)


def replay(warm: WarmStarts, submit: Callable[[list[int]], object], n: int, pin: bool = False) -> int:
    """Prefill the recorded blocks one after another and return how many were kept; logs lengths only.

    Run before the server takes traffic: a block prefilled alone is cut into the same passes as a cold first chat on
    an idle server. Beside another prompt its passes would be cut differently and its state need not match that."""

    prompts = warm.replays(n, pin) if pin else warm.replays(n)
    t0, done = time.perf_counter(), []
    for prompt in prompts:
        try:
            submit(prompt)
            done.append(len(prompt) - len(warm.user))
        except Exception as exc:                          # noqa: BLE001  (a replay is an optimization only)
            print(f"[tensorfold] warm starts: a replay failed ({type(exc).__name__}: {exc})", flush=True)
    if done:
        print(f"[tensorfold] warm starts: {len(done)} system block(s) prefilled again "
              f"({', '.join(str(k) for k in done)} tokens) in {time.perf_counter() - t0:.1f}s, before serving",
              flush=True)
    return len(done)
