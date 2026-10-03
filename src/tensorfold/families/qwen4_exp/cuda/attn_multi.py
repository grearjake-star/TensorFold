"""A concurrent step's attention in one launch a kernel: rows find their caches by table and run the one-stream code."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import gdn as shared

from . import attention as attn_mod, glue
from .attention import CHUNK, SELECT_REGS, _chunk, _merge_row, _pool_block, _radix_place

PTRS = 6                 # a stream's pointers a layer: keys, values, key scales, value scales, index keys, pooled


@triton.jit
def _ptr(TABLE, s, T: tl.constexpr):
    return tl.multiple_of(tl.load(TABLE + s).to(tl.pointer_type(T)), 16)


@triton.jit
def _prep_multi(P, POSR, SID, CP, VP, QW, KW, IW, INV, Q, IQ, eps, N, PW: tl.constexpr, NQ: tl.constexpr,
                NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
                BITS: tl.constexpr, KT: tl.constexpr, VISION: tl.constexpr,
                S1: tl.constexpr = 11, S2: tl.constexpr = 10):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    rope, delta, length = CP, CP, 0
    if VISION:
        rope, delta = _ptr(VP, s, tl.int32), _ptr(VP + N, s, tl.int32)
        length = tl.load(VP + 2 * N + s).to(tl.int32)
    glue._prep_row(P, tl.load(POSR + r), r, tl.program_id(1), QW, KW, IW, INV, Q, _ptr(CP, s, KT),
                   _ptr(CP + N, s, KT), _ptr(CP + 2 * N, s, tl.float16), _ptr(CP + 3 * N, s, tl.float16), IQ,
                   _ptr(CP + 4 * N, s, tl.bfloat16), eps, PW, NQ, NKV, HD, NI, IHD, HALF, BITS,
                   ROPE=rope, DELTA=delta, length=length, MODE=2 if VISION else 0, S1=S1, S2=S2)


@triton.jit
def _pool_multi(CP, VP, P0, RS, W, INV, eps, N, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr,
                VISION: tl.constexpr, S1: tl.constexpr = 11, S2: tl.constexpr = 10):
    s = tl.program_id(0)
    rope, delta, length = CP, CP, 0
    if VISION:
        rope, delta = _ptr(VP, s, tl.int32), _ptr(VP + N, s, tl.int32)
        length = tl.load(VP + 2 * N + s).to(tl.int32)
    _pool_block(_ptr(CP + 4 * N, s, tl.bfloat16), _ptr(CP + 5 * N, s, tl.bfloat16), tl.load(P0 + s),
                tl.program_id(1), W, INV, eps, tl.load(RS + s), DI, HALF, RATIO,
                ROPE=rope, DELTA=delta, length=length, MODE=2 if VISION else 0, S1=S1, S2=S2)


@triton.jit
def _chunks_multi(Q, CP, POSR, SID, PO, PM, PL, IDS, NKR, N, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr,
                  QSA: tl.constexpr, BITS: tl.constexpr, KT: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    n = tl.load(POSR + r) + 1
    sparse = False
    if QSA:                                          # a row is sparse when its select would mark it (end past TOP)
        sparse = n // RATIO > TOP
        n = tl.where(sparse, tl.load(NKR + r), n)
    _chunk(Q, _ptr(CP, s, KT), _ptr(CP + N, s, KT), _ptr(CP + 2 * N, s, tl.float16), _ptr(CP + 3 * N, s, tl.float16),
           n, sparse, r, tl.program_id(1), tl.program_id(2), PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW, QSA,
           BITS)


@triton.jit
def _merge_multi(PO, PM, PL, POSR, OUT, NKR, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                 CH: tl.constexpr, NCH: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr, RATIO: tl.constexpr,
                 TOP: tl.constexpr):
    r = tl.program_id(0)
    n = tl.load(POSR + r) + 1
    if QSA:
        n = tl.where(n // RATIO > TOP, tl.load(NKR + r), n)
    _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, HK, D, G, CH, NCH, BITS)


# --- graph rounds (multi_graphs): every row's select in one launch, its stream's pooled keys found by table ---------
# The same arithmetic as attention._scores/_select/_select_tiles on the rows of one stream (whose launch is offset to
# its first row): a row's end is its own position + 1 (POSR) instead of the launch's P0 + r + 1.

@triton.jit
def _scores_multi(IQ, CP, POSR, SID, SC, NB, N, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr,
                  TOP: tl.constexpr, BB: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1)
    complete = (tl.load(POSR + r) + 1) // RATIO
    if complete > TOP and j * BB < complete:
        POOLED = _ptr(CP + 5 * N, tl.load(SID + r), tl.bfloat16)
        b = j * BB + tl.arange(0, BB)
        ok = b < complete
        d = tl.arange(0, DI)
        k = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        total = tl.zeros((BB,), dtype=tl.float32)
        for h in tl.static_range(HI):
            q = tl.load(IQ + (r * HI + h) * DI + d).to(tl.float32)
            total = total + tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
        tl.store(SC + r * NB + b, total / tl.sqrt(DI * 1.0), mask=ok)


@triton.jit
def _select_multi(SC, POSR, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
                  BLOCK: tl.constexpr):
    r = tl.program_id(0)
    end = tl.load(POSR + r) + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        b = tl.arange(0, BLOCK)
        ok = b < complete
        v = tl.load(SC + r * NB + b, mask=ok, other=0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        key = tl.where(ok, key, 0)
        lo = tl.zeros((), dtype=tl.uint64)
        hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint64)
        k64 = key.to(tl.uint64)
        for _ in range(33):
            mid = (lo + hi + 1) // 2
            count = tl.sum(tl.where(k64 >= mid, 1, 0), axis=0)
            take = count >= TOP
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        cut = lo
        above = ok & (k64 > cut)
        equal = ok & (k64 == cut)
        need = TOP - tl.sum(above.to(tl.int32), axis=0)
        rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (rank <= need))
        place = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        for k in tl.static_range(RATIO):
            tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


@triton.jit
def _select_tiles_multi(SC, POSR, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
                        TB: tl.constexpr):
    r = tl.program_id(0)
    end = tl.load(POSR + r) + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        _radix_place(SC, IDS, r, complete, NB, RATIO, TOP, IDW, TB)
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


def _qsa_multi(b, sc, step: "Step", cp, rows: int, context: int) -> None:
    """Every row's scores and select (dense rows keep their length) with launches bounded by ``context``: qsa_rows'
    lists for each stream's rows, at one launch shape for any split of the window between streams."""

    ratio, top, di = sc.ratio, sc.budget // sc.ratio, b.iq.shape[2]
    blocks = min(sc.nb, max(1, triton.cdiv(context, ratio)))
    bb = 64
    _scores_multi[(rows, triton.cdiv(blocks, bb))](b.iq, cp, step.posr, step.sid, sc.scores, sc.nb, step.n,
                                                   HI=b.iq.shape[1], DI=di, RATIO=ratio, TOP=top, BB=bb, num_warps=4)
    width = triton.next_power_of_2(blocks)
    if width <= SELECT_REGS:
        _select_multi[(rows,)](sc.scores, step.posr, sc.ids, sc.nk, sc.sparse, sc.nb, RATIO=ratio, TOP=top,
                               IDW=sc.idw, BLOCK=width, num_warps=16)
        return
    tb, warps = (4096, 8) if rows >= 64 else (8192, 16)
    _select_tiles_multi[(rows,)](sc.scores, step.posr, sc.ids, sc.nk, sc.sparse, sc.nb, RATIO=ratio, TOP=top,
                                 IDW=sc.idw, TB=tb, num_warps=warps)


def _put(values, dtype, dev, out: torch.Tensor | None) -> torch.Tensor:
    """A table on the device: a new tensor (eager rounds), or written into ``out`` (a graph's persistent table)."""

    if out is None:
        return shared.to_device(values, dtype, dev)
    out.copy_(torch.tensor(values, dtype=dtype).pin_memory(), non_blocking=True)
    return out


class Step:
    """A step's row, stream and cache-pointer tables (the MTP head's with ``mtp``), read now: caches may move."""

    def __init__(self, w, segs: Sequence, mtp: bool, out: dict | None = None, bucket: int | None = None) -> None:
        n, rows = len(segs), segs[-1][2]
        layers = [l for l in w.layers if not l.linear] if not mtp else [w.mtp.layer]
        self.index = {layer.index: i for i, layer in enumerate(layers)}
        first = [st.mtp_len if mtp else st.pos for st, _, _ in segs]
        posr, sid = np.empty((rows,), np.int32), np.empty((rows,), np.int32)
        for s, ((_, a0, a1), p0) in enumerate(zip(segs, first)):
            posr[a0:a1] = p0 + np.arange(a1 - a0)
            sid[a0:a1] = s
        counts = [a1 - a0 for _, a0, a1 in segs]
        ptrs = np.empty((len(layers), PTRS, n), np.int64)
        for s, (st, _, _) in enumerate(segs):
            for i, layer in enumerate(layers):
                if mtp:
                    kc, ikc, pooled = st.mtp_kc, st.mtp_ikc, st.mtp_pooled
                else:
                    a = st.att_index[layer.index]
                    kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
                ptrs[i, :, s] = [kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(),
                                 ikc.data_ptr(), pooled.data_ptr()]
        dev = w.device
        # ``out``: a graph round's persistent tables (multi_graphs), whose launches are bounded by ``bucket`` keys
        out = out or {}
        self.bucket = bucket
        ints = _put(np.concatenate([posr, sid, first, counts]).tolist(), torch.int32, dev, out.get("ints"))
        self.posr, self.sid = ints[:rows], ints[rows:2 * rows]
        self.first, self.counts = ints[2 * rows:2 * rows + n], ints[2 * rows + n:]
        self.ptrs = _put(ptrs.ravel().tolist(), torch.int64, dev, out.get("ptrs")).view(len(layers), PTRS * n)
        self.tails = (_put([st.ple_tail.data_ptr() for st, _, _ in segs], torch.int64, dev, out.get("tails"))
                      if bucket is not None else None)       # each stream's n-gram conv tail (forward.ple_block)
        self.n, self.rows, self.segs = n, rows, list(segs)
        self.ends = [p0 + c for p0, c in zip(first, counts)]
        self.most = max(counts)
        self.vision = any(st.image_positions is not None for st, _, _ in segs)
        self.vision_ptrs = self.ptrs[0]
        if self.vision:
            vp = [[st.image_positions.data_ptr() if st.image_positions is not None else st.pos_dev.data_ptr(),
                   st.rope_delta_dev.data_ptr(), 0 if st.image_positions is None else st.image_positions.shape[0]]
                  for st, _, _ in segs]
            self.vision_ptrs = shared.to_device(np.asarray(vp, dtype=np.int64).T.ravel().tolist(), torch.int64, dev)


def layer(layer, w, b, step: Step, mtp: bool, scale: float) -> torch.Tensor:
    """Every stream's ``layer`` attention: prep, pool, sparse streams' own selects, chunks, merge -> b.attn_o[:R]."""

    c, a, sc = w.cfg, layer.attn, b.attn
    n, rows, cp = step.n, step.rows, step.ptrs[step.index[layer.index]]
    st0 = step.segs[0][0]
    cache0 = st0.mtp_kc if mtp else st0.kc[0]
    bits = 0 if not cache0.quantized else cache0.bits
    kt = {0: tl.bfloat16, 8: tl.int8, 4: tl.uint8}[bits]
    heads = c.heads + c.kv_heads + c.index_heads + 1
    sections = w.cfg.mrope_section
    _prep_multi[(rows, heads)](b.pa[:rows], step.posr, step.sid, cp, step.vision_ptrs,
                               a.q_scale, a.k_scale, a.iq_scale, w.inv_freq,
                               b.q, b.iq, c.eps, n, PW=b.pa.shape[1], NQ=c.heads, NKV=c.kv_heads, HD=c.head_dim,
                               NI=c.index_heads, IHD=c.index_dim, HALF=w.inv_freq.numel(), BITS=bits, KT=kt,
                               VISION=step.vision, S1=sections[1], S2=sections[2],
                               num_warps=2)
    top = sc.budget // sc.ratio
    graph = step.bucket is not None              # a graph round: launch shapes from the window's rows and bucket only
    if sc.qsa:
        most = rows if graph else step.most      # blocks past a stream's rows pool nothing (_pool_block)
        _pool_multi[(n, most // sc.ratio + 2)](cp, step.vision_ptrs, step.first, step.counts,
                                               a.ik_scale, w.inv_freq, c.eps, n,
                                               DI=c.index_dim, HALF=w.inv_freq.numel(), RATIO=sc.ratio,
                                               VISION=step.vision, S1=sections[1], S2=sections[2],
                                               num_warps=1)
        if graph:
            _qsa_multi(b, sc, step, cp, rows, step.bucket)
        else:
            for (st, a0, a1), end in zip(step.segs, step.ends):
                if end // sc.ratio > top:                # this stream has sparse rows: its own select
                    _, _, pooled, pos, _ = _caches(layer, st, mtp)
                    attn_mod.qsa_rows(b.iq[a0:a1], pooled, pos, _rows_from(sc, a0), a1 - a0, context=end)
    keys = step.bucket if graph else max(step.ends)     # chunks past a row's keys write nothing
    if sc.qsa:
        keys = min(keys, (top + 1) * sc.ratio - 1)
    chunks = min(sc.nch, triton.cdiv(keys, CHUNK))
    hk = c.kv_heads
    g = c.heads // hk
    _chunks_multi[(rows, hk, chunks)](b.q, cp, step.posr, step.sid, sc.po, sc.pm, sc.pl, sc.ids, sc.nk, n, H=c.heads,
                                      HK=hk, D=c.head_dim, G=g, CH=CHUNK, NCH=sc.nch, SCALE=scale, IDW=sc.idw,
                                      QSA=sc.qsa, BITS=bits, KT=kt, RATIO=sc.ratio, TOP=top, num_warps=4,
                                      num_stages=1)
    _merge_multi[(rows, hk)](sc.po, sc.pm, sc.pl, step.posr, b.attn_o, sc.nk, H=c.heads, HK=hk, D=c.head_dim, G=g,
                             CH=CHUNK, NCH=sc.nch, QSA=sc.qsa, BITS=bits, RATIO=sc.ratio, TOP=top, num_warps=4)
    return b.attn_o[:rows]


def _caches(layer, st, mtp: bool) -> tuple:
    if mtp:
        return st.mtp_kc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, st.mtp_len
    ai = st.att_index[layer.index]
    return st.kc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, st.pos


def _rows_from(sc, a0: int) -> SimpleNamespace:
    """The scratch as seen by a launch whose row 0 is window row ``a0``."""

    return SimpleNamespace(ratio=sc.ratio, budget=sc.budget, nb=sc.nb, idw=sc.idw, scores=sc.scores[a0:],
                           ids=sc.ids[a0:], nk=sc.nk[a0:], sparse=sc.sparse[a0:])
