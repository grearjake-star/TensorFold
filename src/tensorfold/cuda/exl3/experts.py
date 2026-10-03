"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate 1.5..7.5 (3, 5, .. 15).
K2_SUPPORTED = tuple(range(2, 17))

# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)


# prompt windows rotate their inputs with four warps a program (rot_in4_kernel: each warp rot_in_kernel's arithmetic);
# TF_ROT_IN4=0 keeps the one-warp programs. Decode windows always keep them.
ROT_IN4 = __import__("os").environ.get("TF_ROT_IN4", "1") != "0"


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v8r", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """Half-bits a value of a trellis int16 [K/16, N/16, 16 * K] (16 * K + 8 for the half-integer rates)."""

    w = trellis.shape[-1]
    if w % 8:
        raise ValueError(f"trellis last dim {w} is not a multiple of 8")
    k2 = w // 8
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"unsupported EXL3 bit width {k2 / 2}")
    return k2


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], half-bits a value
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)
    widths_gu: int = 0        # a bit per K2 gate and up hold (bit k2): the prompt kernels' instances to launch
    widths_d: int = 0

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())


def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda") -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
                raise ValueError("trellis must be a contiguous CUDA int16 tensor")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I)
    upp, uk, uks = table(up, D, I)
    dp, dk, dks = table(down, I, D)

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
    bits = lambda ks: sum(1 << k for k in set(ks))          # noqa: E731
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep, bits(gks + uks),
                             bits(dks))


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        maxu = min(P, ex.count)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members_buf = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count
        self.device = device
        self._plan = None          # prompt windows: pairs grouped into items (``plan``), made on first use

    def plan(self):
        """A prompt plan over this scratch's rows (tensorfold.cuda.experts' grouping: items of one expert's next pairs,
        pair order) with one more expert than the layer's, so skipped picks (== E) group apart and are skipped."""

        if self._plan is None:
            from tensorfold.cuda import experts as grouped

            self._plan = grouped.Plan(self.rows, self.slots, self.count_experts + 1, self.device, prefill=True)
            self._pick = torch.empty((self.rows * self.slots,), dtype=torch.int32, device=self.device)
        return self._plan

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


# Prompt windows (more than PROMPT_ROWS rows) take a plan's items instead of the grouping kernel, whose grid spans
# (expert slots) x (member tiles of the whole window), nearly all empty at prompt sizes, and decodes each weight tile
# again for every 16 pairs. TF_EXL3_PROMPT picks the kernel: "prompt" (default; experts_prompt.cuh: each tile decoded
# once a program into shared memory for its item's 64 pairs, the whole K in registers, no split partials), "items"
# (K's parked grouped_items_kernel: ITEM_TILES member tiles an item, split partials kept) or "group" (the grouping kernel
# at every size). Every choice gives every pair the same bits; decode windows always keep the grouping kernel.
_ENV = __import__("os").environ
PROMPT = _ENV.get("TF_EXL3_PROMPT", "prompt")
PROMPT_ROWS = int(_ENV.get("TF_EXL3_PROMPT_ROWS", "64"))
PROMPT_TILE = 64         # pairs a prompt item holds: 4 warps of 16
PROMPT_NT = int(_ENV.get("TF_EXL3_PROMPT_NT", "4"))     # n tiles a prompt program (4 or 8; never a pair's bits)
ITEM_ROWS = PROMPT_ROWS if PROMPT == "items" else 0     # kept for K's guard test (0: the item path off)
ITEM_TILES = 2           # member tiles an item holds (MTB in experts_grouped.cuh: 1, 2 or 4)
ITEM_NT = {1: 8, 2: 8, 4: 4}      # n tiles a program at each MTB (the n tiles never change a pair's bits)


def _mode(R: int, group: bool) -> str:
    if not group:
        return "given"                   # the caller grouped (ids, members) already
    if PROMPT == "prompt" and PROMPT_ROWS > 0 and R > PROMPT_ROWS:
        return "prompt"
    if ITEM_ROWS > 0 and R > ITEM_ROWS:
        return "items"
    return "group"


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True, y_out: torch.Tensor | None = None) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts``; no host sync.

    ``y_out`` ([R * slots, D] bf16, contiguous; prompt windows without ``wts``): Y rounded once to bf16 straight from the
    prompt kernel (the bits of copying the fp32 Y into it); returned in place of the fp32 Y. Other paths fill it by copy."""

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    mode = _mode(R, group)
    if mode in ("items", "prompt"):
        from tensorfold.cuda import experts as grouped

        plan = s.plan()
        tile = PROMPT_TILE if mode == "prompt" else 16 * ITEM_TILES
        picks = torch.clamp(pick[:R].reshape(P), max=E, out=s._pick[:P])     # every skipped pick as E
        grouped.route(picks.view(R, slots), plan, tile)
        bound = grouped.max_items(P, E + 1, tile)
    else:
        ids, members = s.window(R)
        if mode == "group":
            ext.group(pick, ids, s.count, members, R, slots, E)
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E, mode == "prompt" and ROT_IN4)
    nt, w, sk, pf = s.cfg_gu
    if mode == "prompt":
        ext.prompt(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, plan.items, plan.counts, plan.members,
                   s.z, 2, D, I, P, E, (D // 16) // (sk * w), w, bound, ex.cb, PROMPT_NT, ex.widths_gu)
        sk = 1                           # the kernel added the splits in order: one final slice
    elif mode == "items":
        ext.grouped_items(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, plan.items, plan.counts,
                          plan.members, s.z, 2, D, I, P, sk, E, bound, ex.cb, ITEM_NT[ITEM_TILES], w, 1, ITEM_TILES,
                          ex.k2_gu[0], ex.k2_gu[1])
    else:
        ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                    P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit), act_mode)
    nt, w, sk, pf = s.cfg_d
    if mode == "prompt" and sk == 1:                 # down and its epilogue in one kernel (one split: no Z)
        y = y_out if y_out is not None and wts is None else s.y
        ext.prompt_down(s.xd, ex.down_ptr, ex.down_k2, ex.svh_d, plan.items, plan.counts, plan.members, y.view(-1, D),
                        I, D, P, E, (I // 16) // w, w, bound, ex.cb, ex.widths_d)
        if wts is None:
            return y[:P]
        if out is None:
            out = torch.empty((R, D), dtype=torch.float32, device=x.device)
        ext.combine(s.y, wts, out, R, D, slots)       # down_combine's combine: the same fma chain
        return out
    if mode == "prompt":
        ext.prompt(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, plan.items, plan.counts,
                   plan.members, s.z, 1, I, D, P, E, (I // 16) // (sk * w), w, bound, ex.cb, PROMPT_NT, ex.widths_d)
        sk = 1
    elif mode == "items":
        ext.grouped_items(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, plan.items, plan.counts,
                          plan.members, s.z, 1, I, D, P, sk, E, bound, ex.cb, ITEM_NT[ITEM_TILES], w, 1, ITEM_TILES,
                          ex.k2_d[0], ex.k2_d[1])
    else:
        ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                    D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E)
        if y_out is not None:
            y_out[:P].copy_(s.y[:P])
            return y_out[:P]
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E)
    return out


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
