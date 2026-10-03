"""A differentiable PyTorch copy of Flash Next's MTP head for self-distillation, on any checkpoint format
(workstream D; v1's mtp_torch.py semantics, validated against the CUDA head, with the weights' storage behind a
``sources.Source``).

At row i the head reads the main model's streams h_i [4 x 2560] and the embedding of token i+1:
    x = fc_e(norm_e(embed(t))) + fc_h(norm_h(h)) per stream;  x = decoder layer (dense causal attention while the
    head's cache holds <= 2048 entries, MoE);  logits = draft_head(mixer(x)).
Trainable (fp32 masters in the canonical space: effective linears [out, in], centred norms): the source's
``trainable`` linears and every norm. In each forward a linear is rounded to what the target format will store, with
a straight-through gradient: bf16 (modelopt, exl3), or the checkpoint's own 4-bit grid (mlx4, ``grid``), so the
exported file holds exactly the weights training saw. Frozen: routed experts (bf16), the embedding, the draft head
rows (as the engine builds ``w.draft_head``), and any linear the format cannot replace (exl3: the shared expert).
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F

import sources as SRC

D, S, LOW, H, KV, HD, ROT = 2560, 4, 320, 24, 2, 256, 64
EPS = 1e-6
THETA = 1e7
E = SRC.E


def rms(x: torch.Tensor, w: torch.Tensor, group: int | None = None) -> torch.Tensor:
    shp = x.shape
    y = x.float()
    if group is not None:
        y = y.reshape(*shp[:-1], -1, group)
        w = w.reshape(-1, group)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS) * w
    return y.reshape(shp).to(torch.bfloat16)


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    half = ROT // 2
    inv = THETA ** (-torch.arange(half, device=x.device, dtype=torch.float64) / half)
    ang = pos.double()[:, None] * inv[None]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    xf = x.float()
    a, b = xf[..., :half], xf[..., half:ROT]
    return torch.cat([a * cos - b * sin, b * cos + a * sin, xf[..., ROT:]], dim=-1).to(torch.bfloat16)


def key(n: str) -> str:
    return n.replace(".", "__")


class MTPHead(torch.nn.Module):
    def __init__(self, src: SRC.Source, draft_ids: np.ndarray, override: dict | None = None) -> None:
        """``override``: canonical tensors {name: tensor} (``src.read_head`` of a trained file) over the source's."""

        super().__init__()
        self.src = src
        dev = src.device
        ov = override or {}

        def init(n, f):
            return (ov[n] if n in ov else f(n)).float().to(dev)

        train = set(src.trainable)
        self.lin = torch.nn.ParameterDict({key(n): torch.nn.Parameter(init(n, src.linear)) for n in SRC.LINEARS
                                           if n in train})
        for n in SRC.LINEARS:
            if n not in train:
                self.register_buffer("frozen_" + key(n), init(n, src.linear).to(torch.bfloat16), persistent=False)
        self.norms = torch.nn.ParameterDict({key(n): torch.nn.Parameter(init(n, src.norm)) for n in SRC.NORMS})
        self.grids = {}
        for n in SRC.LINEARS:
            g = src.grid(n) if n in train else None
            if g is not None:
                s, b = g
                self.grids[n] = (s.float().repeat_interleave(32, -1).to(dev), b.float().repeat_interleave(32, -1).to(dev))
        # frozen indexer (long contexts: past BUDGET keys a row attends to its best blocks and its tail)
        self.register_buffer("idx_w", src.linear("layers.0.self_attn.indexer.index_qk_proj").to(torch.bfloat16).to(dev),
                             persistent=False)
        self.register_buffer("idx_qn", (1.0 + src.norm("layers.0.self_attn.indexer.q_layernorm.weight").to(
            torch.bfloat16).float()).to(dev), persistent=False)
        self.register_buffer("idx_kn", (1.0 + src.norm("layers.0.self_attn.indexer.k_layernorm.weight").to(
            torch.bfloat16).float()).to(dev), persistent=False)
        self.eg, self.eu, self.ed = src.experts()
        ids = torch.from_numpy(np.asarray(draft_ids, dtype=np.int64)).to(dev)
        self.draft_ids = ids
        self.head = src.draft_rows(ids)
        col = torch.full((248320,), -1, dtype=torch.int64, device=dev)
        col[ids] = torch.arange(len(ids), device=dev)
        self.col = col
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- pieces ------------------------------------------------------------------------------------------
    def W(self, name: str) -> torch.Tensor:
        k = key(name)
        if k not in self.lin:
            return getattr(self, "frozen_" + k)
        w = self.lin[k]
        if name in self.grids:                                   # the 4-bit grid, straight through
            s, b = self.grids[name]
            wq = torch.clamp(torch.round((w.detach() - b) / s), 0, 15) * s + b
            return (w + (wq - w).detach()).to(torch.bfloat16)
        return w.to(torch.bfloat16)                              # bf16 rounding, straight through

    def N(self, name: str) -> torch.Tensor:
        return 1.0 + self.norms[key(name)].to(torch.bfloat16).float()

    def L(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x.to(torch.bfloat16), self.W(name))

    def tensors(self) -> dict[str, torch.Tensor]:
        """The trained canonical tensors (what export writes)."""

        out = {n: self.lin[key(n)].detach() for n in SRC.LINEARS if key(n) in self.lin}
        out.update({n: self.norms[key(n)].detach() for n in SRC.NORMS})
        return out

    def export(self, path: Path, meta: dict | None = None) -> None:
        md = {"format": "tensorfold-mtp-head", "item": "D", "source": self.src.kind, "checkpoint": str(self.src.dir)}
        md.update(meta or {})
        self.src.export(self.tensors(), Path(path), md)

    def hc(self, h: torch.Tensor, base: str, inject: bool):
        normed = rms(h, self.N(base + ".hc_norm.weight"), group=D)
        mix = F.silu(self.L(normed, base + ".input_mix_weight_down").float() / S)
        mix = torch.sigmoid(self.L(mix, base + ".input_mix_weight_up").float())
        mixed = (mix.view(-1, S, D) * normed.float().view(-1, S, D)).mean(1).to(torch.bfloat16)
        if not inject:
            return mixed
        inj = 2 * torch.sigmoid(self.L(normed, base + ".block_inject_weight").float() / S)
        return mixed, inj

    @staticmethod
    def write_back(h, branch, inj):
        add = (branch.float()[:, None, :] * inj[:, :, None]).to(torch.bfloat16)
        return (h.float().view(-1, S, D) + add.float()).to(torch.bfloat16).view(h.shape)

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        probs = torch.softmax(F.linear(x.float(), self.W("layers.0.mlp.gate").float()), dim=-1)
        wts, ex = torch.topk(probs, 10, dim=-1)
        wts = wts / wts.sum(-1, keepdim=True)
        flat = ex.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=E).tolist()
        rows = order // 10
        xs = x[rows]
        outs = []
        at = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            xe = xs[at:at + c]
            g = F.linear(xe, self.eg[e])
            u = F.linear(xe, self.eu[e])
            outs.append(F.linear((F.silu(g.float()) * u.float()).to(torch.bfloat16), self.ed[e]))
            at += c
        y = torch.cat(outs)
        wsorted = wts.reshape(-1)[order]
        routed = torch.zeros((T, D), dtype=torch.float32, device=x.device).index_add(0, rows, y.float() * wsorted[:, None])
        sg = self.L(x, "layers.0.mlp.shared_expert.gate_proj")
        su = self.L(x, "layers.0.mlp.shared_expert.up_proj")
        sh = self.L((F.silu(sg.float()) * su.float()).to(torch.bfloat16), "layers.0.mlp.shared_expert.down_proj")
        gate = torch.sigmoid(F.linear(x.float(), self.W("layers.0.mlp.shared_expert_gate").float()))
        return (routed + sh.float() * gate).to(torch.bfloat16)

    def attn_proj(self, x, pos):
        T = x.shape[0]
        q = self.L(x, "layers.0.self_attn.q_proj").view(T, H, 2 * HD)
        qq, gate = q[..., :HD], q[..., HD:].reshape(T, H * HD)
        qq = rope(rms(qq, self.N("layers.0.self_attn.q_norm.weight")), pos)
        k = rope(rms(self.L(x, "layers.0.self_attn.k_proj").view(T, KV, HD), self.N("layers.0.self_attn.k_norm.weight")),
                 pos)
        v = self.L(x, "layers.0.self_attn.v_proj").view(T, KV, HD)
        return qq, gate, k, v

    def forward(self, streams, tokens, seq_id, pos, depth: int, valid: list[torch.Tensor]):
        """Pass d (1..depth) at row i takes token i+d and predicts token i+d+1; each pass's mixed hidden [T, D].
        Chained passes attend to pass-1 rows up to i plus the chain's own rows at i (decode's cache)."""

        T = streams.shape[0]
        outs = []
        ks, vs = [], []
        h_in = streams
        same = seq_id[:, None] == seq_id[None, :]
        causal = same & (pos[None, :] <= pos[:, None])
        eye = torch.eye(T, dtype=torch.bool, device=streams.device)
        for d in range(1, depth + 1):
            idx = torch.clamp(torch.arange(T, device=streams.device) + d, max=T - 1)
            nxt = torch.where(valid[d - 1], tokens[idx], torch.zeros_like(tokens))
            e = self.L(rms(self.src.embed_rows(nxt), self.N("pre_fc_norm_embedding.weight")), "fc_embedding")
            hn = rms(h_in, self.N("pre_fc_norm_hidden.weight"))
            hs = self.L(hn.view(T * S, D), "fc_hidden").view(T, S, D)
            x = (e.float()[:, None, :] + hs.float()).to(torch.bfloat16).view(T, S * D)
            mixed, inj = self.hc(x, "layers.0.attn_hyper_connection", True)
            q, gate, k, v = self.attn_proj(mixed, pos + (d - 1))
            ks.append(k)
            vs.append(v)
            K = torch.cat(ks, 0)
            V = torch.cat(vs, 0)
            mask = torch.cat([causal] + [eye] * (d - 1), dim=1)
            qh = q.permute(1, 0, 2)[None]
            kh = K.permute(1, 0, 2)[None].repeat_interleave(H // KV, dim=1)
            vh = V.permute(1, 0, 2)[None].repeat_interleave(H // KV, dim=1)
            o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None], scale=HD ** -0.5)
            o = o[0].permute(1, 0, 2).reshape(T, H * HD)
            branch = self.L((o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16), "layers.0.self_attn.o_proj")
            x = self.write_back(x, branch, inj)
            mixed, inj = self.hc(x, "layers.0.mlp_hyper_connection", True)
            x = self.write_back(x, self.moe(mixed), inj)
            outs.append(self.hc(x, "hyper_connection_mixer", False))
            h_in = x
        return outs

    def logits(self, mixed: torch.Tensor) -> torch.Tensor:
        return F.linear(mixed, self.head).float()


IH, ID, RATIO, BUDGET = 4, 128, 4, 2048
TOP = BUDGET // RATIO
QCHUNK = 256


def _attend(q, gate_unused, K, V, mask):
    """q [n, H, HD] bf16, K/V [k, KV, HD], mask [n, k] -> [n, H*HD] (GQA)."""

    o = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], K.permute(1, 0, 2)[None], V.permute(1, 0, 2)[None],
                                       attn_mask=mask[None, None], scale=HD ** -0.5, enable_gqa=True)
    return o[0].permute(1, 0, 2).reshape(q.shape[0], H * HD)


def long_forward(m: "MTPHead", streams, tokens, pos, S: torch.Tensor, depth: int):
    """The head over one long sequence (T rows, the main model's streams), queries only at rows S (sorted, with
    S + depth < T): pass 1 projects keys/values for every row, queries for S; a row past BUDGET keys attends to the
    indexer's best TOP blocks of RATIO keys plus its tail (``Indexer.select``), else causally; chained pass d at row
    i adds the chain's own rows at i. Returns each pass's mixed hidden at S [|S|, D]."""

    from torch.utils.checkpoint import checkpoint

    T = streams.shape[0]
    dev = streams.device
    nxt = torch.cat([tokens[1:], tokens[-1:]])
    e = m.L(rms(m.src.embed_rows(nxt), m.N("pre_fc_norm_embedding.weight")), "fc_embedding")
    hs = m.L(rms(streams, m.N("pre_fc_norm_hidden.weight")).view(T * S_, D), "fc_hidden").view(T, S_, D)
    x = (e.float()[:, None, :] + hs.float()).to(torch.bfloat16).view(T, S_ * D)
    del e, hs
    mixed, inj = m.hc(x, "layers.0.attn_hyper_connection", True)
    K = rope(rms(m.L(mixed, "layers.0.self_attn.k_proj").view(T, KV, HD), m.N("layers.0.self_attn.k_norm.weight")), pos)
    V = m.L(mixed, "layers.0.self_attn.v_proj").view(T, KV, HD)
    # indexer selection (no gradient: a discrete choice); masks [|S|, T]
    with torch.no_grad():
        qk = F.linear(mixed.detach(), m.idx_w).view(T, IH + 1, ID)
        ikey = qk[:, IH]
        nb = T // RATIO
        pooled = ikey[:nb * RATIO].float().view(nb, RATIO, ID).mean(1).to(torch.bfloat16)
        pooled = rope(rms(pooled, m.idx_kn)[:, None], pos[:nb * RATIO:RATIO])[:, 0].float()     # block b at b*RATIO
        iq = rope(rms(qk[S, :IH], m.idx_qn), pos[S]).float()                                   # [|S|, IH, ID]
        ends = (S - S.new_tensor(0)) + 1                                                         # keys a row reads
        complete = ends // RATIO
        keys = torch.arange(T, device=dev)
        masks = []
        for c0 in range(0, len(S), QCHUNK):
            sl = slice(c0, c0 + QCHUNK)
            causal = keys[None, :] < ends[sl, None]
            sc = torch.relu(torch.einsum("shd,bd->shb", iq[sl], pooled)).sum(1) / math.sqrt(ID)
            sc = torch.where(torch.arange(nb, device=dev)[None, :] < complete[sl, None], sc, float("-inf"))
            k_top = min(TOP, nb)
            chosen = torch.topk(sc, k_top, dim=-1).indices
            hit = torch.zeros((sc.shape[0], nb), dtype=torch.bool, device=dev).scatter_(1, chosen, True)
            picked = torch.zeros((sc.shape[0], T), dtype=torch.bool, device=dev)
            picked[:, :nb * RATIO] = hit.repeat_interleave(RATIO, dim=1)
            tail = (keys[None, :] >= (complete[sl] * RATIO)[:, None]) & causal
            sparse = (complete[sl] > TOP)[:, None]
            masks.append(torch.where(sparse, (picked & causal) | tail, causal))
        mask1 = torch.cat(masks)
    outs = []
    chainK, chainV = [], []
    h_in = None
    for d in range(1, depth + 1):
        if d == 1:
            xs, mx_s, inj_s = x[S], mixed[S], inj[S]
        else:
            nx = tokens[S + d]
            e = m.L(rms(m.src.embed_rows(nx), m.N("pre_fc_norm_embedding.weight")), "fc_embedding")
            hs = m.L(rms(h_in, m.N("pre_fc_norm_hidden.weight")).view(-1, D), "fc_hidden").view(-1, S_, D)
            xs = (e.float()[:, None, :] + hs.float()).to(torch.bfloat16).view(-1, S_ * D)
            mx_s, inj_s = m.hc(xs, "layers.0.attn_hyper_connection", True)
        p = pos[S] + (d - 1)
        q, gate, kc, vc = m.attn_proj(mx_s, p)
        if d > 1:
            chainK.append(kc)
            chainV.append(vc)
        Kc = torch.cat([K] + chainK, 0)
        Vc = torch.cat([V] + chainV, 0)
        n = len(S)
        os_ = []
        for c0 in range(0, n, QCHUNK):
            sl = slice(c0, c0 + QCHUNK)
            r = torch.arange(c0, min(n, c0 + QCHUNK), device=dev)
            eye = [(torch.arange(n, device=dev)[None, :] == r[:, None]) for _ in chainK]
            mk = torch.cat([mask1[sl]] + eye, 1)
            os_.append(checkpoint(_attend, q[sl], None, Kc, Vc, mk, use_reentrant=False))
        o = torch.cat(os_)
        branch = m.L((o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16), "layers.0.self_attn.o_proj")
        xs = m.write_back(xs, branch, inj_s)
        mx2, inj2 = m.hc(xs, "layers.0.mlp_hyper_connection", True)
        xs = m.write_back(xs, m.moe(mx2), inj2)
        outs.append(m.hc(xs, "hyper_connection_mixer", False))
        h_in = xs
    return outs


S_ = S


def trainable(m: MTPHead):
    return list(m.lin.parameters()) + list(m.norms.parameters())
