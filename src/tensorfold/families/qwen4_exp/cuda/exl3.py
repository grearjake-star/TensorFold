"""Flash Next from an EXL3 pack into the MLX path's ``Weights``: trellis matrices, fp16 tensors, routed experts at their own widths."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3 import format as fmt

from .exl3_mm import F8, Scratch, f16, hc_fp8, stack, x3
from .exl3_pack import _DT, NgramTable, Pack, is_exl3

def _prefill_rows() -> int:
    """The prompt pieces' rows the n-gram staging holds: 2048, or TENSORFOLD_PREFILL_ROWS when larger."""

    from tensorfold.cuda.geometry import indexed_prefill_rows

    return max(2048, indexed_prefill_rows() or 0)


PREFILL_ROWS = _prefill_rows()

__all__ = ["is_exl3", "load"]


def expert_table(pk: Pack, prefix: str, count: int, shared: str, device) -> x3experts.Exl3RoutedExperts:
    """Experts ``prefix.{0..count-1}`` and the shared expert (as expert ``count``), trellises read in large runs into one buffer."""

    names = [f"{prefix}.{e}" for e in range(count)] + [shared]
    projs = ("gate_proj", "up_proj", "down_proj")
    codebooks = {pk.codebook(f"{nm}.{p}") for nm in names for p in projs}
    if len(codebooks) > 1:
        raise ValueError(f"{prefix}: the experts mix EXL3 codebooks ({', '.join(sorted(codebooks))}); the grouped "
                         "expert kernel takes one codebook a layer")
    parts = {nm + "." + p: (("suh" if pk.has(f"{nm}.{p}.suh") else "su"), ("svh" if pk.has(f"{nm}.{p}.svh") else "sv"))
             for nm in names for p in projs}
    entries = {f"{m}.{part}": pk.entry(f"{m}.{part}") for m, (i, o) in parts.items() for part in ("trellis", i, o)}
    place, total = {}, 0
    for k in entries:
        if k.endswith(".trellis"):
            place[k] = total
            total += -(-(entries[k][2] - entries[k][1]) // 256) * 256
    big = torch.empty((total,), dtype=torch.uint8, device=device)
    small: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key, (file, *_rest) in entries.items():
        by_file.setdefault(file, []).append(key)
    for file, keys in by_file.items():
        keys.sort(key=lambda k: entries[k][1])
        run: list[str] = []

        def flush(run: list[str]) -> None:
            if run:
                b0, b1 = entries[run[0]][1], max(entries[k][2] for k in run)
                host = pk.read(file, b0, b1)
                dev = host.to(device)
                for k in run:
                    _, b, e, dtype, shape = entries[k]
                    if k.endswith(".trellis"):
                        big[place[k]:place[k] + (e - b)].copy_(dev[b - b0:e - b0])
                    else:
                        small[k] = host[b - b0:e - b0].clone().view(_DT[dtype]).reshape(shape)
                del dev, host

        for k in keys:                     # a run spans at most ~2 GB and skips at most 16 MB of other tensors
            if run and (entries[k][2] - entries[run[0]][1] > (2 << 30)
                        or entries[k][1] - max(entries[j][2] for j in run[-4:]) > (16 << 20)):
                flush(run)
                run = []
            run.append(k)
        flush(run)

    def trellis(k: str) -> torch.Tensor:
        _, b, e, dtype, shape = entries[k]
        if dtype != "I16":
            raise ValueError(f"{k}: trellis dtype {dtype}")
        return big[place[k]:place[k] + (e - b)].view(torch.int16).view(shape)

    def scales(m: str, part: str) -> torch.Tensor:
        t = small[f"{m}.{part}"]
        return (t if part in ("suh", "svh") else torch.from_numpy(fmt.unpack_signs(t.numpy()))).to(device)

    lists = {p: [(trellis(f"{nm}.{p}.trellis"), scales(f"{nm}.{p}", parts[f"{nm}.{p}"][0]),
                  scales(f"{nm}.{p}", parts[f"{nm}.{p}"][1])) for nm in names] for p in projs}
    ex = x3experts.prepare(lists["gate_proj"], lists["up_proj"], lists["down_proj"], codebooks.pop(), device=device)
    ex.keep.append(big)
    return ex


def centred_offset(pk: Pack, names: list[str]) -> float:
    """1.0 when the pack stores the centred norms as gamma - 1 (every EXL3 pack seen), 0.0 when as gamma."""

    means = np.array([float(pk.get(n).float().mean()) for n in names if pk.has(n)])
    if not len(means):
        return 1.0
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    if around_zero == around_one:
        raise ValueError(f"cannot tell how the pack stores its norm weights (median mean {np.median(means):.3f})")
    return 1.0 if around_zero else 0.0


def requant_rows(head, ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head's rows ``ids`` decoded through its own EXL3 linear, as MLX 4-bit groups of 32: the draft head (drafts only)."""

    from .exl3_mm import ROWS

    k = head.k
    rows = torch.empty((len(ids), k), dtype=torch.float32, device=device)
    eye = torch.eye(ROWS, dtype=torch.bfloat16, device=device)
    out = torch.empty((ROWS, head.n), dtype=torch.float32, device=device)
    for k0 in range(0, k, ROWS):
        x = torch.zeros((ROWS, k), dtype=torch.bfloat16, device=device)
        x[:, k0:k0 + ROWS] = eye
        head(x, out)
        rows[:, k0:k0 + ROWS] = out[:, ids].t()
    g = rows.view(len(ids), k // 32, 32)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int64)
    words = (q.view(len(ids), k // 8, 8) << (torch.arange(8, device=device, dtype=torch.int64) * 4)).sum(dim=-1)
    words = words & 0xFFFFFFFF
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), scale.contiguous(), bias.contiguous()


# MTP tensors a trained head (TF_MTP_HEAD) may replace on an EXL3 pack: trellis linears become fp16 matrices of the
# head's values, the rest stay tensors. The shared and routed experts (one grouped EXL3 table) and the indexer stay the
# pack's own.
HEAD_LINEARS = ("mtp.fc_embedding", "mtp.fc_hidden", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj",
                "mtp.layers.0.self_attn.v_proj", "mtp.layers.0.self_attn.o_proj")


def exl3_head(pk: Pack, path) -> dict[str, torch.Tensor]:
    """A trained MTP head for an EXL3 pack (``TF_MTP_HEAD``), in the bf16 format (``mtp.*`` names; linears as
    ``.weight`` [out, in], norms stored centred: scale = 1 + w). Each tensor is checked against the pack: a trellis
    linear by its input/output sizes (suh/svh), any other tensor by its shape; both floating. Anything else (an MLX
    triple, an expert, the indexer, a non-MTP name) is refused. Drafts change speed only, never the output."""

    from .weights import is_mlx_head, mtp_head_override

    head = mtp_head_override(path)
    if is_mlx_head(head):
        raise ValueError(f"{path}: an MLX-format head (4-bit triples); an EXL3 pack takes the bf16 head format")
    for name, t in head.items():
        if not t.dtype.is_floating_point:
            raise ValueError(f"{path}: {name} is {t.dtype}; an EXL3 pack takes floating tensors")
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem in HEAD_LINEARS and pk.has(stem + ".trellis"):
            k = pk.entry(stem + ".suh" if pk.has(stem + ".suh") else stem + ".su")[4]
            n = pk.entry(stem + ".svh" if pk.has(stem + ".svh") else stem + ".sv")[4]
            want = [n[0], k[0]]
        elif ".experts." in name or ".shared_expert." in name or ".indexer." in name or not pk.has(name):
            raise ValueError(f"{path}: {name} is not an MTP tensor an EXL3 pack's head can replace")
        else:
            want = list(pk.entry(name)[4])
        if list(t.shape) != want:
            raise ValueError(f"{path}: {name} is {list(t.shape)}, the pack's is {want}")
    return head


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None, table_reads: list | None = None, mtp_head=None):
    from .qmm import make_q4
    from tensorfold.cuda.direct_read import in_background

    from .weights import GDNW, HC, AttnW, Config, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids

    if tp is not None and tp[1] > 1:
        raise ValueError("EXL3 packs of Flash Next run on one GPU; two ranks read the MLX checkpoint")
    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    pk = Pack(model_dir)
    sc = Scratch(cfg.top_k + 1)
    T = "model.language_model."
    t0 = time.time()
    offset = centred_offset(pk, [f"{T}layers.{i}.attn_hyper_connection.hc_norm.weight" for i in range(cfg.layers)])
    override = exl3_head(pk, mtp_head) if mtp and mtp_head else {}
    used: set[str] = set()

    def get(name: str) -> torch.Tensor:               # a pack tensor, or the trained head's (MTP names only)
        if name in override:
            used.add(name)
            return override[name]
        return pk.get(name)

    def lin(name: str):                               # a trellis linear, or the trained head's values as fp16
        if name + ".weight" in override:
            used.add(name + ".weight")
            return f16(sc, [override[name + ".weight"]], device)
        return x3(sc, pk, name, device)

    def plain(name: str) -> torch.Tensor:
        return pk.get(name).to(device)

    def centred(name: str) -> torch.Tensor:
        if name in override:                          # the head file stores centred norms (scale = 1 + w)
            used.add(name)
            return (override[name].float() + 1.0).to(device).contiguous()
        return (pk.get(name).float() + offset).to(device).contiguous()

    def hc(name: str, inject: bool) -> HC:
        rows = [get(name + ".input_mix_weight_down.weight")]
        if inject:
            rows.append(get(name + ".block_inject_weight.weight"))
        down, up = f16(sc, rows, device), f16(sc, [get(name + ".input_mix_weight_up.weight")], device)
        if hc_fp8():                    # opt-in, numerics-changing: decode reads MXFP8 copies, prompts the fp16 rows
            return HC(F8(down), F8(up), centred(name + ".hc_norm.weight"), inject, down, up)
        return HC(down, up, centred(name + ".hc_norm.weight"), inject, down, up)

    def moe(name: str) -> MoEW:
        router = torch.cat([get(name + ".gate.weight").to(torch.bfloat16),
                            get(name + ".shared_expert_gate.weight").to(torch.bfloat16)]).to(device).contiguous()
        return MoEW(router, expert_table(pk, name + ".experts", cfg.experts, name + ".shared_expert", device))

    def attention(name: str) -> AttnW:
        proj = stack(sc, [lin(name + p) for p in (".q_proj", ".k_proj", ".v_proj")]
                     + [x3(sc, pk, name + ".indexer.index_qk_proj", device)])
        return AttnW(proj, centred(name + ".q_norm.weight"), centred(name + ".k_norm.weight"),
                     centred(name + ".indexer.q_layernorm.weight"), centred(name + ".indexer.k_layernorm.weight"),
                     lin(name + ".o_proj"))

    def gdn(name: str) -> GDNW:
        proj = stack(sc, [x3(sc, pk, name + ".in_proj_qkv", device), x3(sc, pk, name + ".in_proj_z", device),
                          f16(sc, [pk.get(name + ".in_proj_b.weight"), pk.get(name + ".in_proj_a.weight")], device)])
        conv = plain(name + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).to(torch.bfloat16).contiguous()
        return GDNW(proj, conv, plain(name + ".A_log").float().contiguous(),
                    plain(name + ".dt_bias").float().contiguous(),
                    plain(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    x3(sc, pk, name + ".out_proj", device))

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        table = NgramTable(pk, name + ".ple_embedding.ngram_embedding.", cfg.ngram_shards, device)
        ngram.check(table.multipliers, table.head_offsets, table.head_sizes)
        if table.rows != ngram.rows or table.dh != ngram.dims:
            raise ValueError(f"n-gram tables of {table.rows} rows of {table.dh} values; the config gives "
                             f"{ngram.rows} of {ngram.dims}")
        if table_reads is not None:                   # its pages come in while the weights load
            in_background(table.prefetch, table_reads)    # the caller waits for it (``wait_all``)
        conv = plain(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, f16(sc, [pk.get(name + ".key_proj.weight")], device),
                    f16(sc, [pk.get(name + ".value_proj.weight")], device), centred(name + ".norm_key.weight"),
                    centred(name + ".norm_query.weight"), centred(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True), hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    embed = plain(T + "embed_tokens.weight")
    if embed.dtype not in (torch.bfloat16, torch.float16):
        embed = embed.to(torch.bfloat16)
    loaded = []
    for i in range(cfg.layers):
        loaded.append(layer(i, f"{T}layers.{i}", cfg.layer_types[i], True))
        pk.release()
        if i % 8 == 7:                            # each release waits for the device; a layer leaves few temporaries
            torch.cuda.empty_cache()
    mixer = hc(T + "hyper_connection_mixer", False)
    head = x3(sc, pk, "lm_head", device, head=True)
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
        -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
    w = Weights(cfg, (embed.contiguous(),), loaded, mixer, head, inv.to(torch.float32).to(device), around_one=True)
    w.meta.update(rank=0, world=1, vocab_offset=0, full=cfg, centred_offset=offset)
    if mtp and pk.has("mtp.fc_embedding.trellis"):
        w.mtp = MTPW(centred("mtp.pre_fc_norm_embedding.weight"), centred("mtp.pre_fc_norm_hidden.weight"),
                     lin("mtp.fc_embedding"), lin("mtp.fc_hidden"),
                     layer(-1, "mtp.layers.0", "attention", False), hc("mtp.hyper_connection_mixer", False))
        if override:
            unused = sorted(set(override) - used)
            if unused:
                raise ValueError(f"{mtp_head}: tensors the MTP head does not read: {unused[:5]}")
            w.meta["mtp_head"] = str(mtp_head)
            print(f"[tensorfold] MTP head: {len(used)} tensors from {mtp_head} (EXL3 pack: linears as fp16)", flush=True)
    ple = next((lay.ple for lay in loaded if lay.ple is not None), None)
    sc.allocate(device, experts=loaded[0].moe.experts, rows=PREFILL_ROWS,
                ple_words=ple.table.words_per_row if ple else 0, ple_heads=ple.ngram.heads if ple else 0,
                ple_dim=cfg.ple_dim)
    w.x3 = sc
    ids = draft_token_ids(draft_vocab)
    if ids is not None and w.mtp is not None:
        ids = ids[ids < cfg.vocab]
        w.draft_ids = torch.from_numpy(ids).to(device)
        w.draft_head = make_q4(*requant_rows(head, w.draft_ids, device))
    pk.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
