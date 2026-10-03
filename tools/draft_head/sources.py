"""Where a draft head's weights come from and how a trained head goes back, per checkpoint format (workstream D).

The PyTorch head (mtp_head.py) trains in one canonical space for every checkpoint: each MTP linear as its effective
fp32 matrix [out, in], norms centred (the engine scales by 1 + w), the router and shared-expert gate as rows. A Source
reads that space from a checkpoint (CPU, headers + the tensors it needs) and gives:
  linear(name), norm(name)            init values of the trainable tensors (canonical space)
  experts()                           the frozen routed experts as bf16 [E, 640, 2560] x2 and [E, 2560, 640]
  embed_rows(tokens)                  the frozen embedding rows (bf16)
  draft_rows(ids)                     the draft head the engine builds (``w.draft_head``), dequantized (bf16)
  trainable                           names the format's loader lets a head replace
  grid(name)                          for a 4-bit target: the checkpoint's per-group (scale, bias) to train on
  export(tensors, path, meta)         a TF_MTP_HEAD file the format's loader takes
Formats:
  modelopt  NVFP4 ModelOpt exports (hibrid48, local-inference-lab): bf16 linears and router, centred bf16 norms,
            NVFP4 routed experts, lm_head NVFP4 (own file) or bf16. Export: bf16 under the checkpoint's names.
  mlx4      MLX 4-bit g32 (the MLX checkpoint; v1's head): triples, norms around one, switch_mlp experts. Export:
            4-bit triples on the checkpoint's own grid (quantization-aware training via grid()), norms around one.
  exl3      EXL3 packs: trellis linears (fc, attention, experts), fp16 HC/router, norms gamma - 1 (or gamma).
            Export: the bf16 head format (exl3.py turns its linears into fp16 matrices); the shared expert lives in
            the pack's grouped expert table, so it stays frozen.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

D, E = 2560, 512
LINEARS = ["fc_embedding", "fc_hidden",
           "layers.0.attn_hyper_connection.input_mix_weight_down", "layers.0.attn_hyper_connection.input_mix_weight_up",
           "layers.0.attn_hyper_connection.block_inject_weight",
           "layers.0.mlp_hyper_connection.input_mix_weight_down", "layers.0.mlp_hyper_connection.input_mix_weight_up",
           "layers.0.mlp_hyper_connection.block_inject_weight",
           "hyper_connection_mixer.input_mix_weight_down", "hyper_connection_mixer.input_mix_weight_up",
           "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj",
           "layers.0.self_attn.o_proj",
           "layers.0.mlp.shared_expert.gate_proj", "layers.0.mlp.shared_expert.up_proj",
           "layers.0.mlp.shared_expert.down_proj", "layers.0.mlp.shared_expert_gate",
           "layers.0.mlp.gate"]                                   # the router
NORMS = ["pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
         "layers.0.attn_hyper_connection.hc_norm.weight", "layers.0.mlp_hyper_connection.hc_norm.weight",
         "hyper_connection_mixer.hc_norm.weight", "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight"]
SHARED = ("layers.0.mlp.shared_expert.gate_proj", "layers.0.mlp.shared_expert.up_proj",
          "layers.0.mlp.shared_expert.down_proj")

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.uint8,
       "I32": torch.int32, "U32": torch.int32, "I16": torch.int16}


class Headers:
    """Tensors by name from a checkpoint's safetensors files (every file's header; index not required)."""

    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        self.at: dict[str, tuple[Path, int, dict]] = {}
        for f in sorted(self.dir.glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.at[k] = (f, 8 + n, v)

    def has(self, name: str) -> bool:
        return name in self.at

    def shape(self, name: str) -> list[int]:
        return list(self.at[name][2]["shape"])

    def get(self, name: str) -> torch.Tensor:
        f, base, ent = self.at[name]
        b0, b1 = ent["data_offsets"]
        raw = np.fromfile(f, dtype=np.uint8, count=b1 - b0, offset=base + b0)
        return torch.from_numpy(raw.copy()).view(_DT[ent["dtype"]]).reshape(ent["shape"])


def detect(model_dir: Path) -> str:
    h = Headers(model_dir)
    if h.has("mtp.fc_hidden.trellis"):
        return "exl3"
    if h.has("language_model.mtp.fc_hidden.scales"):
        return "mlx4"
    if h.has("mtp.fc_hidden.weight"):
        return "modelopt"
    raise ValueError(f"{model_dir}: no Flash Next MTP head in a known format")


def requant4(w: torch.Tensor, round_first: bool = False) -> torch.Tensor:
    """Rows -> MLX 4-bit g32 affine -> fp32 values, as the loaders build draft heads: ``bf16.quantize4`` (codes from
    fp32 scale/min, stored bf16) or, ``round_first``, ``exl3.requant_rows`` (scale/min rounded to bf16 first)."""

    n, k = w.shape
    g = w.float().reshape(n, k // 32, 32)
    mn = g.min(dim=-1, keepdim=True).values
    mx = g.max(dim=-1, keepdim=True).values
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    if round_first:
        s, mn = s.to(torch.bfloat16).float(), mn.to(torch.bfloat16).float()
    q = ((g - mn) / s).round().clamp(0, 15)
    return (q * s.to(torch.bfloat16).float() + mn.to(torch.bfloat16).float()).reshape(n, k)


def mlx_deq(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    k8 = words.shape[-1]
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).float()
    return q * scales.float().repeat_interleave(32, -1) + biases.float().repeat_interleave(32, -1)


def mlx_pack(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.int64).reshape(*codes.shape[:-1], codes.shape[-1] // 8, 8)
    w = (c << (torch.arange(8, device=codes.device, dtype=torch.int64) * 4)).sum(-1)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


class Source:
    kind = ""
    pre = "mtp."

    def __init__(self, model_dir: Path, device="cuda") -> None:
        self.dir, self.h, self.device = Path(model_dir), Headers(model_dir), device

    @property
    def trainable(self) -> list[str]:
        return list(LINEARS)

    def t(self, name: str) -> torch.Tensor:
        return self.h.get(self.pre + name).to(self.device)

    def grid(self, name: str):
        return None

    def norm(self, name: str) -> torch.Tensor:
        return self.t(name).float()                              # centred in modelopt/exl3 (exl3 overrides)

    def embed_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.emb[tokens]

    def read_head(self, path: Path) -> dict[str, torch.Tensor]:
        """A head file this format exported -> canonical tensors {name: fp32} (linears [out, in], centred norms)."""

        from safetensors.torch import load_file

        sd = load_file(str(path), device="cpu")
        out = {}
        for k, v in sd.items():
            n = k[len("language_model."):] if k.startswith("language_model.") else k
            n = n[len("mtp."):]
            if n.endswith((".scales", ".biases")):
                continue
            if n in NORMS:
                out[n] = v.float() - (1.0 if self.kind == "mlx4" else getattr(self, "one", 0.0))
                continue
            stem = n[:-len(".weight")]
            sc = k[:-len(".weight")]
            if sc + ".scales" in sd:
                out[stem] = mlx_deq(v, sd[sc + ".scales"], sd[sc + ".biases"])
            else:
                out[stem] = v.float()
        return {k: v.to(self.device) for k, v in out.items()}


class ModelOpt(Source):
    """NVFP4 ModelOpt exports, and the bf16 original (same names, bf16 experts and head)."""

    kind = "modelopt"

    def __init__(self, model_dir: Path, device="cuda") -> None:
        super().__init__(model_dir, device)
        self.emb = self.h.get("model.language_model.embed_tokens.weight").to(device)
        means = [float(self.h.get(n).float().mean()) for n in (f"model.language_model.layers.{i}.attn_hyper_connection."
                                                              "hc_norm.weight" for i in range(0, 48, 6)) if self.h.has(n)]
        self.one = 1.0 if means and np.median(means) > 0.5 else 0.0   # norms stored around one (gamma) or centred

    def norm(self, name: str) -> torch.Tensor:
        return self.t(name).float() - self.one

    def linear(self, name: str) -> torch.Tensor:
        return self.t(name + ".weight").float()

    def _fp4(self, base: str, rows=None) -> torch.Tensor:
        from tensorfold.families.qwen4_exp.cuda import nvfp4

        w, s = self.h.get(base + ".weight").to(self.device), self.h.get(base + ".weight_scale").to(self.device)
        if rows is not None:
            w, s = w.index_select(0, rows), s.index_select(0, rows)
        return nvfp4.dequantize(w, s, self.h.get(base + ".weight_scale_2"))

    def experts(self):
        eg = torch.empty((E, 640, D), dtype=torch.bfloat16, device=self.device)
        eu, ed = torch.empty_like(eg), torch.empty((E, D, 640), dtype=torch.bfloat16, device=self.device)
        base = f"{self.pre}layers.0.mlp.experts"
        if self.h.has(base + ".gate_up_proj"):                     # stacked bf16 experts [E, 2*NI, D], [E, D, NI]
            gu, dn = self.h.get(base + ".gate_up_proj"), self.h.get(base + ".down_proj")
            eg.copy_(gu[:, :640].to(torch.bfloat16))
            eu.copy_(gu[:, 640:].to(torch.bfloat16))
            ed.copy_(dn.to(torch.bfloat16))
            return eg, eu, ed
        for e in range(E):
            for dst, proj in ((eg, "gate_proj"), (eu, "up_proj"), (ed, "down_proj")):
                b = f"{base}.{e}.{proj}"
                dst[e] = (self._fp4(b) if self.h.has(b + ".weight_scale_2") else self.h.get(b + ".weight")).to(
                    torch.bfloat16)
        return eg, eu, ed

    def draft_rows(self, ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty((len(ids), D), dtype=torch.bfloat16, device=self.device)
        fp4 = self.h.has("lm_head.weight_scale_2")
        full = None if fp4 else self.h.get("lm_head.weight")
        for i0 in range(0, len(ids), 8192):
            sel = ids[i0:i0 + 8192]
            rows = (self._fp4("lm_head", sel) if fp4 else full.index_select(0, sel.cpu()).to(self.device))
            out[i0:i0 + 8192] = requant4(rows.to(torch.bfloat16)).to(torch.bfloat16)
        return out

    def export(self, tensors: dict, path: Path, meta: dict) -> None:
        from safetensors.torch import save_file

        out = {self.pre + n + ".weight": tensors[n].to(torch.bfloat16).cpu().contiguous() for n in LINEARS if n in tensors}
        out.update({self.pre + n: (tensors[n].float() + self.one).to(torch.bfloat16).cpu().contiguous()
                    for n in NORMS if n in tensors})
        save_file(out, str(path), metadata=meta)


class MLX4(Source):
    kind = "mlx4"
    pre = "language_model.mtp."

    def __init__(self, model_dir: Path, device="cuda") -> None:
        super().__init__(model_dir, device)
        p = "language_model.model.embed_tokens."
        self.embq = tuple(self.h.get(p + x).to(device) for x in ("weight", "scales", "biases"))

    def triple(self, name: str):
        return tuple(self.t(name + x) for x in (".weight", ".scales", ".biases"))

    def linear(self, name: str) -> torch.Tensor:
        if self.h.has(self.pre + name + ".scales"):
            return mlx_deq(*self.triple(name))
        return self.t(name + ".weight").float()                  # the router is bf16

    def grid(self, name: str):
        if not self.h.has(self.pre + name + ".scales"):
            return None
        _, s, b = self.triple(name)
        return s, b

    def norm(self, name: str) -> torch.Tensor:
        return self.t(name).float() - 1.0                        # stored around one

    def experts(self):
        out = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w, s, b = (self.h.get(f"{self.pre}layers.0.mlp.switch_mlp.{proj}{x}") for x in (".weight", ".scales", ".biases"))
            t = torch.empty((w.shape[0], w.shape[1], w.shape[2] * 8), dtype=torch.bfloat16, device=self.device)
            for e0 in range(0, w.shape[0], 64):
                t[e0:e0 + 64] = mlx_deq(*(x[e0:e0 + 64].to(self.device) for x in (w, s, b))).to(torch.bfloat16)
            out.append(t)
        return tuple(out)

    def embed_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        w, s, b = self.embq
        return mlx_deq(w[tokens], s[tokens], b[tokens]).to(torch.bfloat16)

    def draft_rows(self, ids: torch.Tensor) -> torch.Tensor:
        w, s, b = (self.h.get("language_model.lm_head" + x) for x in (".weight", ".scales", ".biases"))
        sel = ids.cpu()
        out = torch.empty((len(ids), D), dtype=torch.bfloat16, device=self.device)
        for i0 in range(0, len(sel), 8192):
            at = sel[i0:i0 + 8192]
            out[i0:i0 + 8192] = mlx_deq(w[at].to(self.device), s[at].to(self.device), b[at].to(self.device)).to(torch.bfloat16)
        return out

    def export(self, tensors: dict, path: Path, meta: dict) -> None:
        """4-bit triples on the checkpoint's grid (the codes training's straight-through forward used), norms
        around one, the router bf16: v1's format (the v1 head)."""

        from safetensors.torch import save_file

        out = {}
        for n in LINEARS:
            if n not in tensors:
                continue
            g = self.grid(n)
            if g is None:
                out[self.pre + n + ".weight"] = tensors[n].to(torch.bfloat16).cpu().contiguous()
                continue
            s, b = g
            sx, bx = s.float().repeat_interleave(32, -1), b.float().repeat_interleave(32, -1)
            codes = torch.clamp(torch.round((tensors[n].float().to(sx.device) - bx) / sx), 0, 15)
            out[self.pre + n + ".weight"] = mlx_pack(codes).cpu().contiguous()
            out[self.pre + n + ".scales"] = s.cpu().contiguous()
            out[self.pre + n + ".biases"] = b.cpu().contiguous()
        for n in NORMS:
            if n in tensors:
                out[self.pre + n] = (tensors[n].float() + 1.0).to(torch.bfloat16).cpu().contiguous()
        save_file(out, str(path), metadata=meta)


class EXL3(Source):
    kind = "exl3"

    def __init__(self, model_dir: Path, device="cuda") -> None:
        super().__init__(model_dir, device)
        e = self.h.get("model.language_model.embed_tokens.weight")
        self.emb = e.to(torch.bfloat16).to(device)
        means = [float(self.h.get(f"model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight").float().mean())
                 for i in range(0, 48, 6) if self.h.has(f"model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight")]
        self.offset = 1.0 if np.median(means) < 0.5 else 0.0      # exl3.centred_offset: gamma - 1 vs gamma

    @property
    def trainable(self) -> list[str]:
        return [n for n in LINEARS if n not in SHARED]              # the shared expert sits in the expert table

    def _x3(self, prefix: str) -> torch.Tensor:
        from tensorfold.cuda.exl3 import format as fmt

        from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

        if not hasattr(self, "pk"):
            self.pk = Pack(self.dir)
        pk = self.pk
        tr = pk.get(prefix + ".trellis")
        w = fmt.dequantize(tr, pk.scales(prefix, "su", "suh"), pk.scales(prefix, "sv", "svh"), fmt.bits_of(tr.shape),
                           pk.codebook(prefix))
        return torch.from_numpy(np.ascontiguousarray(w.T)).float()           # [K, N] -> [out, in]

    def linear(self, name: str) -> torch.Tensor:
        if self.h.has(self.pre + name + ".trellis"):
            return self._x3(self.pre + name).to(self.device)
        return self.t(name + ".weight").float()

    def norm(self, name: str) -> torch.Tensor:
        return self.t(name).float() + self.offset - 1.0

    def experts(self):
        eg = torch.empty((E, 640, D), dtype=torch.bfloat16, device=self.device)
        eu, ed = torch.empty_like(eg), torch.empty((E, D, 640), dtype=torch.bfloat16, device=self.device)
        for e in range(E):
            for dst, proj in ((eg, "gate_proj"), (eu, "up_proj"), (ed, "down_proj")):
                dst[e] = self._x3(f"{self.pre}layers.0.mlp.experts.{e}.{proj}").to(torch.bfloat16).to(self.device)
        return eg, eu, ed

    def draft_rows(self, ids: torch.Tensor) -> torch.Tensor:
        full = self._x3("lm_head")                                 # [V, K] fp32 (the head kernel's fp32 rows)
        return requant4(full.index_select(0, ids.cpu()).to(self.device), round_first=True).to(torch.bfloat16)

    def export(self, tensors: dict, path: Path, meta: dict) -> None:
        from safetensors.torch import save_file

        out = {self.pre + n + ".weight": tensors[n].to(torch.bfloat16).cpu().contiguous()
               for n in self.trainable if n in tensors}
        out.update({self.pre + n: tensors[n].to(torch.bfloat16).cpu().contiguous() for n in NORMS if n in tensors})
        save_file(out, str(path), metadata=meta)


def source(model_dir: Path, device="cuda") -> Source:
    return {"modelopt": ModelOpt, "mlx4": MLX4, "exl3": EXL3}[detect(model_dir)](model_dir, device)
