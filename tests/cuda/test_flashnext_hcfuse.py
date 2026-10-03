"""Guards for the EXL3 prompt HC read-out (exl3_hc): the fused kernels give the five-kernel read-out's bits on the real
shapes (4 streams x 2560, low rank 320, inject 4), and the format gate leaves upstream's Q4 (MLX) fusion path as is."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import exl3_hc, exl3_mm, hc_readout, qmm

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _faces(n_dn, wd=None, wu=None, dev="cpu"):
    S, D, LOW = 4, 2560, 320
    sc = exl3_mm.Scratch(11)
    wd = torch.zeros((n_dn, S * D), dtype=torch.float16, device=dev) if wd is None else wd
    wu = torch.zeros((S * D, LOW), dtype=torch.float16, device=dev) if wu is None else wu
    down = exl3_mm.F16(wd, n_dn, S * D, exl3_mm.f16_split(n_dn, S * D), sc)
    up = exl3_mm.F16(wu, S * D, LOW, exl3_mm.f16_split(S * D, LOW), sc)
    sc.part = torch.empty((max(down.sk * n_dn, up.sk * S * D) * exl3_mm.ROWS,), dtype=torch.float32, device=dev)
    return down, up


def _case(R, inject, seed=0):
    from tensorfold.families.qwen4_exp.cuda import glue

    g = torch.Generator(device="cuda").manual_seed(seed)
    S, D, LOW = 4, 2560, 320
    dev = "cuda"
    h = (torch.randn((R, S * D), generator=g, device=dev) * 3).to(torch.bfloat16)
    scale = (1 + 0.2 * torch.randn((S * D,), generator=g, device=dev)).float()
    n_dn = LOW + (S if inject else 0)
    wd = (torch.randn((n_dn, S * D), generator=g, device=dev) * 0.02).half()
    wu = (torch.randn((S * D, LOW), generator=g, device=dev) * 0.05).half()
    down, up = _faces(n_dn, wd, wu, dev)
    pss = torch.empty((R, D // 256, S), dtype=torch.float32, device=dev)
    glue.hc_writeback(h, h, pss, S, 0)
    return S, D, LOW, h, scale, down, up, pss, n_dn


def _nan(*s, dt=torch.bfloat16):
    return torch.full(s, float("nan"), dtype=dt, device="cuda")


def _ref(S, D, LOW, h, scale, down, up, pss, n_dn, inject, eps=1e-6):
    """The stock five kernels: glue's norm, F16 down (upstream's tiles and slice sums), act, F16 up, glue's mix."""
    from tensorfold.families.qwen4_exp.cuda import glue

    R = h.shape[0]
    normed, xsn = _nan(R, S * D), _nan(R, S * D // 32, dt=torch.float32)
    glue.hc_normed(h, pss, scale, normed, xsn, S, eps)
    dn = _nan(R, n_dn)
    down(normed, dn)
    act, xsa, inj = _nan(R, LOW), _nan(R, LOW // 32, dt=torch.float32), (_nan(R, S) if inject else None)
    glue.hc_act(dn, act, xsa, inj, S, LOW)
    upo = _nan(R, S * D)
    up(act, upo)
    mixed, xsm = _nan(R, D), _nan(R, D // 32, dt=torch.float32)
    glue.hc_mix(upo, normed, mixed, xsm, S)
    return dn, act, xsa, inj, mixed


def _fused(S, D, LOW, h, scale, down, up, pss, n_dn, inject, eps=1e-6):
    from tensorfold.families.qwen4_exp.cuda import glue

    R = h.shape[0]
    dn = _nan(R, n_dn)
    exl3_hc.hc_down_prompt(h, pss, scale, eps, S, down, dn)
    act, xsa, inj = _nan(R, LOW), _nan(R, LOW // 32, dt=torch.float32), (_nan(R, S) if inject else None)
    glue.hc_act(dn, act, xsa, inj, S, LOW)
    mixed = _nan(R, D)
    exl3_hc.hc_upmix_prompt(act, up, h, pss, scale, eps, S, mixed, None)     # the forward writes no group sums
    return dn, act, xsa, inj, mixed


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


@cuda
@pytest.mark.parametrize("R", [65, 129, 1000, 2048, 4096])
@pytest.mark.parametrize("inject", [True, False])
def test_fused_readout_bits(R, inject):
    c = _case(R, inject, seed=R + inject)
    assert exl3_hc.fusable(c[5], c[6], R)
    for name, a, b in zip(("dn", "act", "xs_act", "inject", "mixed"), _ref(*c, inject), _fused(*c, inject)):
        if a is not None:
            assert torch.equal(_bits(a), _bits(b)), f"{name} differs at R={R}"


@cuda
def test_readout_plain_routes_exl3_prompts_to_the_same_bits(monkeypatch):
    """Through hc_readout._readout_plain: fused (default) and TF_HC_FUSE=0 write the same mixed/act/inject rows."""
    from tensorfold.families.qwen4_exp.cuda import glue

    R, inject = 300, True
    S, D, LOW, h, scale, down, up, pss, n_dn = _case(R, inject, seed=5)
    hc = SimpleNamespace(down=down, up=up, prefill_down=down, prefill_up=up, scale=scale)

    def run():
        b = SimpleNamespace(prefill=True, pss=pss, dn=_nan(R, n_dn), dn_mix=_nan(R, n_dn), act=_nan(R, LOW),
                            xs_act=_nan(R, LOW // 32, dt=torch.float32), normed=_nan(R, S * D),
                            xs_normed=_nan(R, S * D // 32, dt=torch.float32), up=_nan(R, S * D),
                            mixed=_nan(R, D), xs_mixed=_nan(R, D // 32, dt=torch.float32), part=None)
        inj = _nan(R, S)
        hc_readout._readout_plain(hc, b, h, R, 1e-6, S, LOW, inj)
        return b, inj

    fused, inj_f = run()
    monkeypatch.setenv("TF_HC_FUSE", "0")
    stock, inj_s = run()
    assert torch.isnan(fused.normed.float()).all() and not torch.isnan(stock.normed.float()).any()
    for a, b in ((fused.mixed, stock.mixed), (fused.act, stock.act), (inj_f, inj_s)):
        assert torch.equal(_bits(a), _bits(b))


def test_not_fusable_for_decode_rows_or_when_off(monkeypatch):
    down, up = _faces(324)
    assert down.sk > 1 and exl3_hc.fusable(down, up, 65)
    assert not exl3_hc.fusable(down, up, exl3_hc.FUSED_MIN)      # decode windows and short pieces keep five kernels
    monkeypatch.setenv("TF_HC_FUSE", "0")
    assert not exl3_hc.fusable(down, up, 4096)


def test_q4_and_bf16_faces_are_never_fusable():
    q4 = object.__new__(qmm.Q4)
    b16 = SimpleNamespace(kernel="b16")
    down, up = _faces(324)
    for d, u in ((q4, q4), (b16, b16), (q4, up), (down, q4), (None, None)):
        assert not exl3_hc.fusable(d, u, 4096)


def test_q4_prompt_readout_keeps_upstreams_path(monkeypatch):
    """A Q4 (MLX) HC at 512+ prompt rows: hc_readout runs upstream's norm, down and hc_check upmix, never exl3_hc."""

    calls = []
    fail = lambda *a, **k: pytest.fail("exl3_hc ran on a Q4 face")
    monkeypatch.setattr(exl3_hc, "hc_down_prompt", fail)
    monkeypatch.setattr(exl3_hc, "hc_upmix_prompt", fail)
    monkeypatch.setattr(hc_readout.glue, "hc_normed", lambda *a, **k: calls.append("normed"))
    monkeypatch.setattr(hc_readout, "_down_act", lambda *a, **k: calls.append("down_act"))
    monkeypatch.setattr(hc_readout, "_hc_fuser", lambda dev, upmix=False: (lambda *a, **k: calls.append("upmix")))
    monkeypatch.setattr(hc_readout, "_mm", lambda *a, **k: pytest.fail("the Q4 upmix fusion was bypassed"))
    up = object.__new__(qmm.Q4)
    up.n = 10240
    q4 = object.__new__(qmm.Q4)
    hc = SimpleNamespace(down=q4, up=up, prefill_down=q4, prefill_up=up, scale=None)
    t = torch.empty((512, 1))
    b = SimpleNamespace(prefill=True, pss=t, normed=t, xs_normed=t, act=t, xs_act=t, mixed=t, xs_mixed=t)
    hc_readout._readout_plain(hc, b, torch.empty((512, 1)), 512, 1e-6, 4, 320, None)
    assert calls == ["normed", "down_act", "upmix"]


def _rot_case(R, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    D, E, slots = 2560, 512, 11
    x = (torch.randn((R, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
    pick = torch.randint(0, E + 1, (R, slots), generator=g, device="cuda", dtype=torch.int32)   # E: a skipped pick
    suh_g = (torch.randn((E + 1, D), generator=g, device="cuda")).half()
    suh_u = (torch.randn((E + 1, D), generator=g, device="cuda")).half()
    return x, pick, suh_g, suh_u, D, E, slots


def _rot(case, wide4):
    from tensorfold.cuda.exl3.experts import _ext

    x, pick, suh_g, suh_u, D, E, slots = case
    R = x.shape[0]
    xg = torch.full((R * slots, D), float("nan"), dtype=torch.float16, device="cuda")
    xu = xg.clone()
    _ext().rot_in(x, x.stride(0), pick, suh_g, suh_u, xg, xu, R, D, slots, E, wide4)
    return xg, xu


@cuda
@pytest.mark.parametrize("R", [1, 7, 2048, 4096])
def test_rot_in4_bits(R):
    from tensorfold.cuda.exl3 import experts

    if not hasattr(experts, "ROT_IN4"):
        pytest.skip("rot_in4 arrives with the EXL3 prompt-expert stack (9e85c30 on #212)")
    c = _rot_case(R, seed=R)
    for a, b in zip(_rot(c, False), _rot(c, True)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
