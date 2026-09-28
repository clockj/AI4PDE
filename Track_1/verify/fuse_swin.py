"""Gate: `_fno.fuse` on SwinUNet members -- StackedSwin exactness, fp16, int8, packages, timing.

    python Track_1/verify/fuse_swin.py [--json OUT] [--rounds 9]     # GPU: every section
    python Track_1/verify/fuse_swin.py --quick                        # CPU dry run, few windows

WHY THIS EXISTS. The backbone campaign's P1 package is Swin x4 + ViT x2 in two invocations. The
board charges an invocation (+1.151 board ms on Track 1, +0.866 on Track 2, round 23), so four
Swin members are only competitive FUSED, the way StackedFNO and StackedViT made FNO and ViT members
free. A fold is only worth having if it is exact, and a Swin member only ships if the archive's
vendored class is the training definition.

SECTIONS (on CPU, 4 and the CUDA parts of 6 and 7 are skipped):
  1. vendored SwinUNet == models_alt.SwinUNet, BITWISE, on the real checkpoints: fp32 and fp16,
     as-trained and prepare_inference(). Every archive builds the vendored class.
  2. synthetic members of every Phase-2 variant (K = 3): fused vs loop; grouping -- swin_unet_win16
     has swin_unet's exact state_dict shapes and must not share a stack with it.
  3. real members, fp32: raw outputs, and predict_fno's physical field and sigma at alpha 0 and
     0.60, K = 2 and K = 4. THE GATE: <= 2e-5 physical (fuse_vit.py's tolerance; why at TOL32).
  4. fp16 autocast: fused fp16 vs fused fp32 (what both tracks' AMP self-checks compare, tripping
     at 2e-2), against the prepared and as-trained member loops, and both probes' own metrics.
  5. int8 packing (Linear "tensor" / "row" / "fp16") through pack_state_dict -> load_any -> fuse:
     deviation from the fp32 loop, and rel_l2 / tke / mvpe on val_unseen. DECIDED 09-14: Swin
     packages build with RPDE_LINEAR=fp16 (job 7286905, 336 windows, Swin x2: tke +0.00049 for
     "tensor", +0.00012 "row", +0.00003 "fp16"; 6.8 MB a member is no size constraint).
  6. packages in temporary directories, never staged (Swin x2 + ViT x2, int8 tensor, R24's band):
     Track 1 predict() under fp32 / fp16 / fp16 + GRAPH -- AMP_ACTIVE, the probe, GRAPH_ACTIVE,
     permutation invariance -- and Track 2's FrozenEnsemble -- graph captured, replay vs eager on
     real windows.
  7. interleaved one-process timing, b64 (ms per window) and b1 CUDA-graph replay (ms per step),
     fp32 and fp16: Swin x1 prepared, Swin x2 looped (prepared), Swin x2 fused, Swin x4 fused,
     FNO x4 fused (the shipped group, SPEC bmm), ViT x2 fused.

TF32 is disabled throughout, as in both submissions: attention and the stacked matmuls are
GEMMs, and TF32 turns a 1e-7 agreement into 1e-3 without either path being wrong.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.dirname(T1)
sys.path.insert(0, T1)
sys.path.insert(0, os.path.join(T1, "unet"))
sys.path.insert(0, os.path.join(T1, "submit_fno"))

import _fno  # noqa: E402
import models_alt  # noqa: E402

SCREEN = "/fs/scratch/PAS1187/qwu/tmp/realpde/screen"
SWIN = [f"{SCREEN}/w1/w1_swin_unet_s1.pth", f"{SCREEN}/w2/w2_swin_unet_w64_s2.pth"]
FNO4 = [os.path.join(T1, "unet", f"fno_all_w96L8cmb3sim_s{i}.pth") for i in (1, 2, 3, 4)]
VIT2 = [os.path.join(T1, "unet", n) for n in ("arch_vit_all_s1.pth", "arch_vit_all_cmb3sim_s3.pth")]
BAND = os.path.join(T1, "bands", "r24_xfam6.npz")
SUB1 = os.path.join(T1, "submit_fno", "submission.py")
SUB2 = os.path.join(REPO, "Track_2", "submit", "submission.py")
# Physical, fused vs loop, fp32 -- the gate. 2e-5, fuse_vit.py's tolerance (decided 09-14). The stack is
# bitwise equal to the loop on CPU, but on GPU its Linears are batched cuBLAS GEMMs whose rounding
# departs from each member's own addmm/mm by 3e-6..1e-5 per stage (section 3b), and compose's division
# by the temporal std amplifies that ~6x at alpha 0.60: job 7286905 read 2.4e-7 at alpha 0 and 1.55e-6
# at alpha 0.60, over a 1e-6 target that assumed FNO/ViT-like 1e-8 agreement.
TOL32 = 2e-5
TOL_SYN = 1e-5      # normalized raw outputs of random-weight members
AMP_TRIP = 2e-2     # both tracks' AMP self-check threshold

OK: list = []
REC: dict = {}


# ------------------------------------------------------------------------------ reporting
def rep(key, label, v, tol=None, fmt=".3e"):
    """Record and print one number; with `tol` it is a gate."""
    v = float(v)
    REC[key] = v
    if tol is None:
        print(f"  {label:<74s} {v:{fmt}}", flush=True)
        return
    good = v <= tol
    OK.append(good)
    print(f"  {label:<74s} {v:{fmt}}  {'OK' if good else 'FAIL'} (<= {tol:g})", flush=True)


def check(key, label, cond):
    REC[key] = bool(cond)
    OK.append(bool(cond))
    print(f"  {label:<74s} {'OK' if cond else 'FAIL'}", flush=True)


def guarded(name, fn, *a):
    """A section that raises is a FAIL, not the end of the gate."""
    try:
        return fn(*a)
    except Exception:
        traceback.print_exc()
        check(f"{name}.raised", f"{name} raised (traceback above)", False)
        return None


def mx(a, b):
    return float((a.float() - b.float()).abs().max())


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


# ------------------------------------------------------------------------------ model helpers
def amp_ctx(on):
    # cache_enabled=False: required for autocast inside a CUDA-graph capture, harmless eager
    return (torch.autocast("cuda", dtype=torch.float16, cache_enabled=False) if on
            else contextlib.nullcontext())


@torch.no_grad()
def fwd(m, x, amp=False):
    with amp_ctx(amp):
        f, ls = m(x)
    return f.float(), ls.float()


@torch.no_grad()
def loop(ms, x, amp=False):
    """The member loop reduced as predict_fno reduces it: field mean, log of the mean sigma."""
    outs = [fwd(m, x, amp) for m in ms]
    f = torch.stack([o[0] for o in outs]).mean(0)
    s = torch.stack([o[1].clamp(-8.0, 2.0).exp() for o in outs]).mean(0)
    return f, s.log()


def swin(paths, impl, dev, prepare=False):
    """Swin members through load_any with SWIN_IMPL forced to `impl`."""
    old = _fno.SWIN_IMPL
    _fno.SWIN_IMPL = impl
    try:
        ms = [_fno.load_any(p, dev) for p in paths]
    finally:
        _fno.SWIN_IMPL = old
    if prepare:
        for m in ms:
            m.prepare_inference()
    return ms


def prev_of(x_raw, dev):
    x = torch.from_numpy(np.ascontiguousarray(x_raw[..., :2])).to(dev).float()
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    return ((x - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(x), 40, 32, 64).contiguous()


def predict(models, X, dev, alpha=0.60, amp=None, batch=64):
    P = np.zeros(X.shape, np.float32)
    S = np.zeros((len(X), 1, 32, 64, 2), np.float32)
    _fno.predict_fno(models, X, dev, alpha=alpha, batch=batch, out=P, sigma_out=S, amp=amp)
    return P, S


def perturbed(arch, seed, dev, width=32, nl=2):
    """A random-weight training-definition member: the strict case, since trained members agree
    with each other and would hide a swapped member axis."""
    torch.manual_seed(seed)
    m = models_alt.build(arch, width, True, 0, 12, 16, nl, dropout=0.3).eval()
    with torch.no_grad():
        for q in m.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    return m.to(dev)


# ------------------------------------------------------------------------------ 1
def section1(dev, X):
    print("\n1. vendored SwinUNet == models_alt.SwinUNet on the real checkpoints (bitwise)")
    cuda = dev == "cuda"
    x = prev_of(X[:32], dev)
    for p in SWIN:
        tag = os.path.basename(p)[:-4]
        a = swin([p], "models_alt", dev)[0]
        v = swin([p], "vendored", dev)[0]
        check(f"s1.{tag}.classes", f"{tag}: models_alt.SwinUNet / _fno.SwinUNet built",
              type(a) is models_alt.SwinUNet and type(v) is _fno.SwinUNet)
        for prep in (False, True):
            A = copy.deepcopy(a).prepare_inference() if prep else a
            V = copy.deepcopy(v).prepare_inference() if prep else v
            for amp in ((False, True) if cuda else (False,)):
                name = f"{'prepared' if prep else 'as-trained'} {'fp16' if amp else 'fp32'}"
                fa, la = fwd(A, x, amp)
                fv, lv = fwd(V, x, amp)
                rep(f"s1.{tag}.{name.replace(' ', '_')}", f"{tag}: {name}, max|d| field and logstd",
                    max(mx(fa, fv), mx(la, lv)), 0.0)
        if cuda:
            A = copy.deepcopy(a).prepare_inference(fp32_max_batch=1)
            V = copy.deepcopy(v).prepare_inference(fp32_max_batch=1)
            fa, la = fwd(A, x[:1], True)
            fv, lv = fwd(V, x[:1], True)
            rep(f"s1.{tag}.fp32_max_batch", f"{tag}: fp32_max_batch=1 at b1 under autocast",
                max(mx(fa, fv), mx(la, lv)), 0.0)


# ------------------------------------------------------------------------------ 2
def section2(dev):
    print("\n2. synthetic members, every Phase-2 variant (K = 3): fused vs loop, and grouping")
    torch.manual_seed(123)
    x = torch.randn(4, 40, 32, 64, device=dev)
    for arch in _fno.SWIN_VARIANTS:
        ms = [perturbed(arch, s, dev) for s in (1, 2, 3)]
        fu = _fno.fuse(ms)
        check(f"s2.{arch}.fused", f"{arch}: 3 members -> one StackedSwin",
              len(fu) == 1 and type(fu[0]).__name__ == "StackedSwin" and fu[0].n_members == 3)
        f0, l0 = loop(ms, x)
        f1, l1 = fwd(fu[0], x)
        rep(f"s2.{arch}.field", f"{arch}: field  max|loop - fused| (normalized, rms "
            f"{float(f0.pow(2).mean().sqrt()):.2f})", mx(f0, f1), TOL_SYN)
        rep(f"s2.{arch}.logstd", f"{arch}: logstd max|loop - fused|", mx(l0, l1), TOL_SYN)
    s = [perturbed("swin_unet", i, dev) for i in (1, 2)]
    w = [perturbed("swin_unet_win16", i, dev) for i in (1, 2)]
    g = _fno.fuse([s[0], w[0]])
    check("s2.group.win16_apart", "swin_unet + swin_unet_win16 (equal shapes) -> two members",
          len(g) == 2 and g[0] is s[0] and g[1] is w[0])
    g = _fno.fuse([s[0], w[0], s[1], w[1]])
    check("s2.group.two_stacks", "2 swin_unet + 2 swin_unet_win16 -> two StackedSwin of 2",
          [type(o).__name__ for o in g] == ["StackedSwin"] * 2 and [o.n_members for o in g] == [2, 2])
    g = _fno.fuse([s[0]])
    check("s2.group.single", "a lone Swin member stays unfused", len(g) == 1 and g[0] is s[0])


# ------------------------------------------------------------------------------ 3
def section3(dev, X):
    print(f"\n3. real members, fp32: StackedSwin vs the member loop ({len(X)} val_unseen windows)")
    x = prev_of(X[:64], dev)
    ms = swin(SWIN, "vendored", dev)
    fu = _fno.fuse(swin(SWIN, "vendored", dev))
    check("s3.k2.fused", "two real members -> one StackedSwin",
          len(fu) == 1 and type(fu[0]).__name__ == "StackedSwin")
    f0, l0 = loop(ms, x)
    f1, l1 = fwd(fu[0], x)
    rep("s3.k2.raw_field", "K=2 raw field  max|loop - fused| (normalized)", mx(f0, f1))
    rep("s3.k2.raw_logstd", "K=2 raw logstd max|loop - fused|", mx(l0, l1))
    for alpha in (0.0, 0.60):
        Pa, Sa = predict(ms, X, dev, alpha)
        Pb, Sb = predict(fu, X, dev, alpha)
        rep(f"s3.k2.a{alpha:.2f}.field", f"K=2 predict_fno alpha {alpha:.2f}: field max|loop - fused| "
            "(physical)", np.abs(Pa - Pb).max(), TOL32)
        rep(f"s3.k2.a{alpha:.2f}.sigma", f"K=2 predict_fno alpha {alpha:.2f}: sigma max|loop - fused|",
            np.abs(Sa - Sb).max(), TOL32)
    ms4 = swin(SWIN + SWIN, "vendored", dev)
    fu4 = _fno.fuse(swin(SWIN + SWIN, "vendored", dev))
    check("s3.k4.fused", "four members (the two, twice, as distinct modules) -> one StackedSwin",
          len(fu4) == 1 and getattr(fu4[0], "n_members", 0) == 4)
    Pa, Sa = predict(ms4, X, dev)
    Pb, Sb = predict(fu4, X, dev)
    rep("s3.k4.field", "K=4 predict_fno alpha 0.60: field max|loop - fused| (physical)",
        np.abs(Pa - Pb).max(), TOL32)
    rep("s3.k4.sigma", "K=4 predict_fno alpha 0.60: sigma max|loop - fused|", np.abs(Sa - Sb).max(), TOL32)
    return ms, fu


@torch.no_grad()
def section3b(dev, X, ms, fu):
    """LOCAL error per stage: each stacked stage fed the member loop's own inputs, so a departure
    is located where it is made instead of read off the output after it has been amplified."""
    print("\n3b. where the fused path departs from the loop: LOCAL max|d| per stage kind, both fed the "
          "loop's own inputs (K=2, fp32)")
    F = torch.nn.functional
    st, K = fu[0], len(ms)
    prev = prev_of(X[:16], dev)
    B = len(prev)
    tr = []
    for m in ms:
        o = {"in": torch.cat([prev, m.grid.expand(B, -1, -1, -1)], 1)}
        o["embed"] = m.embed(o["in"]).permute(0, 2, 3, 1)
        t = o["embed_norm"] = m.embed_norm(o["embed"])
        skips = []
        for i, (blocks, merge, sblk) in enumerate(zip(m.enc, m.merge, m.skip_blocks)):
            for j, blk in enumerate(blocks):
                o[f"enc{i}.{j}.in"] = t
                t = o[f"enc{i}.{j}"] = blk(t)
            o[f"skip{i}.in"] = t
            skips.append(sblk(t.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
            o[f"skip{i}"] = skips[-1]
            o[f"merge{i}.in"] = t
            t = o[f"merge{i}"] = merge(t)
        for j, blk in enumerate(m.bottleneck):
            o[f"bott{j}.in"] = t
            t = o[f"bott{j}"] = blk(t)
        for j, (ex, sk, blocks, s) in enumerate(zip(m.expand, m.skip, m.dec, reversed(skips))):
            o[f"expand{j}.in"] = t
            o[f"expand{j}"] = ex(t)
            o[f"skiplin{j}.in"] = torch.cat([o[f"expand{j}"], s], -1)
            t = o[f"skiplin{j}"] = sk(o[f"skiplin{j}.in"])
            for q, blk in enumerate(blocks):
                o[f"dec{j}.{q}.in"] = t
                t = o[f"dec{j}.{q}"] = blk(t)
        o["final.in"] = t
        o["final"] = m.final(t)
        hc = o["final"].permute(0, 3, 1, 2)
        r = hc + m.refine(hc)
        o["refine"] = r.permute(0, 2, 3, 1)
        o["field"] = m.head_field(r).permute(0, 2, 3, 1)
        o["logstd"] = m.head_logstd(r).permute(0, 2, 3, 1)
        tr.append(o)
    worst = {}

    def note(kind, fused_tok, key):
        v = max(mx(fused_tok[k], tr[k][key]) for k in range(K))
        worst[kind] = max(worst.get(kind, 0.0), v)

    def inp(key):
        return torch.stack([o[key] for o in tr])

    e = st.embed(tr[0]["in"])
    note("embed conv", e.view(B, K, -1, *e.shape[-2:]).permute(1, 0, 3, 4, 2), "embed")
    note("LayerNorm split affine", st.embed_norm(inp("embed")), "embed_norm")
    for i, blocks in enumerate(st.enc):
        for j, blk in enumerate(blocks):
            note("Swin blocks", blk(inp(f"enc{i}.{j}.in")), f"enc{i}.{j}")
        z = _fno._tok_to_cl(inp(f"skip{i}.in"))
        for cb in st.skip_blocks[i]:
            z = cb(z)
        note("ConvNeXt skips", _fno._cl_to_tok(z, K), f"skip{i}")
        note("patch merges", st.merge[i](inp(f"merge{i}.in")), f"merge{i}")
    for j, blk in enumerate(st.bottleneck):
        note("Swin blocks", blk(inp(f"bott{j}.in")), f"bott{j}")
    for j in range(len(st.dec)):
        note("patch expands", st.expand[j](inp(f"expand{j}.in")), f"expand{j}")
        note("skip Linears", st.skip[j](inp(f"skiplin{j}.in")), f"skiplin{j}")
        for q, blk in enumerate(st.dec[j]):
            note("Swin blocks", blk(inp(f"dec{j}.{q}.in")), f"dec{j}.{q}")
    note("final expand", st.final(inp("final.in")), "final")
    z = _fno._tok_to_cl(inp("final"))
    note("refine conv", _fno._cl_to_tok(z + st.refine(F.gelu(z)), K), "refine")
    r = inp("refine")                                   # tokens: the heads are batched matmuls
    note("heads", st.hf[1](F.gelu(st.hf[0](r))), "field")
    note("heads", st.hl[1](F.gelu(st.hl[0](r))), "logstd")
    for kind, v in worst.items():
        rep(f"s3b.{kind.replace(' ', '_')}", f"local, worst over the {kind}", v)


# ------------------------------------------------------------------------------ 4
def t1_probe(models, dev):
    """Track 1's `_amp_verified` metric, verbatim: predict_fno fp16 vs fp32 on 8 random windows."""
    g = np.random.default_rng(0)
    probe = np.zeros((8, 20, 32, 64, 3), dtype=np.float32)
    probe[..., :2] = (_fno.MEAN + _fno.STD * g.standard_normal((8, 20, 32, 64, 2))).astype(np.float32)
    s0 = np.empty((8, 1, 32, 64, 2), dtype=np.float32)
    s1 = np.empty_like(s0)
    p0 = _fno.predict_fno(models, probe, dev, alpha=0.60, batch=8, sigma_out=s0, amp=None)
    p1 = _fno.predict_fno(models, probe, dev, alpha=0.60, batch=8, sigma_out=s1, amp=torch.float16)
    return max(float(np.linalg.norm(p1 - p0) / max(float(np.linalg.norm(p0)), 1e-12)),
               float(np.linalg.norm(s1 - s0) / max(float(np.linalg.norm(s0)), 1e-12)))


@torch.no_grad()
def t2_probe(models, dev):
    """Track 2's `_amp_verified` metric, verbatim: per group, raw outputs on one randn window."""
    g = torch.Generator(device="cpu").manual_seed(0)
    pv = torch.randn((1, 40, 32, 64), generator=g).to(dev)
    worst = 0.0
    for mdl in models:
        r0, l0 = mdl(pv)
        with torch.autocast("cuda", dtype=torch.float16):
            r1, l1 = mdl(pv)
        worst = max(worst, rel(r1, r0))
        if l1 is not None:
            worst = max(worst, rel(l1, l0))
    return worst


def section4(dev, X, ms, fu):
    print("\n4. fp16 autocast -- both tracks' AMP self-checks fall back to fp32 above 2e-2")
    x = prev_of(X[:64], dev)
    F32, L32 = fwd(fu[0], x, False)
    F16, L16 = fwd(fu[0], x, True)
    rep("s4.fused16_vs_fused32.field_max", "fused fp16 vs fused fp32: field max (normalized)", mx(F16, F32))
    rep("s4.fused16_vs_fused32.field_rel", "fused fp16 vs fused fp32: field rel-L2", rel(F16, F32))
    rep("s4.fused16_vs_fused32.logstd_max", "fused fp16 vs fused fp32: logstd max", mx(L16, L32))
    fp, lp = loop([copy.deepcopy(m).prepare_inference() for m in ms], x, True)
    rep("s4.fused16_vs_loopprep16.field_max", "fused fp16 vs PREPARED loop fp16: field max", mx(F16, fp))
    rep("s4.fused16_vs_loopprep16.logstd_max", "fused fp16 vs PREPARED loop fp16: logstd max", mx(L16, lp))
    fa, la = loop(ms, x, True)
    rep("s4.fused16_vs_loopraw16.field_max", "fused fp16 vs AS-TRAINED loop fp16: field max", mx(F16, fa))
    rep("s4.loopraw16_vs_loop32.field_max", "context: as-trained loop fp16 vs fp32: field max",
        mx(fa, loop(ms, x, False)[0]))
    f32b, _ = fwd(fu[0], x[:1], False)
    f16b, _ = fwd(fu[0], x[:1], True)
    rep("s4.b1.field_max", "b1 (Track 2's batch): fused fp16 vs fp32 field max", mx(f16b, f32b))
    Pa, Sa = predict(fu, X, dev, 0.60, None)
    Pb, Sb = predict(fu, X, dev, 0.60, torch.float16)
    rep("s4.phys.field_max", "predict_fno fused fp16 vs fp32: field max (physical)", np.abs(Pb - Pa).max())
    rep("s4.phys.sigma_max", "predict_fno fused fp16 vs fp32: sigma max (physical)", np.abs(Sb - Sa).max())
    vit = _fno.fuse([_fno.load_any(p, dev) for p in VIT2])
    for label, models in (("Swin x2 fused", fu), ("Swin x2 + ViT x2 fused", fu + vit)):
        key = label.split(" fused")[0].replace(" ", "").replace("+", "_")
        rep(f"s4.probe_t1.{key}", f"Track 1 AMP probe metric, {label}", t1_probe(models, dev), AMP_TRIP)
        rep(f"s4.probe_t2.{key}", f"Track 2 AMP probe metric, {label}", t2_probe(models, dev), AMP_TRIP)


# ------------------------------------------------------------------------------ 5
def section5(dev, X, Y, tmp, ms):
    print(f"\n5. int8 packing: pack_state_dict -> load_any -> fuse, against the fp32 loop "
          f"({len(X)} val_unseen windows)")
    from sps9 import sc

    def metrics(P):
        return (float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))), float(sc.mvpe_rel_l2(P, Y)))

    P0, S0 = predict(ms, X, dev)
    m0 = metrics(P0)
    REC["s5.fp32_loop.metrics"] = m0
    print(f"  fp32 member loop: rel_l2 {m0[0]:.5f}  tke {m0[1]:.5f}  mvpe {m0[2]:.5f}")
    for linear in ("tensor", "row", "fp16"):
        d = os.path.join(tmp, f"int8_{linear}")
        os.makedirs(d, exist_ok=True)
        paths = []
        for i, p in enumerate(SWIN):
            blob = torch.load(p, map_location="cpu")
            blob["model_state_dict"] = _fno.pack_state_dict(blob["model_state_dict"], fp16=True,
                                                            int8=True, linear=linear)
            q = os.path.join(d, f"m{i}.pth")
            torch.save(blob, q)
            paths.append(q)
        fq = _fno.fuse(swin(paths, "vendored", dev))
        P, S = predict(fq, X, dev)
        mq = metrics(P)
        REC[f"s5.{linear}.metrics"] = mq
        rep(f"s5.{linear}.field_rel", f"linear={linear}: ||dP|| / ||P|| (quant8.py's metric)",
            np.linalg.norm(P - P0) / np.linalg.norm(P0))
        rep(f"s5.{linear}.field_max", f"linear={linear}: field max|d| (physical)", np.abs(P - P0).max())
        rep(f"s5.{linear}.sigma_max", f"linear={linear}: sigma max|d|", np.abs(S - S0).max())
        rep(f"s5.{linear}.mb", f"linear={linear}: MB per member", sum(map(os.path.getsize, paths)) / 2e6,
            fmt=".2f")
        print(f"    rel_l2 {mq[0]:.5f} ({mq[0] - m0[0]:+.5f})  tke {mq[1]:.5f} ({mq[1] - m0[1]:+.5f})"
              f"  mvpe {mq[2]:.5f} ({mq[2] - m0[2]:+.5f})", flush=True)


# ------------------------------------------------------------------------------ 6
def make_pkg(tmp, track, members, consts):
    """A package directory laid out as build_zip.py / build_t2.py lay one out."""
    d = tempfile.mkdtemp(prefix=f"t{track}pkg_", dir=tmp)
    txt = open(SUB1 if track == 1 else SUB2).read()
    for k, v in consts.items():
        txt, n = re.subn(rf"(?m)^{k} = .*$", f"{k} = {v}", txt)
        if n != 1:
            raise RuntimeError(f"track {track}: expected one `{k} = ` line, found {n}")
    open(os.path.join(d, "submission.py"), "w").write(txt)
    shutil.copy2(os.path.join(T1, "submit_fno", "_fno.py"), os.path.join(d, "_fno.py"))
    shutil.copy2(BAND, os.path.join(d, "band.npz"))
    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(members) + 1)]
    for src, name in zip(members, names):
        blob = torch.load(src, map_location="cpu")
        blob["model_state_dict"] = _fno.pack_state_dict(blob["model_state_dict"], fp16=True, int8=True,
                                                        linear="tensor")
        torch.save(blob, os.path.join(d, name))
    return d


def load_pkg(d, name):
    """Import a package's submission.py with ITS OWN _fno.py, as the platform would."""
    saved = sys.modules.pop("_fno")
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(d, "submission.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules["_fno"] = saved
    assert os.path.dirname(os.path.abspath(mod._fno.__file__)) == os.path.abspath(d), "wrong _fno"
    return mod


def section6(dev, Xall, tmp):
    cuda = dev == "cuda"
    members = SWIN + VIT2
    Xp = Xall[:150] if cuda else Xall[:6]
    print(f"\n6. packages in temporary directories: Swin x2 + ViT x2, int8 tensor, band r24_xfam6 "
          f"({len(Xp)} windows; never staged)")
    t1_variants = [("fp32", {"AMP": "None", "GRAPH": "False"})]
    if cuda:
        t1_variants += [("fp16", {"AMP": '"fp16"', "GRAPH": "False"}),
                        ("fp16_graph", {"AMP": '"fp16"', "GRAPH": "True"})]
    for tag, consts in t1_variants:
        d = make_pkg(tmp, 1, members, dict(consts, FUSE="True", ALPHA="0.60"))
        sub = load_pkg(d, f"t1pkg_{tag}")
        st = sub._state
        check(f"s6.t1.{tag}.loaded", f"T1 {tag}: members loaded at import", st is not None)
        if st is None:
            continue
        kinds = [type(o).__name__ for o in st[0]]
        check(f"s6.t1.{tag}.groups", f"T1 {tag}: invocations {kinds}", kinds == ["StackedSwin", "StackedViT"])
        check(f"s6.t1.{tag}.vendored", f"T1 {tag}: the archive's _fno resolves Swin to its vendored class",
              sub._fno.SWIN_IMPL == "auto"
              and not os.path.exists(os.path.join(sub._fno._UNET_DIR, "models_alt.py")))
        REC[f"s6.t1.{tag}.state"] = dict(AMP_ACTIVE=sub.AMP_ACTIVE, AMP_PROBE_DIFF=sub.AMP_PROBE_DIFF,
                                         GRAPH_ACTIVE=getattr(sub, "GRAPH_ACTIVE", None),
                                         GRAPH_PROBE_DIFF=getattr(sub, "GRAPH_PROBE_DIFF", None), batch=st[5])
        print(f"    {REC[f's6.t1.{tag}.state']}", flush=True)
        if consts["AMP"] != "None":
            check(f"s6.t1.{tag}.amp", f"T1 {tag}: AMP self-check kept fp16 (probe {sub.AMP_PROBE_DIFF})",
                  sub.AMP_ACTIVE == "float16")
        if consts["GRAPH"] == "True":
            check(f"s6.t1.{tag}.graph", f"T1 {tag}: CUDA graph captured, verified exact against eager",
                  sub.GRAPH_ACTIVE is True and sub.GRAPH_PROBE_DIFF == 0.0)
        out = {k: np.array(v, copy=True) for k, v in sub.predict(Xp, None).items()}
        perm = np.random.default_rng(0).permutation(len(Xp))
        outp = {k: np.array(v, copy=True) for k, v in sub.predict(Xp[perm], None).items()}
        P = out["prediction"][perm]
        rep(f"s6.t1.{tag}.perm_drift", f"T1 {tag}: permutation drift ||d|| / ||P|| (validate_strict 1e-4)",
            np.linalg.norm(P - outp["prediction"]) / np.linalg.norm(P), 1e-4)
        rep(f"s6.t1.{tag}.perm_max", f"T1 {tag}: permutation max|d| over prediction, lower, upper",
            max(float(np.abs(out[k][perm] - outp[k]).max()) for k in ("prediction", "lower", "upper")))
        check(f"s6.t1.{tag}.bounds", f"T1 {tag}: finite, lower <= upper, p zero",
              all(np.isfinite(out[k]).all() for k in out) and bool((out["lower"] <= out["upper"]).all())
              and not out["prediction"][..., 2].any())
        del sub, st, out, outp
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    mean_in = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
    std_in = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
    xin = torch.from_numpy((Xp[:8] - mean_in) / std_in).float()
    for tag, consts in [("fp32", {"AMP": "None"})] + ([("fp16", {"AMP": '"fp16"'})] if cuda else []):
        d = make_pkg(tmp, 2, members, dict(consts, FUSE="True", ALPHA="0.60"))
        sub = load_pkg(d, f"t2pkg_{tag}")
        m = sub.get_ttt_model(d, dev)
        kinds = [type(o).__name__ for o in m.models]
        check(f"s6.t2.{tag}.groups", f"T2 {tag}: invocations {kinds}", kinds == ["StackedSwin", "StackedViT"])
        REC[f"s6.t2.{tag}.state"] = dict(AMP_ACTIVE=sub.AMP_ACTIVE, AMP_PROBE_DIFF=sub.AMP_PROBE_DIFF,
                                         graph=m.graph is not None)
        print(f"    {REC[f's6.t2.{tag}.state']}", flush=True)
        if consts["AMP"] != "None":
            check(f"s6.t2.{tag}.amp", f"T2 {tag}: AMP self-check kept fp16 (probe {sub.AMP_PROBE_DIFF})",
                  sub.AMP_ACTIVE == "float16")
        if cuda:
            check(f"s6.t2.{tag}.graph", f"T2 {tag}: CUDA graph captured (load-time check demands 0.0)",
                  m.graph is not None)
        if m.graph is not None:
            worst = 0.0
            with torch.no_grad():
                for i in range(len(xin)):
                    m.gx.copy_(xin[i:i + 1])
                    m._body()
                    ref = [t.clone() for t in (m.go, m.gl, m.gh)]
                    m.gx.copy_(xin[i:i + 1])
                    m.graph.replay()
                    torch.cuda.synchronize()
                    worst = max([worst] + [mx(a, b) for a, b in zip((m.go, m.gl, m.gh), ref)])
            rep(f"s6.t2.{tag}.replay", f"T2 {tag}: graph replay vs eager, 8 real windows (pred, lower, upper)",
                worst, 0.0)
        o, info = m.ttt_step(xin[:1], None)
        check(f"s6.t2.{tag}.contract", f"T2 {tag}: ttt_step returns finite prediction and bounds",
              bool(torch.isfinite(o).all()) and "lower" in info and bool((info["lower"] <= info["upper"]).all()))
        del sub, m
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------ 7
def section7(dev, X, rounds):
    cuda = dev == "cuda"
    print("\n7. interleaved timing in one process (median over rounds after the first)")
    _fno._SPEC = "bmm"                       # the shipped SPEC_PATH, as screen_latency.py sets it
    arms = {
        "fno4_fused": _fno.fuse([_fno.load_any(p, dev) for p in FNO4]),
        "vit2_fused": _fno.fuse([_fno.load_any(p, dev) for p in VIT2]),
        "swin1_prep": swin(SWIN[:1], "vendored", dev, prepare=True),
        "swin2_loop": swin(SWIN, "vendored", dev, prepare=True),
        "swin2_fused": _fno.fuse(swin(SWIN, "vendored", dev)),
        "swin4_fused": _fno.fuse(swin(SWIN + SWIN, "vendored", dev)),
    }
    print("  arms: " + "; ".join(f"{k} = {[type(m).__name__ for m in v]}" for k, v in arms.items()))
    x64 = prev_of(X[:64] if cuda else X[:2], dev)
    nb = len(x64)

    @torch.no_grad()
    def step(ms, x, amp):
        with amp_ctx(amp):
            return [m(x) for m in ms]

    settings = [("b64", False), ("b64", True), ("b1g", False), ("b1g", True)] if cuda else [("b64", False)]
    runs, niter, keep = {}, {}, []
    for mode, amp in settings:
        for name, ms in arms.items():
            if mode == "b1g":
                sx = x64[:1].clone()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    for _ in range(3):
                        step(ms, sx, amp)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = step(ms, sx, amp)
                keep.append((g, sx, out))
                run = g.replay
            else:
                run = (lambda ms=ms, amp=amp: step(ms, x64, amp))
            for _ in range(3):
                run()
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            run()
            if cuda:
                torch.cuda.synchronize()
            niter[(name, mode, amp)] = int(min(2000, max(3, math.ceil(0.08 / max(time.perf_counter() - t, 1e-6)))))
            runs[(name, mode, amp)] = run
    keys = list(runs)
    acc = {k: [] for k in keys}
    for r in range(rounds if cuda else 2):
        for k in keys[r % len(keys):] + keys[:r % len(keys)]:
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(niter[k]):
                runs[k]()
            if cuda:
                torch.cuda.synchronize()
            acc[k].append((time.perf_counter() - t) / niter[k])
    table = {}
    for name in arms:
        row = {}
        for mode, amp in settings:
            v = acc[(name, mode, amp)][1:] or acc[(name, mode, amp)]
            per = nb if mode == "b64" else 1
            row[f"{mode}_{'fp16' if amp else 'fp32'}"] = 1e3 * float(np.median(v)) / per
        table[name] = row
    REC["s7.ms"] = table
    cols = list(next(iter(table.values())))
    print("  ms: b64 per window, b1g per step\n  " + f"{'arm':<13s}" + "".join(f"{c:>11s}" for c in cols)
          + "".join(f"{'x' + c:>13s}" for c in cols))
    ref = table["fno4_fused"]
    for name, row in table.items():
        print("  " + f"{name:<13s}" + "".join(f"{row[c]:11.4f}" for c in cols)
              + "".join(f"{row[c] / ref[c]:13.3f}" for c in cols), flush=True)
    p1 = {c: table["swin4_fused"][c] + table["vit2_fused"][c] for c in cols}
    r25 = {c: table["fno4_fused"][c] + table["vit2_fused"][c] for c in cols}
    REC["s7.p1_vs_r25"] = {c: (p1[c], r25[c]) for c in cols}
    print("  P1 (Swin x4 fused + ViT x2 fused) vs R25 (FNO x4 fused + ViT x2 fused), two invocations each:")
    for c in cols:
        print(f"    {c:<9s} P1 {p1[c]:.4f}  R25 {r25[c]:.4f}  ratio {p1[c] / r25[c]:.3f}  "
              f"delta {p1[c] - r25[c]:+.4f} ms", flush=True)


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--quick", action="store_true", help="CPU dry run on a few windows")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if (torch.cuda.is_available() and not args.quick) else "cpu"
    cuda = dev == "cuda"
    if not cuda:
        torch.set_num_threads(8)
    z = np.load(os.path.join(T1, "val_unseen.npz"))
    Xall, Yall = z["x"].astype(np.float32), z["y"].astype(np.float32)
    n = 128 if cuda else 8
    X = Xall[np.linspace(0, len(Xall) - 1, n).round().astype(int)]
    md5 = hashlib.md5(open(_fno.__file__, "rb").read()).hexdigest()
    REC["env"] = dict(torch=torch.__version__, device=torch.cuda.get_device_name() if cuda else "cpu",
                      fno_md5=md5, swin=SWIN)
    print(f"device {REC['env']['device']}  torch {torch.__version__}  TF32 off  _fno.py {md5}")
    print("Swin members: " + ", ".join(os.path.basename(p) for p in SWIN))
    tmp = tempfile.mkdtemp(prefix="fuse_swin_")
    try:
        guarded("section1", section1, dev, X)
        guarded("section2", section2, dev)
        got = guarded("section3", section3, dev, X)
        if got is not None:
            guarded("section3b", section3b, dev, X, *got)
        if got is not None and cuda:
            guarded("section4", section4, dev, X, *got)
        if got is not None:
            k = len(Xall) if cuda else 8
            guarded("section5", section5, dev, Xall[:k], Yall[:k], tmp, got[0])
        del got
        gc.collect()
        guarded("section6", section6, dev, Xall, tmp)
        guarded("section7", section7, dev, X, args.rounds)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    REC["pass"] = all(OK)
    REC["n_checks"], REC["n_fail"] = len(OK), OK.count(False)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(REC, f, indent=1, default=float)
    print("\nPASS" if all(OK) else f"\nFAIL ({OK.count(False)} of {len(OK)} checks)")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
