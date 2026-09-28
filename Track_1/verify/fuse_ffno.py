"""Gate: `_fno.fuse` on FFNO2d members -- StackedFFNO exactness per spectral path, fp16, int8, packages, timing.

    python Track_1/verify/fuse_ffno.py [--ffno A.pth B.pth] [--json OUT] [--rounds 9]   # GPU: every section
    python Track_1/verify/fuse_ffno.py --quick                                             # CPU dry run

WHY THIS EXISTS. The second package line, P2-L8 = F-FNO L8 x4 + ViT x2, is two invocations only if the four
F-FNOs fuse (+1.151 board ms on Track 1 / +0.866 on Track 2 per extra invocation, round 23). The stack runs one
of two spectral paths -- "dense" (the 1-D FFTs as matmuls on a precomputed basis, fp16-native; Track 1's batch)
or "fft" (Track 2's batch 1) -- and each is checked against the member loop ON THE SAME PATH. Dense against fft
is a property of the model, not of the fold, and is reported as context (forge: 1.9e-6 physical in fp32).

SECTIONS (on CPU, 4 and the CUDA parts of 6 and 7 are skipped):
  1. vendored FFNO2d == models_alt.FFNO2d, BITWISE, on the real checkpoints: fft and dense, fp32 and fp16.
  2. synthetic members (ffno, ffno_nff) x (fft, dense), K = 3: fused vs loop; grouping (paths never share).
  3. real members, fp32, per path: raw outputs, predict_fno's physical field and sigma at alpha 0 and 0.60,
     K = 2 and K = 4. THE GATE: <= 2e-5 physical (fuse_vit.py's and fuse_swin.py's tolerance).
     3b: the local error per stage kind, both sides fed the loop's own inputs.
  4. fp16 autocast per path: fused fp16 vs fused fp32 (what both AMP self-checks compare, 2e-2), against the
     member loop in fp16, and both probes' metrics on F-FNO x2 and on F-FNO x2 + ViT x2.
  5. int8 packing (Linear tensor / row / fp16) -> load_any -> fuse on the fft path: deviation from the fp32
     loop and rel_l2 / tke / mvpe on val_unseen.
  6. packages in temporary directories, never staged (F-FNO x2 + ViT x2, int8 tensor, R24's band). No
     submission.py constant selects the path yet, so the archive's own _fno.py carries FFNO_PATH = "dense"
     (both tracks bake dense, main 09-14): T1 fp32 / fp16 / fp16 + GRAPH (AMP, graph, permutation, bounds), T2
     fp32 / fp16 (graph captured, replay vs eager on real windows).
  7. interleaved one-process timing, b64 (ms per window) and b1 CUDA-graph replay (ms per step), fp32 and fp16:
     F-FNO x1, x4 looped and x4 fused on each path, FNO x4 fused (SPEC bmm), ViT x2 fused; the P2-L8 vs R25
     step on each track's best path.

TF32 is disabled throughout, as in both submissions.
"""
from __future__ import annotations

import argparse
import contextlib
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

P2 = "/fs/scratch/PAS1187/qwu/tmp/realpde/screen/p2"
FFNO = [f"{P2}/p2_ffno_l8_s1.pth", f"{P2}/p2_ffno_l8_s2.pth"]
FNO4 = [os.path.join(T1, "unet", f"fno_all_w96L8cmb3sim_s{i}.pth") for i in (1, 2, 3, 4)]
VIT2 = [os.path.join(T1, "unet", n) for n in ("arch_vit_all_s1.pth", "arch_vit_all_cmb3sim_s3.pth")]
BAND = os.path.join(T1, "bands", "r24_xfam6.npz")
SUB1 = os.path.join(T1, "submit_fno", "submission.py")
SUB2 = os.path.join(REPO, "Track_2", "submit", "submission.py")
PATHS = ("fft", "dense")
TOL32 = 2e-5        # physical, fused vs loop, fp32 -- the gate (fuse_swin.py's TOL32 explains the size)
TOL_SYN = 1e-5      # normalized raw outputs of random-weight members
AMP_TRIP = 2e-2     # both tracks' AMP self-check threshold

OK: list = []
REC: dict = {}
MEMBERS: list = []


# ------------------------------------------------------------------------------ reporting
def rep(key, label, v, tol=None, fmt=".3e"):
    v = float(v)
    REC[key] = v
    if tol is None:
        print(f"  {label:<76s} {v:{fmt}}", flush=True)
        return
    good = v <= tol
    OK.append(good)
    print(f"  {label:<76s} {v:{fmt}}  {'OK' if good else 'FAIL'} (<= {tol:g})", flush=True)


def check(key, label, cond):
    REC[key] = bool(cond)
    OK.append(bool(cond))
    print(f"  {label:<76s} {'OK' if cond else 'FAIL'}", flush=True)


def guarded(name, fn, *a):
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
    return (torch.autocast("cuda", dtype=torch.float16, cache_enabled=False) if on
            else contextlib.nullcontext())


@torch.no_grad()
def fwd(m, x, amp=False):
    with amp_ctx(amp):
        f, ls = m(x)
    return f.float(), ls.float()


@torch.no_grad()
def loop(ms, x, amp=False):
    outs = [fwd(m, x, amp) for m in ms]
    f = torch.stack([o[0] for o in outs]).mean(0)
    s = torch.stack([o[1].clamp(-8.0, 2.0).exp() for o in outs]).mean(0)
    return f, s.log()


def ffno(paths, impl, dev, path=None):
    """F-FNO members through load_any with FFNO_IMPL forced to `impl`, on spectral `path`."""
    old = _fno.FFNO_IMPL
    _fno.FFNO_IMPL = impl
    try:
        ms = [_fno.load_any(p, dev) for p in paths]
    finally:
        _fno.FFNO_IMPL = old
    if path:
        for m in ms:
            m.set_spectral_path(path)
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


def perturbed(arch, seed, dev, path, width=16, my=6, mx_=10, nl=2):
    torch.manual_seed(seed)
    m = models_alt.build(arch, width, True, 0, my, mx_, nl, dropout=0.3).eval()
    with torch.no_grad():
        for q in m.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    return m.to(dev).set_spectral_path(path)


# ------------------------------------------------------------------------------ 1
def section1(dev, X):
    print("\n1. vendored FFNO2d == models_alt.FFNO2d on the real checkpoints (bitwise)")
    cuda = dev == "cuda"
    x = prev_of(X[:32], dev)
    for p in sorted(set(MEMBERS)):
        tag = os.path.basename(p)[:-4]
        a = ffno([p], "models_alt", dev)[0]
        v = ffno([p], "vendored", dev)[0]
        check(f"s1.{tag}.classes", f"{tag}: models_alt.FFNO2d / _fno.FFNO2d built",
              type(a) is models_alt.FFNO2d and type(v) is _fno.FFNO2d)
        for path in PATHS:
            a.set_spectral_path(path)
            v.set_spectral_path(path)
            for amp in ((False, True) if cuda else (False,)):
                fa, la = fwd(a, x, amp)
                fv, lv = fwd(v, x, amp)
                name = f"{path} {'fp16' if amp else 'fp32'}"
                rep(f"s1.{tag}.{name.replace(' ', '_')}", f"{tag}: {name}, max|d| field and logstd",
                    max(mx(fa, fv), mx(la, lv)), 0.0)


# ------------------------------------------------------------------------------ 2
def section2(dev):
    print("\n2. synthetic members (ffno, ffno_nff) x (fft, dense), K = 3: fused vs loop, and grouping")
    torch.manual_seed(123)
    x = torch.randn(4, 40, 32, 64, device=dev) * 0.5
    for arch in _fno.FFNO_VARIANTS:
        for path in PATHS:
            ms = [perturbed(arch, s, dev, path) for s in (1, 2, 3)]
            fu = _fno.fuse(ms)
            check(f"s2.{arch}.{path}.fused", f"{arch} {path}: 3 members -> one StackedFFNO on that path",
                  len(fu) == 1 and type(fu[0]).__name__ == "StackedFFNO" and fu[0].n_members == 3
                  and all(l.dense == (path == "dense") for l in fu[0].layers))
            f0, l0 = loop(ms, x)
            f1, l1 = fwd(fu[0], x)
            rep(f"s2.{arch}.{path}.field", f"{arch} {path}: field  max|loop - fused| (normalized)", mx(f0, f1), TOL_SYN)
            rep(f"s2.{arch}.{path}.logstd", f"{arch} {path}: logstd max|loop - fused|", mx(l0, l1), TOL_SYN)
    a, b = perturbed("ffno", 1, dev, "fft"), perturbed("ffno", 2, dev, "dense")
    g = _fno.fuse([a, b])
    check("s2.group.paths_apart", "an fft member and a dense member (equal shapes) -> two members",
          len(g) == 2 and g[0] is a and g[1] is b)
    g = _fno.fuse([a])
    check("s2.group.single", "a lone F-FNO member stays unfused", len(g) == 1 and g[0] is a)


# ------------------------------------------------------------------------------ 3
def section3(dev, X):
    print(f"\n3. real members, fp32, per path: StackedFFNO vs the member loop ({len(X)} val_unseen windows)")
    x = prev_of(X[:64], dev)
    out = {}
    for path in PATHS:
        ms = ffno(MEMBERS, "vendored", dev, path)
        fu = _fno.fuse(ffno(MEMBERS, "vendored", dev, path))
        check(f"s3.{path}.k2.fused", f"{path}: two real members -> one StackedFFNO",
              len(fu) == 1 and type(fu[0]).__name__ == "StackedFFNO")
        f0, l0 = loop(ms, x)
        f1, l1 = fwd(fu[0], x)
        rep(f"s3.{path}.k2.raw_field", f"{path} K=2 raw field  max|loop - fused| (normalized)", mx(f0, f1))
        rep(f"s3.{path}.k2.raw_logstd", f"{path} K=2 raw logstd max|loop - fused|", mx(l0, l1))
        for alpha in (0.0, 0.60):
            Pa, Sa = predict(ms, X, dev, alpha)
            Pb, Sb = predict(fu, X, dev, alpha)
            rep(f"s3.{path}.k2.a{alpha:.2f}.field", f"{path} K=2 predict_fno alpha {alpha:.2f}: field max|loop - fused|",
                np.abs(Pa - Pb).max(), TOL32)
            rep(f"s3.{path}.k2.a{alpha:.2f}.sigma", f"{path} K=2 predict_fno alpha {alpha:.2f}: sigma max|loop - fused|",
                np.abs(Sa - Sb).max(), TOL32)
        m4 = (MEMBERS * 2)[:4]
        ms4 = ffno(m4, "vendored", dev, path)
        fu4 = _fno.fuse(ffno(m4, "vendored", dev, path))
        check(f"s3.{path}.k4.fused", f"{path}: four members -> one StackedFFNO of 4",
              len(fu4) == 1 and getattr(fu4[0], "n_members", 0) == 4)
        Pa, Sa = predict(ms4, X, dev)
        Pb, Sb = predict(fu4, X, dev)
        rep(f"s3.{path}.k4.field", f"{path} K=4 predict_fno alpha 0.60: field max|loop - fused|", np.abs(Pa - Pb).max(), TOL32)
        rep(f"s3.{path}.k4.sigma", f"{path} K=4 predict_fno alpha 0.60: sigma max|loop - fused|", np.abs(Sa - Sb).max(), TOL32)
        out[path] = (ms, fu)
    Pf, _ = predict(out["fft"][0], X, dev)
    Pd, _ = predict(out["dense"][0], X, dev)
    rep("s3.loop_dense_vs_fft", "context: member loop dense vs fft, field max (physical)", np.abs(Pd - Pf).max())
    return out


@torch.no_grad()
def section3b(dev, X, out):
    print("\n3b. where the fused path departs from the loop: LOCAL max|d| per stage kind (K=2, fp32)")
    F = torch.nn.functional
    prev = prev_of(X[:16], dev)
    B = len(prev)
    for path in PATHS:
        ms, fu = out[path]
        st, K = fu[0], len(ms)
        tr = []
        for m in ms:
            o = {"in": torch.cat([prev, m.grid.expand(B, -1, -1, -1)], 1)}
            o["lift"] = m.lift(o["in"])
            h = F.pad(o["lift"], (0, m.PAD[1], 0, m.PAD[0]))
            for i, layer in enumerate(m.layers):
                o[f"L{i}.in"] = h
                o[f"L{i}.spec"] = layer._spectral_dense(h) if layer.dense else layer._spectral(h.float())
                h = layer(h)
                o[f"L{i}"] = h
            o["crop"] = h[..., :32, :64]
            o["norm_out"] = m.norm_out(o["crop"])
            o["field"] = m.head_field(o["norm_out"])
            o["logstd"] = m.head_logstd(o["norm_out"])
            tr.append(o)
        worst = {}

        def note(kind, v):
            worst[kind] = max(worst.get(kind, 0.0), v)

        def ch(key):
            return torch.cat([o[key] for o in tr], 1)

        def split(z, kk):
            return z.reshape(z.shape[0], K, -1, *z.shape[-2:])[:, kk]

        lifted = st.lift2(F.gelu(st.lift0(tr[0]["in"])))
        note("lift convs", max(mx(split(lifted, kk), tr[kk]["lift"]) for kk in range(K)))
        for i, sl in enumerate(st.layers):
            xin = ch(f"L{i}.in")
            spec = sl._spectral_dense(xin) if sl.dense else sl._spectral_fft(xin.float())
            note("spectral operators", max(mx(split(spec, kk), tr[kk][f"L{i}.spec"]) for kk in range(K)))
            note("whole layers", max(mx(split(sl(xin), kk), tr[kk][f"L{i}"]) for kk in range(K)))
        t = st.norm_out(ch("crop").reshape(B, K, -1, 32, 64).permute(1, 0, 3, 4, 2))
        note("norm_out", max(mx(t[kk].permute(0, 3, 1, 2), tr[kk]["norm_out"]) for kk in range(K)))
        tt = torch.stack([o["norm_out"].permute(0, 2, 3, 1) for o in tr])
        f = st.hf[1](F.gelu(st.hf[0](tt)))
        note("heads", max(mx(f[kk].permute(0, 3, 1, 2), tr[kk]["field"]) for kk in range(K)))
        lg = st.hl[1](F.gelu(st.hl[0](tt)))
        note("heads", max(mx(lg[kk].permute(0, 3, 1, 2), tr[kk]["logstd"]) for kk in range(K)))
        for kind, v in worst.items():
            rep(f"s3b.{path}.{kind.replace(' ', '_')}", f"{path}: local, worst over the {kind}", v)


# ------------------------------------------------------------------------------ 4
def t1_probe(models, dev):
    """Track 1's `_amp_verified` metric, verbatim."""
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
    """Track 2's `_amp_verified` metric, verbatim."""
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


def section4(dev, X, out):
    print("\n4. fp16 autocast per path -- both tracks' AMP self-checks fall back to fp32 above 2e-2")
    x = prev_of(X[:64], dev)
    vit = _fno.fuse([_fno.load_any(p, dev) for p in VIT2])
    for path in PATHS:
        ms, fu = out[path]
        F32, L32 = fwd(fu[0], x, False)
        F16, L16 = fwd(fu[0], x, True)
        rep(f"s4.{path}.fused16_vs_fused32.field_max", f"{path}: fused fp16 vs fused fp32, field max (normalized)", mx(F16, F32))
        rep(f"s4.{path}.fused16_vs_fused32.field_rel", f"{path}: fused fp16 vs fused fp32, field rel-L2", rel(F16, F32))
        rep(f"s4.{path}.fused16_vs_fused32.logstd_max", f"{path}: fused fp16 vs fused fp32, logstd max", mx(L16, L32))
        fl, ll = loop(ms, x, True)
        rep(f"s4.{path}.fused16_vs_loop16.field_max", f"{path}: fused fp16 vs member loop fp16, field max", mx(F16, fl))
        rep(f"s4.{path}.fused16_vs_loop16.logstd_max", f"{path}: fused fp16 vs member loop fp16, logstd max", mx(L16, ll))
        f32b, _ = fwd(fu[0], x[:1], False)
        f16b, _ = fwd(fu[0], x[:1], True)
        rep(f"s4.{path}.b1.field_max", f"{path}: b1 fused fp16 vs fp32, field max", mx(f16b, f32b))
        Pa, Sa = predict(fu, X, dev, 0.60, None)
        Pb, Sb = predict(fu, X, dev, 0.60, torch.float16)
        rep(f"s4.{path}.phys.field_max", f"{path}: predict_fno fused fp16 vs fp32, field max (physical)", np.abs(Pb - Pa).max())
        rep(f"s4.{path}.phys.sigma_max", f"{path}: predict_fno fused fp16 vs fp32, sigma max (physical)", np.abs(Sb - Sa).max())
        for label, models in ((f"F-FNO x2 fused ({path})", fu), (f"F-FNO x2 + ViT x2 fused ({path})", fu + vit)):
            key = "ffno2" if len(models) == 1 else "ffno2_vit2"
            rep(f"s4.{path}.probe_t1.{key}", f"Track 1 AMP probe metric, {label}", t1_probe(models, dev), AMP_TRIP)
            rep(f"s4.{path}.probe_t2.{key}", f"Track 2 AMP probe metric, {label}", t2_probe(models, dev), AMP_TRIP)


# ------------------------------------------------------------------------------ 5
def section5(dev, X, Y, tmp):
    print(f"\n5. int8 packing on the fft path: pack_state_dict -> load_any -> fuse, against the fp32 loop "
          f"({len(X)} val_unseen windows)")
    from sps9 import sc

    def metrics(P):
        return (float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))), float(sc.mvpe_rel_l2(P, Y)))

    P0, S0 = predict(ffno(MEMBERS, "vendored", dev, "fft"), X, dev)
    m0 = metrics(P0)
    REC["s5.fp32_loop.metrics"] = m0
    print(f"  fp32 member loop: rel_l2 {m0[0]:.5f}  tke {m0[1]:.5f}  mvpe {m0[2]:.5f}")
    for linear in ("tensor", "row", "fp16"):
        d = os.path.join(tmp, f"int8_{linear}")
        os.makedirs(d, exist_ok=True)
        paths = []
        for i, p in enumerate(MEMBERS):
            blob = torch.load(p, map_location="cpu")
            blob["model_state_dict"] = _fno.pack_state_dict(blob["model_state_dict"], fp16=True, int8=True, linear=linear)
            q = os.path.join(d, f"m{i}.pth")
            torch.save(blob, q)
            paths.append(q)
        P, S = predict(_fno.fuse(ffno(paths, "vendored", dev, "fft")), X, dev)
        mq = metrics(P)
        REC[f"s5.{linear}.metrics"] = mq
        rep(f"s5.{linear}.field_rel", f"linear={linear}: ||dP|| / ||P|| (quant8.py's metric)", np.linalg.norm(P - P0) / np.linalg.norm(P0))
        rep(f"s5.{linear}.field_max", f"linear={linear}: field max|d| (physical)", np.abs(P - P0).max())
        rep(f"s5.{linear}.sigma_max", f"linear={linear}: sigma max|d|", np.abs(S - S0).max())
        rep(f"s5.{linear}.mb", f"linear={linear}: MB per member", sum(map(os.path.getsize, paths)) / 1e6 / len(paths), fmt=".2f")
        print(f"    rel_l2 {mq[0]:.5f} ({mq[0] - m0[0]:+.5f})  tke {mq[1]:.5f} ({mq[1] - m0[1]:+.5f})"
              f"  mvpe {mq[2]:.5f} ({mq[2] - m0[2]:+.5f})", flush=True)


# ------------------------------------------------------------------------------ 6
def make_pkg(tmp, track, members, consts, ffno_path):
    d = tempfile.mkdtemp(prefix=f"t{track}ffno_", dir=tmp)
    txt = open(SUB1 if track == 1 else SUB2).read()
    for k, v in consts.items():
        txt, n = re.subn(rf"(?m)^{k} = .*$", f"{k} = {v}", txt)
        if n != 1:
            raise RuntimeError(f"track {track}: expected one `{k} = ` line, found {n}")
    open(os.path.join(d, "submission.py"), "w").write(txt)
    fno_txt, n = re.subn(r"(?m)^FFNO_PATH = None$", f'FFNO_PATH = "{ffno_path}"',
                         open(os.path.join(T1, "submit_fno", "_fno.py")).read())
    if n != 1:
        raise RuntimeError(f"expected one `FFNO_PATH = None` line in _fno.py, found {n}")
    open(os.path.join(d, "_fno.py"), "w").write(fno_txt)
    shutil.copy2(BAND, os.path.join(d, "band.npz"))
    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(members) + 1)]
    for src, name in zip(members, names):
        blob = torch.load(src, map_location="cpu")
        blob["model_state_dict"] = _fno.pack_state_dict(blob["model_state_dict"], fp16=True, int8=True, linear="tensor")
        torch.save(blob, os.path.join(d, name))
    return d


def load_pkg(d, name):
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
    members = MEMBERS + VIT2
    Xp = Xall[:150] if cuda else Xall[:6]
    print(f"\n6. packages in temporary directories: F-FNO x2 + ViT x2, int8 tensor, band r24_xfam6 ({len(Xp)} windows)")
    t1_variants = [("fp32", {"AMP": "None", "GRAPH": "False"})]
    if cuda:
        t1_variants += [("fp16", {"AMP": '"fp16"', "GRAPH": "False"}), ("fp16_graph", {"AMP": '"fp16"', "GRAPH": "True"})]
    for tag, consts in t1_variants:
        d = make_pkg(tmp, 1, members, dict(consts, FUSE="True", ALPHA="0.60"), "dense")
        sub = load_pkg(d, f"t1ffno_{tag}")
        st = sub._state
        check(f"s6.t1.{tag}.loaded", f"T1 {tag}: members loaded at import", st is not None)
        if st is None:
            continue
        kinds = [type(o).__name__ for o in st[0]]
        check(f"s6.t1.{tag}.groups", f"T1 {tag}: invocations {kinds}, F-FNO on the dense path",
              kinds == ["StackedFFNO", "StackedViT"] and all(l.dense for l in st[0][0].layers))
        check(f"s6.t1.{tag}.vendored", f"T1 {tag}: the archive's _fno resolves F-FNO to its vendored class",
              sub._fno.FFNO_IMPL == "auto" and not os.path.exists(os.path.join(sub._fno._UNET_DIR, "models_alt.py")))
        REC[f"s6.t1.{tag}.state"] = dict(AMP_ACTIVE=sub.AMP_ACTIVE, AMP_PROBE_DIFF=sub.AMP_PROBE_DIFF,
                                         GRAPH_ACTIVE=getattr(sub, "GRAPH_ACTIVE", None), GRAPH_PROBE_DIFF=getattr(sub, "GRAPH_PROBE_DIFF", None), batch=st[5])
        print(f"    {REC[f's6.t1.{tag}.state']}", flush=True)
        if consts["AMP"] != "None":
            check(f"s6.t1.{tag}.amp", f"T1 {tag}: AMP self-check kept fp16 (probe {sub.AMP_PROBE_DIFF})", sub.AMP_ACTIVE == "float16")
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
        d = make_pkg(tmp, 2, members, dict(consts, FUSE="True", ALPHA="0.60"), "dense")
        sub = load_pkg(d, f"t2ffno_{tag}")
        m = sub.get_ttt_model(d, dev)
        kinds = [type(o).__name__ for o in m.models]
        check(f"s6.t2.{tag}.groups", f"T2 {tag}: invocations {kinds}, F-FNO on the dense path",
              kinds == ["StackedFFNO", "StackedViT"] and all(l.dense for l in m.models[0].layers))
        REC[f"s6.t2.{tag}.state"] = dict(AMP_ACTIVE=sub.AMP_ACTIVE, AMP_PROBE_DIFF=sub.AMP_PROBE_DIFF, graph=m.graph is not None)
        print(f"    {REC[f's6.t2.{tag}.state']}", flush=True)
        if consts["AMP"] != "None":
            check(f"s6.t2.{tag}.amp", f"T2 {tag}: AMP self-check kept fp16 (probe {sub.AMP_PROBE_DIFF})", sub.AMP_ACTIVE == "float16")
        if cuda:
            check(f"s6.t2.{tag}.graph", f"T2 {tag}: CUDA graph captured (load-time check demands 0.0)", m.graph is not None)
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
            rep(f"s6.t2.{tag}.replay", f"T2 {tag}: graph replay vs eager, 8 real windows (pred, lower, upper)", worst, 0.0)
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
    _fno._SPEC = "bmm"
    m4 = (MEMBERS * 2)[:4]
    arms = {"fno4_fused": _fno.fuse([_fno.load_any(p, dev) for p in FNO4]),
            "vit2_fused": _fno.fuse([_fno.load_any(p, dev) for p in VIT2])}
    for path in PATHS:
        arms[f"ffno1_{path}"] = ffno(MEMBERS[:1], "vendored", dev, path)
        arms[f"ffno4_loop_{path}"] = ffno(m4, "vendored", dev, path)
        arms[f"ffno4_fused_{path}"] = _fno.fuse(ffno(m4, "vendored", dev, path))
    print("  arms: " + "; ".join(f"{k} = {[type(m).__name__ for m in v]}" for k, v in arms.items()))
    x64 = prev_of(X[:64] if cuda else X[:2], dev)
    nb = len(x64)

    @torch.no_grad()
    def step(ms, x, amp):
        with amp_ctx(amp):
            return [m(x) for m in ms]

    settings = [("b64", True), ("b1g", True), ("b64", False), ("b1g", False)] if cuda else [("b64", False)]
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
            niter[(name, mode, amp)] = int(min(2000, max(3, math.ceil(0.06 / max(time.perf_counter() - t, 1e-6)))))
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
        table[name] = {f"{mode}_{'fp16' if amp else 'fp32'}": 1e3 * float(np.median(acc[(name, mode, amp)][1:] or acc[(name, mode, amp)]))
                       / (nb if mode == "b64" else 1) for mode, amp in settings}
    REC["s7.ms"] = table
    cols = list(next(iter(table.values())))
    print("  ms: b64 per window, b1g per step\n  " + f"{'arm':<18s}" + "".join(f"{c:>11s}" for c in cols)
          + "".join(f"{'x' + c:>13s}" for c in cols))
    for name, row in table.items():
        print("  " + f"{name:<18s}" + "".join(f"{row[c]:11.4f}" for c in cols)
              + "".join(f"{row[c] / table['fno4_fused'][c]:13.3f}" for c in cols), flush=True)
    for c in cols:
        best = min(PATHS, key=lambda p: table[f"ffno4_fused_{p}"][c])
        p2 = table[f"ffno4_fused_{best}"][c] + table["vit2_fused"][c]
        r25 = table["fno4_fused"][c] + table["vit2_fused"][c]
        REC[f"s7.p2_vs_r25.{c}"] = dict(path=best, p2=p2, r25=r25)
        print(f"  {c:<9s} fused/loop fft {table['ffno4_fused_fft'][c] / table['ffno4_loop_fft'][c]:.3f}  "
              f"dense {table['ffno4_fused_dense'][c] / table['ffno4_loop_dense'][c]:.3f}   "
              f"P2-L8 ({best}) {p2:.4f} vs R25 {r25:.4f} (x{p2 / r25:.3f}, {p2 - r25:+.4f} ms)", flush=True)
    # THE SPEED GATE (main 09-14): the stack earns its place only if it is at least as fast as the member loop it
    # replaces at both tracks' operating points, on the path both tracks bake (dense). rp_pkg_p2ffno waits on it.
    if cuda:
        fast = all(table["ffno4_fused_dense"][c] <= table["ffno4_loop_dense"][c] for c in ("b64_fp16", "b1g_fp16"))
        check("s7.speed_gate", "SPEED GATE: fused F-FNO x4 dense <= its member loop at b64 fp16 and b1g fp16", fast)


# ------------------------------------------------------------------------------ main
def main() -> None:
    global MEMBERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffno", nargs="+", default=None, help="held-out F-FNO checkpoints (default L8 s1 + s2)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--quick", action="store_true", help="CPU dry run on a few windows")
    args = ap.parse_args()
    MEMBERS = args.ffno or [p for p in FFNO if os.path.exists(p)]
    if len(MEMBERS) == 1:
        MEMBERS = MEMBERS * 2           # two distinct modules from one file until seed 2 lands
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if (torch.cuda.is_available() and not args.quick) else "cpu"
    cuda = dev == "cuda"
    if not cuda:
        torch.set_num_threads(8)
    z = np.load(os.path.join(T1, "val_unseen.npz"))
    Xall, Yall = z["x"].astype(np.float32), z["y"].astype(np.float32)
    X = Xall[np.linspace(0, len(Xall) - 1, 128 if cuda else 8).round().astype(int)]
    md5 = hashlib.md5(open(_fno.__file__, "rb").read()).hexdigest()
    REC["env"] = dict(torch=torch.__version__, device=torch.cuda.get_device_name() if cuda else "cpu",
                      fno_md5=md5, ffno=MEMBERS)
    print(f"device {REC['env']['device']}  torch {torch.__version__}  TF32 off  _fno.py {md5}")
    print("F-FNO members: " + ", ".join(os.path.basename(p) for p in MEMBERS))
    tmp = tempfile.mkdtemp(prefix="fuse_ffno_")
    try:
        guarded("section1", section1, dev, X)
        guarded("section2", section2, dev)
        out = guarded("section3", section3, dev, X)
        if out is not None:
            guarded("section3b", section3b, dev, X, out)
            if cuda:
                guarded("section4", section4, dev, X, out)
        del out
        gc.collect()
        k = len(Xall) if cuda else 8
        guarded("section5", section5, dev, Xall[:k], Yall[:k], tmp)
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
