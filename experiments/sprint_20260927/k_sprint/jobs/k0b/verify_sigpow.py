"""Verifier for the T2 SIGPOW knob on an extracted package (static band, R1 off):
graph vs eager bit-exact over the 336-step stream; lower<=P<=upper, finite; width == numpy formula of design_0 s3 (+contain-P)."""
import sys, os, json, numpy as np, torch
W = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_0b"
sys.path.insert(0, W + "/Track_2"); sys.path.insert(0, W + "/Track_1")
pkg, npz = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
import stream_eval as se, t2_common as t2
from pathlib import Path
mod = se.import_submission(Path(pkg).resolve())
m = mod.get_ttt_model(str(Path(pkg).resolve()), "cuda")
assert m.graph is not None, "graph NOT captured"
assert mod.SIGPOW is not None and not mod.R1
stream = t2.build_stream(t2.UNSEEN)
Pg, Lg, Ug, Y, _ = se.run(m, stream, "cuda")
rec = []
orig = m._group
def grp(mm, pv, err):
    o = orig(mm, pv, err); rec.append(o); return o
m.graph = None; m._group = grp
steps = []
Pe, Le, Ue = [], [], []
for name, tid, x_raw, y_raw, first in stream:
    if first: m.reset_ttt_state()
    rec.clear()
    p, info = m.ttt_step(torch.from_numpy(t2.normalize_in(x_raw)[None]), None); torch.cuda.synchronize()
    tot = sum(getattr(g, "n_members", 1) for g in m.models)
    y0 = sum(o[0] for o in rec) / tot; sg = sum(o[1] for o in rec) / tot
    steps.append((y0.cpu().numpy()[0], sg.cpu().numpy()[0]))
    Pe.append(t2.denorm_pred(np.asarray(p)[0])); Le.append(t2.denorm_pred(np.asarray(info["lower"])[0])); Ue.append(t2.denorm_pred(np.asarray(info["upper"])[0]))
Pe, Le, Ue = map(np.stack, (Pe, Le, Ue))
print(f"graph vs eager bit-exact: pred {np.array_equal(Pg,Pe)} lower {np.array_equal(Lg,Le)} upper {np.array_equal(Ug,Ue)}  "
      f"max|d| {max(np.abs(Pg-Pe).max(), np.abs(Lg-Le).max(), np.abs(Ug-Ue).max()):.3e}")
fin = all(np.isfinite(a).all() for a in (Pg, Lg, Ug))
print(f"finite {fin}  lower<=pred {bool((Lg[...,:2] <= Pg[...,:2]).all())}  pred<=upper {bool((Pg[...,:2] <= Ug[...,:2]).all())}")
b = np.load(npz); base = b["base"][0, ..., :2].astype(np.float64); sref = b["sigma_ref"][0, ..., :2].astype(np.float64); p = float(b["p"])
q = np.array(mod.SIGPOW, np.float64)
STD = mod._fno.STD.astype(np.float64).reshape(-1)[:2]; MEAN = mod._fno.MEAN.astype(np.float64).reshape(-1)[:2]
wd_all, rel_all = [], []
for i, (y0, sg) in enumerate(steps):
    sp = sg.astype(np.float64) * STD
    half = base * np.clip(sp / sref, 0.3, 3.0) ** p * (sp + 1e-8) ** q
    c = y0.astype(np.float64) * STD + MEAN      # centre0 in physical units
    P = Pg[i, ..., :2].astype(np.float64)
    lo = np.minimum(c - half, P); hi = np.maximum(c + half, P)
    wd_all.append(np.abs((hi - lo) - (Ug[i, ..., :2] - Lg[i, ..., :2])).max())
    rel_all.append((np.abs((hi - lo) - (Ug[i, ..., :2] - Lg[i, ..., :2])) / np.maximum(hi - lo, 1e-9)).max())
print(f"width vs numpy formula (design_0 s3, contain-P): max|d| {max(wd_all):.3e} phys units, max rel {max(rel_all):.3e}; "
      f"mean width {float((Ug[...,:2]-Lg[...,:2]).mean()):.5f}")
