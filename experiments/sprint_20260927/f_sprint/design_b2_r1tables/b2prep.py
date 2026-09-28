"""Design B2 step 0: cache the PREDICTION-side stream (P, centre C after the D1c shift, static half-width h, stretched target Yf)
with design C's own probe (stream_c/probe.py, copied read-only), so band-side variants can be replayed fast on CPU.
    python b2prep.py <in|ood> <d1c_n0> <f> [vc_n0]
Writes $S/ckpt/b2/<tag>.npz with P C h Y names a (per-window accuracy factor) and the asym-R1 reference bounds' sps."""
import os, sys, time
import numpy as np, torch
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "stream_c")
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927"
src = open(os.path.join(SC, "probe.py")).read()
hook = "        Pn[i], Ln[i], Un[i] = P, lo, hi\n"
assert src.count(hook) == 1
src = src.replace(hook, hook + "        D['_C'][i] = C; D['_h'][i] = h\n")
ns = {"__file__": os.path.join(SC, "probe.py"), "__name__": "probe_c"}
exec(compile(src, "probe_c", "exec"), ns)
which, d1c, f = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); vc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if which == "in":
    D = ns["load"](f"{S}/ckpt/c/twins_unseen.npz", (0, 1, 2), (3, 4))
else:
    D = ns["load"](f"{S}/ckpt/c/ood_aoa20.npz", (0, 1), (4,))
band = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/bands/r35_centre0_fill.npz")
t0 = time.time()
D["_C"] = np.zeros_like(D["Y"]); D["_h"] = np.zeros_like(D["Y"])
D["Pref"] = ns["run"](D, band, {}, 1.0)[0]
cfg = dict(d1c_n0=d1c, d1c_after_fw=False) if d1c > 0 else {}
if vc > 0:
    cfg["vc_n0"] = vc
P, L, U, Yf = ns["run"](D, band, cfg, f)
sc = ns["sc"]
pad = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1).astype(np.float32)
P3, Y3 = pad(P), pad(Yf)
dm, tk, mv = sc.rel_l2_per_sample(P3, Y3, 2), sc.tke_rel_l2_per_sample(P3, Y3, 2), sc.mvpe_rel_l2_per_sample(P3, Y3)
g = lambda m: 0.5 / (0.5 + m)
a = 0.5 * g(dm) + 0.3 * g(tk) + 0.2 * g(mv); a = np.where(np.isfinite(a), a, 0.0)
ref = ns["score"](P, L, U, Yf)
tag = f"{which}_d{int(d1c)}_f{int(round(f * 100))}" + (f"_vc{int(vc)}" if vc > 0 else "")
np.savez(f"{S}/ckpt/b2/{tag}.npz", P=P, C=D["_C"], h=D["_h"], Y=Yf, names=D["names"], a=a, dm=dm, tk=tk, mv=mv,
         Lref=L, Uref=U, ref_sps=ref["sps"], ref_comp=ref["comp"], ref_rel=ref["rel"], ref_tke=ref["tke"], ref_mvpe=ref["mvpe"])
print(tag, "windows", len(P), "ref", {k: round(float(v), 4) for k, v in ref.items()}, "%.0fs" % (time.time() - t0), flush=True)
