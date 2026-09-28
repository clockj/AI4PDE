"""EC: aggregate ec_matrix.py json into the gauge + candidate matrix."""
import json, glob, os, sys
O = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c_work/mx"
LAT = {}   # arm -> added ms/step vs d1c30 (filled from argv json if given)
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    LAT = json.load(open(sys.argv[2]))
R = {}
for p in glob.glob(f"{O}/*.json"):
    b = os.path.basename(p)[:-5]; reg, f, arm = b.split("_", 2)
    R[(reg, f, arm)] = json.load(open(p))
BOARD = dict(rel=94.770908, tke=79.515595, mvpe=94.131921, sps=43.556207)
W = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
e = lambda s: 2 * (100 / s - 1)
BOARD.update(rel_e=e(BOARD["rel"]), tke_e=e(BOARD["tke"]), mvpe_e=e(BOARD["mvpe"]), comp=sum(W[q] * BOARD[q] for q in W), cov=float("nan"))
print("== GAUGE (baseline = asym R1, no D1c; band r38)")
print(f"{'regime':26s} {'rel':>7s} {'tke':>7s} {'mvpe':>7s} {'sps':>7s} | {'rel_e':>7s} {'tke_e':>7s} {'mvpe_e':>7s} {'cov':>6s} | comp")
rows = {}
for f in ("f1.00", "f1.29"):
    for lab, key, part in (("in full", ("in", f, "base"), "full"), ("in AoA15-20", ("in", f, "base"), "a15"), ("OOD AoA20", ("ood", f, "base"), "full")):
        if key in R:
            d = R[key][part]; rows[(lab, f)] = d
            print(f"{lab+' '+f:26s} {d['rel']:7.3f} {d['tke']:7.3f} {d['mvpe']:7.3f} {d['sps']:7.3f} | {d['rel_e']:7.4f} {d['tke_e']:7.4f} {d['mvpe_e']:7.4f} {d['cov']:6.3f} | {d['comp']:.3f}")
d = BOARD
print(f"{'BOARD T2 09-25':26s} {d['rel']:7.3f} {d['tke']:7.3f} {d['mvpe']:7.3f} {d['sps']:7.3f} | {d['rel_e']:7.4f} {d['tke_e']:7.4f} {d['mvpe_e']:7.4f} {'  -':>6s} | {d['comp']:.3f}")
for f in ("f1.00", "f1.29"):
    a, o = rows.get(("in AoA15-20", f)), rows.get(("OOD AoA20", f))
    if a and o:
        lam = {q: (BOARD[q] - a[q]) / (o[q] - a[q]) for q in ("rel_e", "tke_e", "mvpe_e", "sps", "comp")}
        print(f"lambda {f}: " + "  ".join(f"{q} {v:+.2f}" for q, v in lam.items()))
print()
arms = sorted({k[2] for k in R if k[0] == "in" and k[1] == "f1.00"} - {"base"})
res = []
for arm in arms:
    def dd(reg, f, part):
        k, b = (reg, f, arm), (reg, f, "base")
        if k not in R or b not in R: return None, ""
        x, y = R[k], R[b]
        dp = {t: x["per"][t] - y["per"][t] for t in x["per"]}
        if part == "a15": dp = {t: v for t, v in dp.items() if x["aoa"][t] >= 15}
        return x[part]["comp"] - y[part]["comp"], f"{sum(v > 0 for v in dp.values())}/{len(dp)}"
    din, pin = dd("in", "f1.00", "full"); d15, p15 = dd("in", "f1.00", "a15"); dood, pood = dd("ood", "f1.00", "full")
    d15b, _ = dd("in", "f1.29", "a15")
    lat = LAT.get(arm, 0.0) * 0.07 * 0.0 if False else LAT.get(arm, 0.0)
    E = {l: (1 - l) * d15 + l * dood - 0.07 * lat if (d15 is not None and dood is not None) else float("nan") for l in (0.25, 0.5, 0.75)}
    res.append((arm, din, pin, d15, p15, d15b, dood, pood, lat, E))
res.sort(key=lambda r: -r[9][0.75])
print(f"{'arm':14s} {'d_in_full':>10s} {'pos':>5s} {'d_in_a15':>9s} {'pos':>4s} {'a15@1.29':>9s} {'d_OOD':>8s} {'pos':>6s} {'+ms':>6s} | E(.25) E(.5) E(.75)")
for arm, din, pin, d15, p15, d15b, dood, pood, lat, E in res:
    f = lambda v: f"{v:+.4f}" if v is not None else "   n/a"
    print(f"{arm:14s} {f(din):>10s} {pin:>5s} {f(d15):>9s} {p15:>4s} {f(d15b):>9s} {f(dood):>8s} {pood:>6s} {lat:6.3f} | {E[0.25]:+.4f} {E[0.5]:+.4f} {E[0.75]:+.4f}")
