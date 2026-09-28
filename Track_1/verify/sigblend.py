"""SIGBLEND band verifier (E0a, 09-27).  From a Track-1 package's OWN outputs on val_unseen (k0a_dump.py: prediction, lower,
upper, the package's sigma head and alpha-0 centre) recompute the intended interval in numpy

    half = base' * clip(sigma/sigma_ref, .3, 3)^p' * exp(lam*q_c * log(sigma + 1e-8))
    [lower, upper] = [min(c0 - half, P), max(c0 + half, P)]

and compare it with what the package returned.  Also asserts: prediction identical to the shipped-band package, lower <= P <=
upper, finite, channel p exactly 0 in all three arrays.
usage: sigblend.py <ship_dump.npz> <sb_dump.npz> <sigblend band.npz> <EBAND literal or .json>"""
import ast, json, sys
import numpy as np
ship, sb, bandp, eb = sys.argv[1:5]
if eb.endswith(".json"):
    eb = json.load(open(eb))["EBAND"]
beta, mode, clip = ast.literal_eval(eb)
assert mode == "mult" and clip is None and all(v == 0.0 for b in beta for v in b[1:]), "not a SIGBLEND EBAND"
lq = np.array([beta[0][0], beta[1][0]], np.float32)
A, B = np.load(ship), np.load(sb)
z = np.load(bandp); base = z["base"].astype(np.float32)[..., :2]; ref = z["sigma_ref"].astype(np.float32)[..., :2]; p = np.float32(z["p"])
P, lo, hi, sig, c0 = B["P"], B["lo"], B["hi"], B["sig"], B["c0"]
fails = []
def chk(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"); fails.append(name) if not ok else None
dP = float(np.abs(A["P"] - P).max())
chk(f"prediction identical to shipped-band package (max|d| {dP:.3e})", dP == 0.0)
chk("lower, upper, prediction finite", all(np.isfinite(a).all() for a in (P, lo, hi)))
chk("lower <= prediction <= upper everywhere", bool((lo <= P).all() and (P <= hi).all()))
chk("channel p exactly 0 in prediction, lower, upper", all(float(np.abs(a[..., 2]).max()) == 0.0 for a in (P, lo, hi)))
chk("centre (alpha-0 field) identical between packages", float(np.abs(A["c0"] - c0).max()) == 0.0)
half = (base * np.clip(sig / ref, 0.3, 3.0) ** p * np.exp(lq * np.log(sig + np.float32(1e-8)))).astype(np.float32)
Pu = P[..., :2]
elo, ehi = np.minimum(c0 - half, Pu), np.maximum(c0 + half, Pu)
for nm, a, b in (("lower", lo[..., :2], elo), ("upper", hi[..., :2], ehi), ("width", (hi - lo)[..., :2], ehi - elo)):
    d = np.abs(a.astype(np.float64) - b); r = d / np.maximum(np.abs(b), 1e-6)
    print(f"  {nm:<6s} package vs numpy formula: max|d| {d.max():.3e}  max rel {r.max():.3e}  mean|d| {d.mean():.3e}")
    chk(f"{nm} matches formula to float32 noise", d.max() < 1e-5)
# the same check for the half width where the contain-P rule does not bind
free = (c0 - half <= Pu) & (Pu <= c0 + half)
w = (hi - lo)[..., :2]; rw = np.abs(w[free] - 2 * half[free]) / (2 * half[free])
print(f"  half-width where contain-P is inactive ({100*free.mean():.1f}% of elements): max rel |w - 2 half| {rw.max():.3e}")
# and the shipped-band package vs ITS formula (base, p of the shipped band, no EBAND)
print("  width ratio SIGBLEND / shipped (median, u / v):", [float(np.median(w[..., c] / (A["hi"] - A["lo"])[..., c])) for c in (0, 1)])
print("VERIFIER", "PASS" if not fails else "FAIL " + str(fails))
