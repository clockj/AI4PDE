"""E0a edge cases for an EXTRACTED Track-1 package (EBAND on, SMALL_GRAPH off).
usage: k0a_edge.py <pkgdir> seq            # one process: N in NS, f32 then f64, repeated calls, permutation
       k0a_edge.py <pkgdir> one <N> <dt>   # fresh process, a single call"""
import importlib.util, os, sys
import numpy as np
pkg, mode = sys.argv[1], sys.argv[2]
d = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
X, Y = d["x"], d["y"]
spec = importlib.util.spec_from_file_location("sc", "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
NS = (1, 2, 13, 14, 56, 99, 100, 129, 336)
bad = []


def check(o, n, tag):
    ok = isinstance(o, dict) and set(o) == {"prediction", "lower", "upper"}
    P, lo, hi = o["prediction"], o["lower"], o["upper"]
    ok &= all(a.shape == (n, 20, 32, 64, 3) and a.dtype == np.float32 and np.isfinite(a).all() for a in (P, lo, hi))
    ok &= bool((lo <= P).all() and (P <= hi).all())
    ok &= all(float(np.abs(a[..., 2]).max()) == 0.0 for a in (P, lo, hi))
    print(f"  {tag:<28s} N={n:<4d} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        bad.append(tag)
    return P, lo, hi


def rel(a, b):
    return float(np.abs(a[..., :2] - b[..., :2]).max()), float(np.abs(a[..., :2] - b[..., :2]).mean() / np.sqrt((b[..., :2] ** 2).mean()))


if mode == "one":
    n, dt = int(sys.argv[3]), sys.argv[4]
    try:
        check(sub.predict(X[:n].astype(dt)), n, f"fresh-process {dt}")
    except Exception as e:
        print(f"  fresh-process {dt} N={n} CRASH {e!r}"); bad.append("crash")
    print("RESULT", "PASS" if not bad else "FAIL"); sys.exit(1 if bad else 0)

print("EBAND", sub.EBAND, "SMALL_GRAPH", sub.SMALL_GRAPH, flush=True)
ref = check(sub.predict(X), 336, "full f32 (reference)")
for dt in ("float32", "float64"):
    for rep in (1, 2):
        for n in NS:
            s = slice(336 - n, 336) if n % 2 else slice(0, n)
            try:
                P, lo, hi = check(sub.predict(X[s].astype(dt)), n, f"{dt} call{rep}")
            except Exception as e:
                print(f"  {dt} call{rep} N={n} CRASH {e!r}", flush=True); bad.append(f"crash {dt} {n}"); continue
            if rep == 1:
                print("      vs full-batch same windows: pred max|d| %.2e (mean %.1e rms)  lower %.2e  upper %.2e" %
                      (*rel(P, ref[0][s]), rel(lo, ref[1][s])[0], rel(hi, ref[2][s])[0]), flush=True)
perm = np.random.default_rng(0).permutation(336)
o = sub.predict(X[perm]); P, lo, hi = check(o, 336, "permuted")
iv = np.argsort(perm)
Pr, lr, hr = P[iv], lo[iv], hi[iv]
print("  permutation: pred max|d| %.2e (mean %.1e rms)  lower %.2e upper %.2e" % (*rel(Pr, ref[0]), rel(lr, ref[1])[0], rel(hr, ref[2])[0]))
for nm, f in (("rel", lambda P: np.mean(sc.rel_l2_per_sample(P, Y, 2))), ("tke", lambda P: np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))),
              ("mvpe", lambda P: sc.mvpe_rel_l2(P, Y))):
    a, b = f(ref[0]), f(Pr); print(f"  permutation {nm}: {a:.7f} -> {b:.7f} (|d| {abs(a-b):.1e})"); bad += [nm] if abs(a - b) >= 1e-5 else []
w0 = sc.aggregate_sps(ref[0], Y, 2, lower=ref[1], upper=ref[2])[0]; w1 = sc.aggregate_sps(Pr, Y, 2, lower=lr, upper=hr)[0]
print(f"  permutation sps: {100*w0:.5f} -> {100*w1:.5f}"); bad += ["sps"] if abs(w0 - w1) * 100 >= 1e-3 else []
print("RESULT", "PASS" if not bad else "FAIL " + str(bad))
