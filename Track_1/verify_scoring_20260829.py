"""Anchor-verify the v9 `scoring.py` RECONSTRUCTION against two zero-parameter predictors.

The official kit is auth-walled from the OSC node, so `~/RealPDE/realpde_t1_starting_kit_v9/
realpde_t1_starting_kit_v9/scoring.py` is a reimplementation.  It already agrees with
`sps9.py`'s independent fast path, but that only cross-checks SPS against another piece of our
own code.  This script is the EXTERNAL check: two predictors whose scores were measured through
the REAL kit on this exact split are recomputed here, and the numbers must match.

    persistence   (last input frame x20) : rel-L2 0.138  TKE 1.000  MVPE 0.141
    input-time-mean (mean of 20 x20)     : rel-L2 0.134  TKE 1.000  MVPE 0.124

Tolerance +-0.002 (the documented figures are quoted to three decimals).

RESULT 2026-08-29: rel-L2 and TKE reproduce exactly; **MVPE does not**.  The reconstruction
pools all 36 probes into one rel-L2; the real kit takes rel-L2 per probe COLUMN (each of the 4
x-stations carries its own denominator over its 9 y-sites x 2 channels) and averages the four.
The per-column form hits both anchors to four decimals.  See report/setup_20260829_osc.md.

Because both anchor predictors are constant in time, `var_t` of the prediction is exactly 0 and
TKE is ||0 - kt||/||kt|| = 1 for any ddof -- that anchor cannot discriminate the
population/sample variance convention.  The ddof and aggregation variants are therefore
reported unconditionally in the DIAGNOSTIC section rather than only on failure, so a mismatch
anywhere is attributable.

    python verify_scoring_20260829.py
"""
from __future__ import annotations

import os

import numpy as np

import sps9
from sps9 import sc

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.002

# (name, rel-L2, TKE, MVPE) measured through the real kit on val_unseen (CLAUDE.md, "Measured
# local numbers (336 real windows, 8 held-out trajectories)").
ANCHORS = {
    "persistence": (0.138, 1.000, 0.141),
    "time-mean": (0.134, 1.000, 0.124),
}
# score_time curve points documented in CLAUDE.md (strategy item 3).
TIME_ANCHORS = [(1e-3, 96.4), (10e-3, 89.5), (100e-3, 73.0), (700e-3, 50.5)]

# The probe geometry, identical in scoring.py and in the three in-repo transcriptions.
PROBE_Y = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PROBE_X = [13, 21, 29, 37]


def persistence(X):
    return np.repeat(X[:, -1:], X.shape[1], axis=1)


def time_mean(X):
    return np.repeat(X.mean(axis=1, keepdims=True), X.shape[1], axis=1)


# ------------------------------------------------------------------ MVPE candidate forms

def _tmean(A, mc=2):
    return np.asarray(A, np.float64)[..., :mc].mean(axis=1)      # (N, H, W, mc)


def mvpe_pooled(P, Y):
    """What the reconstruction currently does: one rel-L2 over all 36 probes x 2 channels."""
    idx = lambda a: a[:, np.array(PROBE_Y)[:, None], np.array(PROBE_X)[None, :], :] \
        .reshape(len(a), -1)                                      # noqa: E731
    p, t = idx(_tmean(P)), idx(_tmean(Y))
    return np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1)


def mvpe_percol(P, Y):
    """The real kit: rel-L2 per probe COLUMN, then averaged over the four columns.

    Transcribed independently three times while the kit was still reachable --
    unet/train2.py:407-420 ("scoring.mvpe_rel_l2_per_sample, verbatim"),
    diag_mean_floor.py:36-43, diag_history.py:71-73.  Per-column denominators up-weight the far
    wake, where velocities are small; that is why the two forms differ by ~10%.
    """
    mp, mt = _tmean(P), _tmean(Y)
    cols = []
    for px in PROBE_X:
        pp = mp[:, PROBE_Y, px, :].reshape(len(mp), -1)
        tt = mt[:, PROBE_Y, px, :].reshape(len(mt), -1)
        cols.append(np.linalg.norm(pp - tt, axis=1) / np.linalg.norm(tt, axis=1).clip(1e-8))
    return np.stack(cols).mean(axis=0)


def mvpe_perrow(P, Y):
    """Row-wise analogue, for completeness -- rejected by the anchors."""
    mp, mt = _tmean(P), _tmean(Y)
    rows = []
    for py in PROBE_Y:
        pp = mp[:, py, PROBE_X, :].reshape(len(mp), -1)
        tt = mt[:, py, PROBE_X, :].reshape(len(mt), -1)
        rows.append(np.linalg.norm(pp - tt, axis=1) / np.linalg.norm(tt, axis=1).clip(1e-8))
    return np.stack(rows).mean(axis=0)


# ------------------------------------------------------------------ other variants

def rel_l2_pooled(P, Y, mc=2):
    p = np.asarray(P, np.float64)[..., :mc].ravel()
    t = np.asarray(Y, np.float64)[..., :mc].ravel()
    return float(np.linalg.norm(p - t) / np.linalg.norm(t))


def tke_ddof(P, Y, ddof, mc=2):
    ke = lambda x: 0.5 * np.var(np.asarray(x, np.float64)[..., :mc], axis=-4, ddof=ddof).sum(-1)
    kp, kt = ke(P).reshape(len(P), -1), ke(Y).reshape(len(Y), -1)
    per = np.linalg.norm(kp - kt, axis=1) / np.linalg.norm(kt, axis=1)
    return float(per.mean()), float(np.linalg.norm(kp - kt) / np.linalg.norm(kt))


# ------------------------------------------------------------------ main

def main() -> None:
    d = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y = d["x"], d["y"]
    print(f"val_unseen: x{X.shape} y{Y.shape}  "
          f"{len({str(i).split(':')[0] for i in d['ids']})} trajectories\n")

    preds = {"persistence": persistence(X), "time-mean": time_mean(X)}

    hdr = (f"{'predictor':14s} {'metric':16s} {'expected':>9s} {'got':>9s} {'|diff|':>8s}  "
           f"{'verdict':7s}")
    print("=" * len(hdr))
    print("ANCHOR TABLE  (tolerance +-%.3f)   'as-shipped' = the reconstruction on disk" % TOL)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    ok_core, ok_recon, ok_fix = True, True, True
    for name, P in preds.items():
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        m_recon = sc.mvpe_rel_l2(P, Y)
        m_fix = float(mvpe_percol(P, Y).mean())
        er, ek, em = ANCHORS[name]
        for lbl, g, e in (("rel-L2", r, er), ("TKE", k, ek)):
            o = abs(g - e) <= TOL
            ok_core &= o
            print(f"{name:14s} {lbl:16s} {e:9.3f} {g:9.4f} {abs(g - e):8.4f}  "
                  f"{'PASS' if o else 'FAIL':7s}")
        o = abs(m_recon - em) <= TOL
        ok_recon &= o
        print(f"{name:14s} {'MVPE as-shipped':16s} {em:9.3f} {m_recon:9.4f} "
              f"{abs(m_recon - em):8.4f}  {'PASS' if o else 'FAIL':7s}")
        o = abs(m_fix - em) <= TOL
        ok_fix &= o
        print(f"{name:14s} {'MVPE per-column':16s} {em:9.3f} {m_fix:9.4f} "
              f"{abs(m_fix - em):8.4f}  {'PASS' if o else 'FAIL':7s}")
    print("-" * len(hdr))

    # score_time curve.
    t_ok = True
    print("\nscore_time curve")
    for t, e in TIME_ANCHORS:
        g = sc.score_time(t)
        o = abs(g - e) <= 0.1
        t_ok &= o
        print(f"  {t * 1000:6.1f} ms  expected {e:6.2f}  got {g:6.2f}  {'PASS' if o else 'FAIL'}")

    # sps9 fast path vs the reconstruction's own aggregate_sps.  These are two independent
    # implementations at DIFFERENT precisions -- sps9.prep compares a float32 |p-t| against the
    # half-width, aggregate_sps compares float64 target against float64 pred +- half -- so a
    # handful of elements sitting within 1 ulp of the interval edge flip.  Judge relatively.
    print("\nsps9 fast path vs scoring.aggregate_sps (independent impls, float32 vs float64)")
    P = preds["persistence"][:64]
    Ys = Y[:64]
    s_ok = True
    for half in (0.004, 0.010, 0.025):
        A, a, scored, n = sps9.prep(P, Ys)
        fast_in = (A <= np.float32(half)) & scored
        P64, Y64 = P.astype(np.float64)[..., :2], Ys.astype(np.float64)[..., :2]
        kit_in = (Y64 >= P64 - half) & (Y64 <= P64 + half) & (Y64 != 0.0)
        flip = int(np.count_nonzero(fast_in != kit_in))
        f, r = sps9.self_check(P, Ys, np.float32(half))
        rel = abs(f - r) / max(r, 1e-12)
        o = rel < 1e-5
        s_ok &= o
        print(f"  half={half:.3f}  fast {f:11.8f}  kit {r:11.8f}  rel {rel:.2e}  "
              f"edge-flips {flip}/{n} ({flip / max(n, 1):.1e})  {'PASS' if o else 'FAIL'}")

    # SPS of the two predictors with their own optimal bands (context, not an anchor).
    print("\nSPS with per-element optimal band (in-sample on val_unseen; context only)")
    for name, Pp in preds.items():
        A, a, scored, n = sps9.prep(Pp, Y)
        band = sps9.optimal_band(A, a, scored)
        print(f"  {name:14s} default-band sps "
              f"{sc.score_sps(sc.aggregate_sps(Pp, Y, 2)[0]):6.2f}"
              f"   optimal-band sps {sps9.sps_score(A, a, scored, n, band):6.2f}")

    # ------------------------------------------------------------ diagnostics
    print("\n" + "=" * len(hdr))
    print("DIAGNOSTIC: convention variants (which one reproduces the anchor)")
    print("=" * len(hdr))
    for name, Pp in preds.items():
        e = ANCHORS[name]
        mark = lambda g, t: "  <== ANCHOR" if abs(g - t) <= TOL else ""    # noqa: E731
        r_mean = float(np.mean(sc.rel_l2_per_sample(Pp, Y, 2)))
        r_pool = rel_l2_pooled(Pp, Y)
        k0m, k0p = tke_ddof(Pp, Y, 0)
        k1m, k1p = tke_ddof(Pp, Y, 1)
        print(f"\n  {name}   (anchor rel {e[0]:.3f} / tke {e[1]:.3f} / mvpe {e[2]:.3f})")
        print(f"    rel-L2  mean-of-per-sample {r_mean:8.4f}{mark(r_mean, e[0])}")
        print(f"            pooled             {r_pool:8.4f}{mark(r_pool, e[0])}")
        print(f"    TKE     ddof=0 mean {k0m:8.4f}  pooled {k0p:8.4f}"
              f"  |  ddof=1 mean {k1m:8.4f}  pooled {k1p:8.4f}")
        print(f"            (both predictors are time-constant, so var_t(P)=0 and TKE=1 for "
              f"either ddof -- not discriminating)")
        for lbl, fn in (("per-COLUMN x4  ", mvpe_percol), ("per-ROW x9     ", mvpe_perrow)):
            v = float(fn(Pp, Y).mean())
            print(f"    MVPE    {lbl}    {v:8.4f}{mark(v, e[2])}")
        v = float(mvpe_pooled(Pp, Y).mean())
        print(f"    MVPE    pooled-36 (mean) {v:8.4f}{mark(v, e[2])}")
        v = rel_l2_pooled(_tmean(Pp)[:, PROBE_Y][:, :, PROBE_X],
                          _tmean(Y)[:, PROBE_Y][:, :, PROBE_X], mc=2)
        print(f"    MVPE    pooled-36 (all)  {v:8.4f}{mark(v, e[2])}")

    print("\n" + "=" * len(hdr))
    print(f"rel-L2 + TKE + score_time + sps9 cross-check : "
          f"{'PASS' if (ok_core and t_ok and s_ok) else 'FAIL'}")
    print(f"MVPE, reconstruction as shipped (pooled-36)  : "
          f"{'PASS' if ok_recon else 'FAIL  <-- BUG'}")
    print(f"MVPE, per-column form                        : "
          f"{'PASS  <-- this is the real kit' if ok_fix else 'FAIL'}")
    print("=" * len(hdr))
    if not ok_recon and ok_fix:
        print("""
ACTION REQUIRED (owner of scoring.py): replace _probe_means / mvpe_rel_l2_per_sample with

    def _probe_means(A, mc: int = 2):
        a = np.asarray(A, np.float64)[..., :mc].mean(axis=1)
        a = a[:, PROBE_Y[:, None], PROBE_X[None, :], :]           # (N, 9, 4, mc)
        return a.transpose(0, 2, 1, 3).reshape(len(A), len(PROBE_X), -1)   # (N, 4, 9*mc)

    def mvpe_rel_l2_per_sample(P, Y):
        p, t = _probe_means(P), _probe_means(Y)
        per_col = np.linalg.norm(p - t, axis=2) / np.linalg.norm(t, axis=2).clip(1e-8)
        return per_col.mean(axis=1)

`mvpe_rel_l2` (mean over samples) is unchanged.  All 24 call sites go through these two
functions, so the fix propagates everywhere -- including SPS, whose per-sample accuracy factor
a_i weights mvpe at 0.2.""")
    raise SystemExit(0 if (ok_core and t_ok and s_ok and ok_fix) else 1)


if __name__ == "__main__":
    main()
