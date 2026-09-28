# Design K0 — SIGBLEND band (commander, 2026-09-27 17:10 EDT)

## 1. Why (evidence from the live board, `leaderboard_2040Z.md`)
21 teams are above us on Track 1; all but one have containment C in 0.609-0.624 at the same accuracy (abar 0.713-0.722);
ours is 0.5994. Even teams ranked 23-31 have C 0.607-0.617. So the deficit is structural in OUR band, not in the models.

## 2. What is wrong with the shipped band (held-out twins, val_unseen, real shipped npz)
`half = base * clip(sigma/sigma_ref, 0.3, 3)^0.45`: `base` has 81,920 numbers fitted on the twins' errors on three AoA 15-20
trajectories of `val_seen` (trajectories the twins TRAINED on), x1.39; `sigma_ref` is the per-pixel mean sigma of those same
windows. For a window whose sigma level is far below that reference (low Re, low AoA) the clip floor 0.3 binds and the
band is far too wide: coverage on AoA <= 10 is 94.7 % against 87.6 % on AoA 15-20 (the optimum is ~88 %).
A band that is a pure function of the window's OWN sigma, `half = kappa_{t,c} * sigma^q_c` (q = 0.485 / 0.491, fitted on
honest held-out errors, 42 numbers), is as good as the per-element band on AoA 15-20 (-0.3...-0.5 C x100) and much better on
AoA <= 10 (+0.9) and at low Re (+3.4 on 3750_0).

## 3. The trick: geometric blend, evaluated by the EXISTING package code
    half = base' * clip(sigma/sigma_ref, .3, 3)^p' * exp(lam * q_c * log(sigma + 1e-8))
    base' = base^(1-lam) * kappa_{t,c}^lam        p' = (1-lam) * 0.45            -> a band .npz, nothing else
    T1: EBAND = (((lam*q_u,0,0,0,0,0), (lam*q_v,0,0,0,0,0)), "mult", None)      -> build knob RPDE_SHIP_EBAND (exists)
    T2: needs a 3-line knob (SIGPOW): half *= (sigma_phys + 1e-8) ** (lam*q_c)   -> executor E0b
The prediction is untouched; the centre stays y0; the "contain P" rule is unchanged. Only lower/upper move.

## 4. Held-out read (numpy, exactly the package formula; leave-one-trajectory-out over the 8 val_unseen trajectories for
##    the fitted power law; paired against the real shipped band; C x100)
| lam | stretch f | AoA 15-20 (3) | AoA <= 10 (5) | all 8 |
|---|---|---|---|---|
| 0.3 | 1.00 | +0.147 (3/3) | +0.607 (5/5) | +0.434 (8/8) |
| 0.3 | 1.29 | +0.083 (3/3) | +0.522 (5/5) | +0.357 (8/8) |
| 0.3 | 1.39 | +0.060 (3/3) | +0.480 (5/5) | +0.322 (8/8) |
| 0.5 | 1.00 | +0.146 (3/3) | +0.915 (5/5) | +0.626 (8/8) |
| 0.5 | 1.29 | +0.029 (2/3) | +0.754 (5/5) | +0.482 (7/8) |
| 0.5 | 1.39 | -0.010 (2/3) | +0.678 (5/5) | +0.420 (7/8) |
| 0.7 | 1.29 | -0.112 (0/3) | +0.890 (4/5) | +0.515 (4/8) |
| 1.0 | 1.29 | -0.483 (0/3) | +0.911 (4/5) | +0.388 (4/8) |
Per trajectory, lam 0.5, f 1.29: 3750_0 +2.09, 26700_0 +0.50, 16500_5 +0.68, 6300_10 +0.43, 13950_10 +0.07, 21600_15 +0.07,
8850_20 +0.07, 11400_20 -0.05.  Log: `$K/logs/k0_make_band.log`.  Code: `$K/wt_final/Track_1/probe_k0/make_k0_band.py`.

## 5. Prediction (T1, vs the same package with the shipped band)
Only `sps` moves. dC x100: +0.03 (test set like AoA 15-20) ... +0.48 (like the full split) ... more if low-Re heavy.
lam 0.5: **sps +0.02 ... +0.35, final +0.005 ... +0.09; point estimate +0.04.**  lam 0.3: final +0.015 ... +0.065.
History warns that band changes over-read locally; this one is a robustness change (never worse than shipped on any of
the 8 held-out trajectories at lam 0.3), not a tightening. Latency: EBAND computes 5 features, 4 of them unused.

## 6. Constants (fitted on all 8 held-out trajectories, inflate 1.29)
`$K/pkg/k0/band_k0_sigblend_lam{0.3,0.5,0.7}.npz` (+ `.json` with q, log kappa, the EBAND literal and the T2 exponents).
lam 0.5: `RPDE_SHIP_EBAND='(((0.2423255890607834, 0.0, 0.0, 0.0, 0.0, 0.0), (0.24535682797431946, 0.0, 0.0, 0.0, 0.0, 0.0)), "mult", None)'`
lam 0.3: `RPDE_SHIP_EBAND='(((0.14539535343647003, 0.0, 0.0, 0.0, 0.0, 0.0), (0.1472140997648239, 0.0, 0.0, 0.0, 0.0, 0.0)), "mult", None)'`
sigma_ref in the npz is unchanged (all-data members' scale); the power law was fitted against the twins' sigma.

## 7. Traps
* A SIGBLEND npz WITHOUT the sigma-power factor is ~3x too wide: never put it in a package whose code does not apply the
  factor (T1 without EBAND, T2 without SIGPOW). One build makes both tracks -> check each zip.
* T1: `EBAND` raises RuntimeError unless the GPU-band path runs; the small-graph path (N < SMALL_N) bypasses it
  (`submission.py` ~723-728) -> build with SMALL_GRAPH off, and test N = 1, 2, 13, 14, 99, 100, 336.
* Reproducibility: the band fit is one more script (`make_k0_band.py`) after the twins exist; no training involved.
