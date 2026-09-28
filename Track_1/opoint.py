"""The operating point of the packages of record of 2026-09-27 -- ONE place for the instruments' defaults.

R37 (Track 1, 81.671149) and T2R24 (Track 2, 82.055553) ship the same compose and the same band:

    compose   ALPHA 0.80, JOINT_GAIN (one amplitude gain per pixel for u and v, channels weighted by their
              physical variance; _fno.JOINT_GAIN)
    interval  centred on the alpha-0 field y0 (the pre-compose member mean) and widened to contain the
              prediction P:  lower = min(y0 - half, P),  upper = max(y0 + half, P)
    band      SIGBLEND: half = base' * clip(sigma/sigma_ref, 0.3, 3)^p' * (sigma + 1e-8)^(lam*q_c)
              base' and p' = 0.225 are in FINAL_BAND; the exponents lam*q_c live ONLY in FINAL_SIDECAR.
              Its static parent PARENT_BAND (p = BAND_P) was fitted with
              `fit_band.py --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --inflate 1.39`.

`Track_1/submit_fno/build_final.sh` is the AUTHORITY for what is BAKED into the archives; this module only
repeats those constants so that every local instrument defaults to them (2026-09-28). If the two ever disagree,
build_final.sh wins and this file is the bug. Nothing on the build path imports it: `fit_band.py`'s own defaults
are load-bearing for build_round13.sh and stay where they are.

Dependency-free on purpose (no numpy, no torch, no kit): an instrument can import it before anything else.
Paths are relative to `Track_1/`; `path()` resolves them.
"""
from __future__ import annotations

import os

# --- compose -------------------------------------------------------------------------------------------------
ALPHA = 0.80                  # compose alpha of R37 / T2R24 (build_round13.sh 0.80)
JOINT_GAIN = True             # RPDE_SHIP_JOINT_GAIN=True; valid only with CENTRE_ALPHA == 0 (submission.py asserts it)

# --- the SPS interval ----------------------------------------------------------------------------------------
CENTRE_ALPHA = 0.0            # centre on y0 (band npz key `centre_alpha`); None = the pre-09-21 pred-centred band
BAND_P = 0.45                 # sigma-ratio exponent of the STATIC parent band (fit_band.py --band-p)
INFLATE = 1.39                # error stretch the static band is FITTED at (fit_band.py --inflate; board tail, round 7)
F_TAIL = 1.36                 # board per-element tail at which a band is SCORED (predict_board; calib_board11, R10)
FILL_UNSCORED = True          # fit_band.py --fill-unscored
CLIP_LO, CLIP_HI = 0.3, 3.0   # sigma-ratio clip, hard-coded in submission.py / _fno.predict_fno (not read from the npz)
SIGMA_EPS = 1e-8              # the EBAND/SIGPOW feature is log(sigma + 1e-8), physical sigma

# --- the band files (relative to Track_1/) -------------------------------------------------------------------
PARENT_BAND = "bands/r38_a080_c0_fill.npz"               # static, p = 0.45, centre_alpha 0
FINAL_BAND = "bands/k27_sigblend_lam0.5.npz"             # SIGBLEND, p' = 0.225 -- ~3x too wide WITHOUT its sidecar
FINAL_SIDECAR = "sigblend/band_k0_sigblend_lam0.5.json"  # lam, q, the EBAND literal (T1) and T2_sigpow (T2)
# npz basename -> its sidecar, for SIGBLEND bands whose json does not sit next to them as `<stem>.json`
SIDECARS = {os.path.basename(FINAL_BAND): FINAL_SIDECAR}

T1 = os.path.dirname(os.path.abspath(__file__))


def path(rel: str) -> str:
    """Resolve a path of this module (relative to Track_1/) to an absolute one."""
    return rel if os.path.isabs(rel) else os.path.join(T1, rel)
