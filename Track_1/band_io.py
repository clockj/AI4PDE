"""Load an SPS band the way the PACKAGE reads it, and build the interval the package builds (2026-09-28).

    band = band_io.load_band("bands/k27_sigblend_lam0.5.npz")      # finds its sidecar, or raises
    half = band_io.half_width(band, sigma_phys)                     # (N, 20, H, W, 2) float32
    lo, up = band_io.interval(band, y0, P, sigma_phys)              # honours band.centre_alpha

WHY THIS EXISTS. Every instrument used to read `base / sigma_ref / p` from the npz and compute
`base * clip(sigma/sigma_ref, .3, 3)^p` around the prediction. That is the pre-09-21 package. The packages of
record (09-27) differ in two ways an npz reader cannot see:

  * the interval is centred on the alpha-0 field y0 (`centre_alpha` 0) and widened to contain P, and
  * the band is SIGBLEND: the npz stores base' and p' = 0.225, and the remaining factor (sigma + 1e-8)^(lam*q_c)
    lives ONLY in a sidecar json (baked as EBAND on Track 1, SIGPOW on Track 2). Scored without it, the band is
    about 3x too wide -- and nothing fails, the instrument just prints a much worse sps.

So `load_band` REFUSES a band whose p is not the static parent's 0.45 unless it finds (or is given) the sidecar,
and `half_width` / `interval` are the package's formulas, not approximations of them:
  static half   : `submission.py` host path / `_fno.predict_fno` band_gpu -- np.power(np.clip(sigma/ref, 0.3, 3.0),
                  float32(p)) in numpy float32, times base
  SIGBLEND      : `_fno.predict_fno` EBAND fast path -- half *= exp(log(sigma + 1e-8) * b0 + b5), float32
  interval      : `submission.py` / `_fno.predict_fno` -- centre = y0 + (c/ALPHA)(P - y0), lower = min(centre - half,
                  P), upper = max(centre + half, P); no centre (legacy band, no `centre_alpha` key) = P -/+ half.
The package evaluates the last two on the GPU in torch float32; numpy float32 agrees to an ulp or so on the
exp/log (well below any sps digit). `verify/sigblend.py` checks the package against `sigblend/bandlab.py`.

numpy only (plus opoint): the kit is not imported, so the build path can import this too without side effects.
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field

import numpy as np

import opoint

P_STATIC = opoint.BAND_P
P_TOL = 1e-6


@dataclass(frozen=True)
class Band:
    """An SPS band as the package evaluates it. Arrays are float32 and keep their stored channel count (3)."""
    path: str
    base: np.ndarray                   # (1, 20, H, W, C) per-element half-width (base' for SIGBLEND)
    sigma_ref: np.ndarray              # (1, 1, H, W, C)
    p: float                           # sigma-ratio exponent (0.45 static, 0.225 SIGBLEND lam 0.5)
    centre_alpha: float | None = None  # None = centred on the prediction (pre-09-21); 0 = on y0 (09-27)
    q: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))   # (2,) lam*q_c; zeros = static
    b: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))   # (2,) EBAND intercept (0 for SIGBLEND)
    sidecar: str | None = None
    kind: str = "static"               # "static" | "sigblend"

    def describe(self) -> str:
        c = "pred" if self.centre_alpha is None else f"alpha {self.centre_alpha:g}"
        s = f"{os.path.basename(self.path)}: {self.kind}, p {self.p:.4f}, centre {c}"
        if self.kind == "sigblend":
            s += f", q (u, v) = ({self.q[0]:.4f}, {self.q[1]:.4f}) from {os.path.basename(self.sidecar)}"
        return s


def _find_sidecar(path: str) -> str | None:
    stem = os.path.splitext(path)[0] + ".json"
    if os.path.exists(stem):
        return stem
    rel = opoint.SIDECARS.get(os.path.basename(path))
    if rel is not None and os.path.exists(opoint.path(rel)):
        return opoint.path(rel)
    return None


def _read_sidecar(js: str, p_npz: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (q, b) per channel, from the EBAND literal Track 1 bakes; cross-checked against T2_sigpow and p'."""
    d = json.load(open(js))
    if "EBAND" not in d:
        raise ValueError(f"{js}: not a SIGBLEND sidecar (no EBAND key)")
    beta, mode, clip = ast.literal_eval(d["EBAND"])
    beta = np.asarray(beta, np.float32)
    if mode != "mult" or clip is not None or beta.shape != (2, 6) or np.any(beta[:, 1:5] != 0):
        # the general EBAND (gradient / time / input-std features) needs the input window and C0; no instrument
        # implements it, so refuse rather than score a different band
        raise NotImplementedError(f"{js}: EBAND {d['EBAND']} is not a sigma-only 'mult' factor (SIGBLEND); "
                                  "band_io implements only that one")
    q, b = beta[:, 0].copy(), beta[:, 5].copy()
    if "T2_sigpow" in d and not np.allclose(np.float32(d["T2_sigpow"]), q, rtol=0, atol=1e-6):
        raise ValueError(f"{js}: EBAND exponents {q.tolist()} disagree with T2_sigpow {d['T2_sigpow']}")
    if "p_new" in d and abs(float(d["p_new"]) - p_npz) > P_TOL:
        raise ValueError(f"{js}: sidecar is for p' = {d['p_new']}, the npz stores p = {p_npz} -- "
                         "the npz and its sidecar are not a pair")
    return q, b


def load_band(path: str, sidecar: str | None = None) -> Band:
    """Read a band npz the way the package does.

    sidecar: None = look for `<npz stem>.json` next to the npz, then `opoint.SIDECARS`; a path = use that json;
    "static" = assert the npz is a plain static band even though its p is not 0.45 (an old --band-p sweep).
    A band whose p differs from 0.45 by more than 1e-6 and has no sidecar RAISES: that is the SIGBLEND trap.
    """
    path = opoint.path(path)
    z = np.load(path)
    for k in ("base", "sigma_ref", "p"):
        if k not in z.files:
            raise ValueError(f"{path}: missing key {k!r} (a legacy band.npy is not supported here)")
    if "err_mode" in z.files and str(z["err_mode"]) != "none":
        raise NotImplementedError(f"{path}: an --err-head band (err_mode {z['err_mode']}) needs the heads' error "
                                  "field; band_io does not implement it")
    for k, v in (("clip_lo", opoint.CLIP_LO), ("clip_hi", opoint.CLIP_HI)):
        # the package hard-codes the clip and ignores these keys; a band fitted at another clip would be mis-scored
        if k in z.files and abs(float(z[k]) - v) > 1e-6:
            raise ValueError(f"{path}: {k} = {float(z[k])}, but the package clips at {v}")
    p = float(z["p"])
    ca = float(z["centre_alpha"]) if "centre_alpha" in z.files else None
    kw = dict(path=path, base=z["base"].astype(np.float32), sigma_ref=z["sigma_ref"].astype(np.float32), p=p,
              centre_alpha=ca)
    if sidecar == "static":
        return Band(**kw)
    js = sidecar if sidecar is not None else (_find_sidecar(path) if abs(p - P_STATIC) > P_TOL else None)
    if js is None:
        if abs(p - P_STATIC) > P_TOL:
            raise ValueError(
                f"{path}: p = {p:.6f}, not the static band's {P_STATIC}, and no sidecar json was found "
                f"(looked for {os.path.splitext(os.path.basename(path))[0]}.json beside it and in opoint.SIDECARS). "
                "A SIGBLEND npz (p' = (1-lam)*0.45) stores only base' and p'; the factor (sigma + 1e-8)^(lam*q_c) "
                "is in its sidecar (EBAND / SIGPOW). Scored without it the band is ~3x TOO WIDE and sps comes out "
                "several points low with no other symptom. Pass sidecar=<json>, or sidecar='static' if this really "
                "is a static band fitted at another --band-p.")
        return Band(**kw)
    q, b = _read_sidecar(opoint.path(js), p)
    return Band(**kw, q=q, b=b, sidecar=opoint.path(js), kind="sigblend")


def joint_gain_for(band: Band | None, mode: bool | str | None = None) -> bool:
    """Resolve an instrument's --joint-gain / --no-joint-gain (None = neither given = auto) against its band.

    auto = opoint.JOINT_GAIN when the band is centred on y0 (or no band is scored), else False. submission.py
    asserts centre_alpha 0 before it turns JOINT_GAIN on (a centre between y0 and P relies on compose being linear
    in alpha, which JOINT_GAIN breaks), so an old pred-centred band (every band before r35) is scored with the
    per-channel compose it was fitted under. "on" with such a band raises: no package can ship that pair.
    """
    if mode is None or mode == "auto":
        return bool(opoint.JOINT_GAIN) and (band is None or band.centre_alpha == 0.0)
    if isinstance(mode, str):
        mode = mode == "on"
    if mode and band is not None and band.centre_alpha != 0.0:
        raise ValueError(f"--joint-gain on with {os.path.basename(band.path)} (centre_alpha "
                         f"{band.centre_alpha}): the package asserts centre_alpha 0 under JOINT_GAIN")
    return bool(mode)


def half_width(band: Band, sigma_phys: np.ndarray) -> np.ndarray:
    """The package's half-width on the measured channels: (N, 20, H, W, 2) float32.

    sigma_phys: the members' mean sigma head in PHYSICAL units, (N, 1, H, W, >=2) (predict_fno's sigma_out).
    """
    s = np.asarray(sigma_phys, np.float32)[..., :2]
    r = np.power(np.clip(s / band.sigma_ref[..., :2], opoint.CLIP_LO, opoint.CLIP_HI), np.float32(band.p))
    h = band.base[..., :2] * r
    if band.kind == "sigblend":
        # _fno.predict_fno SIGBLEND fast path: exp(log(sigma + 1e-8) * b0 + b5) on the (N,1,H,W,2) sigma field
        fac = np.exp(np.log(s + np.float32(opoint.SIGMA_EPS)) * band.q + band.b).astype(np.float32)
        h = h * fac
    return h.astype(np.float32, copy=False)


def centre(band: Band, y0: np.ndarray | None, P: np.ndarray, alpha: float | None = None,
           joint_gain: bool = False) -> np.ndarray:
    """The interval centre the package uses: P (no centre key), y0 (centre_alpha 0), or y0 + (c/alpha)(P - y0)."""
    c = band.centre_alpha
    if c is None or (alpha is not None and c == alpha):
        return P
    if y0 is None:
        raise ValueError(f"{os.path.basename(band.path)} is centred at alpha {c:g}: pass y0 (predict_fno's centre_out)")
    if c == 0.0:
        return y0
    if alpha is None:
        raise ValueError(f"centre_alpha {c:g} needs the compose alpha to place the centre")
    if joint_gain:
        # compose is no longer linear in alpha under JOINT_GAIN, which is why submission.py asserts centre 0
        raise ValueError(f"centre_alpha {c:g} is undefined under JOINT_GAIN (the package asserts centre_alpha 0)")
    return (y0 + np.float32(c / alpha) * (P - y0)).astype(np.float32)


def interval(band: Band, y0: np.ndarray | None, P: np.ndarray, sigma_phys: np.ndarray,
             alpha: float | None = None, joint_gain: bool = False, half: np.ndarray | None = None):
    """(lower, upper) on the measured channels, (N, 20, H, W, 2) float32, exactly as the package builds them.

    y0: the pre-compose member mean (physical; predict_fno's centre_out), needed when band.centre_alpha is set.
    P : the scored prediction. Either may carry a third (pressure) channel; only [..., :2] is used.
    With a centre away from P the interval is widened to contain P (min/max), as submission.py does.
    """
    P2 = np.asarray(P, np.float32)[..., :2]
    y2 = None if y0 is None else np.asarray(y0, np.float32)[..., :2]
    h = half_width(band, sigma_phys) if half is None else half
    c = centre(band, y2, P2, alpha, joint_gain)
    if c is P2:
        return (P2 - h).astype(np.float32), (P2 + h).astype(np.float32)
    return np.minimum(c - h, P2).astype(np.float32), np.maximum(c + h, P2).astype(np.float32)


def stretch_target(band: Band, Y: np.ndarray, y0: np.ndarray | None, P: np.ndarray, f: float,
                   alpha: float | None = None, joint_gain: bool = False) -> np.ndarray:
    """The board-tail stretch about the INTERVAL CENTRE: y_f = c + f (Y - c) on the measured channels.

    For a pred-centred band this is exactly the old `f * |P - Y|` convention of predict_board / calib_band13;
    for a y0-centred band it is `sigblend/bandlab.py`'s (target moved about y0). f = 1 returns Y's u, v.
    """
    Y2 = np.asarray(Y, np.float32)[..., :2]
    if f == 1.0:
        return Y2
    P2 = np.asarray(P, np.float32)[..., :2]
    c = centre(band, None if y0 is None else np.asarray(y0, np.float32)[..., :2], P2, alpha, joint_gain)
    return (c + np.float32(f) * (Y2 - c)).astype(np.float32)
