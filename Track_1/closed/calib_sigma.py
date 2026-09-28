"""Post-hoc recalibration of the variance head against the window and history std.

    python calib_sigma.py --ckpt unet/wp2i_s5.pth

diag_history.py measured that a least-squares combination of the head's sigma, the input
window's own std and the causal trajectory-history std predicts the target's 20-frame std at
rel-L2 0.338 against the head's 0.358. This applies that at inference on an already-trained
checkpoint: the coefficients are three scalars per channel fitted on val_seen, the two extra
std fields cost a handful of elementwise ops, and no weight is touched.

Errors only. Timing is deliberately not reported here -- it must be measured on an idle GPU
(report_20260730.md section 7), and the calibration adds no measurable forward cost anyway.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from eval_wp import HERE, run, sc, val_history

T = 20


def compose_np(y: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """model.compose in numpy, physical units. sigma is (N,1,H,W,2)."""
    if alpha == 0.0:
        return y
    m = y[..., :2].mean(1, keepdims=True)
    f = y[..., :2] - m
    s = y[..., :2].std(1, keepdims=True)
    out = m + f * ((alpha * sigma + (1 - alpha) * s) / (s + 1e-6))
    return np.concatenate([out, np.zeros_like(out[..., :1])], -1).astype(np.float32)


def features(X: np.ndarray, S: np.ndarray, hist: np.ndarray) -> list[np.ndarray]:
    """The three std estimators, each (N,1,H,W,2) in physical units."""
    win = X[..., :2].std(1, keepdims=True)
    his = hist[:, 2:].transpose(0, 2, 3, 1)[:, None]
    return [S, win, his]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet/wp2i_s5.pth")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    args = ap.parse_args()

    data, H = {}, {}
    for tag in ("seen", "unseen"):
        d = np.load(os.path.join(HERE, f"val_{tag}.npz"))
        data[tag] = (d["x"], d["y"])
        H[tag] = val_history(d["x"], d["ids"])

    # alpha=0 gives the raw field and the head's sigma; compose is applied below in numpy.
    raw, sig = {}, {}
    for tag, (X, _) in data.items():
        raw[tag], sig[tag], _ = run(os.path.join(HERE, args.ckpt), X, 0.0, hist=H[tag])

    # Fit per channel on SEEN: sigma_cal = a*head + b*window + c*history + d.
    coef = []
    for ch in range(2):
        F = features(data["seen"][0], sig["seen"], H["seen"])
        A = np.stack([f[..., ch].ravel() for f in F] + [np.ones(F[0][..., ch].size)], 1)
        b = data["seen"][1][..., ch].std(1).ravel()
        coef.append(np.linalg.lstsq(A, b, rcond=None)[0])
    coef = np.stack(coef, -1)                       # (4, 2)
    print(f"fitted on val_seen  [head, window, history, const] per channel:")
    for ch, nm in enumerate("uv"):
        print(f"  {nm}: " + "  ".join(f"{c:+.4f}" for c in coef[:, ch]))

    def calibrate(tag: str) -> np.ndarray:
        F = features(data[tag][0], sig[tag], H[tag])
        out = sum(coef[i] * F[i] for i in range(3)) + coef[3]
        return np.maximum(out, 1e-6).astype(np.float32)

    print(f"\n{'':28s} {'sigma relL2':>12s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s}")
    for label, getter in (("head sigma (baseline)", lambda t: sig[t]),
                          ("calibrated sigma", calibrate)):
        Su = getter("unseen")
        Yu = data["unseen"][1]
        true = Yu[..., :2].std(1, keepdims=True)
        srel = float(np.linalg.norm(Su - true) / np.linalg.norm(true))
        for alpha in args.alphas:
            P = compose_np(raw["unseen"], Su, alpha)
            r = float(np.mean(sc.rel_l2_per_sample(P, Yu, 2)))
            k = float(np.mean(sc.tke_rel_l2_per_sample(P, Yu, 2)))
            mv = sc.mvpe_rel_l2(P, Yu)
            print(f"  {label:24s} a={alpha:.2f} {srel:12.4f} {r:7.4f} {k:7.4f} {mv:7.4f}"
                  f"   [scores {sc.score_error(r):.2f} {sc.score_error(k):.2f} "
                  f"{sc.score_error(mv):.2f}]")


if __name__ == "__main__":
    main()
