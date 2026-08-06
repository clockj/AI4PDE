"""Choose an SPS band that is right for the error scale the model meets on the BOARD.

    python inflate_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

The band is the only part of the submission fitted directly against an error distribution, and
that makes it the only part that breaks when the distribution shifts.  It has:

    board rel-L2 err 0.1179  vs  local val_unseen 0.0807   ->  1.46x

on every submission so far, and a band sized for 0.0807 is simply too narrow at 0.1179:
elements fall outside, and an element outside contributes exactly zero no matter how good the
prediction was.  That is most of why board SPS reads 36.9 against 49.0 local.

This script does not guess the inflation.  It scales the local error field by a range of
factors, re-solves the exact band at each one, and scores every band against every scale.  The
output is a matrix: rows are the scale a band was BUILT for, columns the scale it is MET with.
Reading down a column gives the cost of having built for the wrong scale, and the diagonal is
the unattainable matched-band optimum.  A band is chosen for robustness across the plausible
range rather than for the best entry of any single column, because the true factor is a
one-sample estimate.

Scaling |E| is a model of the shift, not the shift itself: it assumes the board's errors are
the local ones stretched, i.e. the same spatial and temporal error structure with a larger
amplitude.  That is consistent with what the boards show (every subscore degrades, none
inverts) but it would miss a shift that redistributes error across the field.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def make_inflater(P, Y):
    """f -> the prep tuple with every error magnified by f, accuracy factors included.

    a_i is re-derived from the scaled metrics rather than rescaled: its dependence on a metric
    is 0.5/(0.5+m), which is concave, so a metric that doubles does not halve its term.
    Holding a_i at its local value would credit a board-scale error field with local accuracy
    and overstate every entry of the table by the same optimistic constant.

    rel-L2 and MVPE are norms of the error and scale by f exactly.  TKE is a norm of a
    quadratic in the error and does not; scaling it by f is the one approximation here, and it
    is conservative in the direction that matters least (a_i enters every band identically, so
    it shifts the whole table rather than reordering it).
    """
    A0, _, scored, n = prep(P, Y)
    dm, tk, mv = sps9.sample_metrics(P, Y)

    def at(f):
        a = sps9.weights_from_metrics(f * dm, f * tk, f * mv)
        return np.float32(f) * A0, a, scored, n
    return at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--factors", type=float, nargs="+",
                    default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    Ps, Ys = infer(models, ds["x"], dev, args.alpha), ds["y"]
    Pu, Yu = infer(models, du["x"], dev, args.alpha), du["y"]
    S_at, U_at = make_inflater(Ps, Ys), make_inflater(Pu, Yu)

    # One band per build-scale, each the exact per-element optimum on val_seen at that scale.
    bands = {f: optimal_band(*S_at(f)[:3]) for f in args.factors}

    print("SPS on val_unseen. rows: scale the band was BUILT for. cols: scale it is MET with.")
    print(f"{'built':>7s} {'median h':>9s} | " +
          " ".join(f"{f:>6.1f}" for f in args.factors) + f" | {'worst':>6s} {'mean':>6s}")
    print("-" * (19 + 7 * len(args.factors) + 16))
    rows = {}
    for fb, band in bands.items():
        vals = [sps_score(*U_at(fm), band) for fm in args.factors]
        rows[fb] = vals
        print(f"{fb:7.1f} {np.median(band):9.5f} | " + " ".join(f"{v:6.2f}" for v in vals)
              + f" | {min(vals):6.2f} {np.mean(vals):6.2f}")

    # A constant band is the robustness baseline worth beating: it has no error structure to
    # get wrong, so it degrades gracefully by construction.
    print()
    best_c, best_worst = None, -1.0
    for c in np.linspace(0.003, 0.020, 35):
        vals = [sps_score(*U_at(fm), np.float32(c)) for fm in args.factors]
        if min(vals) > best_worst:
            best_c, best_worst = c, min(vals)
    vals = [sps_score(*U_at(fm), np.float32(best_c)) for fm in args.factors]
    print(f"{'const':>7s} {best_c:9.5f} | " + " ".join(f"{v:6.2f}" for v in vals)
          + f" | {min(vals):6.2f} {np.mean(vals):6.2f}")

    diag = [rows[f][i] for i, f in enumerate(args.factors)]
    print("\nmatched-band diagonal (unattainable): "
          + " ".join(f"{f:.1f}->{v:.2f}" for f, v in zip(args.factors, diag)))
    print(f"board rel-L2 0.1179 / local 0.0807 = "
          f"{0.1179/float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2))):.2f}x")


if __name__ == "__main__":
    main()
