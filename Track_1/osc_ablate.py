"""Is the temporal branch doing the work, or is the head a training-time regularizer?

Handles both temporal branches -- `--osc-modes` (`head_osc`) and `--tbasis` (`head_tb`) --
since the question and the ablation are identical for either.

    python osc_ablate.py --ckpt unet/fno_ozdo.pth unet/fno_ozdo_b.pth        # CPU is fine

§22 measured that `ozdo` moves Γ² (+1.66 %) and the fluctuation part of the error (−1.4 %),
replicated over four seeds — the first change in the project to move either. §23 measured what
frequencies the head picks. Neither says WHICH of two very different mechanisms is responsible,
and the follow-up differs completely between them:

* **The term.** `field + osc` is a better forecast than `field`. Then the follow-up is about
  the term's form — more modes, a free learned temporal basis, a different band — and a
  `--tbasis` arm is the right next experiment.
* **The head.** Carrying an oscillatory branch during training reshapes the trunk, and the term
  itself contributes little at inference. Then it is a **regularizer**, it belongs next to
  `--dropout` rather than next to the fluctuation work, and `--tbasis` tests nothing.

This is the same distinction §13.4 and §18 got wrong in opposite directions (the head was first
called "a mean-head in disguise", then that was withdrawn), so it is worth measuring rather
than arguing. The measurement is an inference-time ablation: run the identical weights with the
osc term zeroed. `head_osc`'s output conv is zero-initialized, so "no term" is exactly the
model's own initialization for that branch.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import _fno  # noqa: E402
from eval_v9 import infer, sc  # noqa: E402


def split(P, Y):
    """(rel-L2, time-mean part, fluctuation part) -- orthogonal, so they add in quadrature."""
    n = np.sqrt(((P - Y) ** 2).reshape(len(Y), -1).sum(1))
    d = np.sqrt((Y ** 2).reshape(len(Y), -1).sum(1))
    dm = (P - Y).mean(1, keepdims=True)
    m = np.sqrt(((dm ** 2) * P.shape[1]).reshape(len(Y), -1).sum(1))
    f = np.sqrt((((P - Y) - dm) ** 2).reshape(len(Y), -1).sum(1))
    return (float(np.mean(n / d)), float(np.mean(m / d)), float(np.mean(f / d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--set", default="val_unseen")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(os.path.join(HERE, f"{args.set}.npz"))
    X, Y = d["x"].astype(np.float32), d["y"]
    print(f"{args.set}: {len(X)} windows, alpha {args.alpha}\n")
    hdr = f"{'checkpoint':22s} {'branch':>10s} {'rel-L2':>8s} {'mean':>8s} {'fluc':>8s} {'TKE':>8s}"
    print(hdr); print("-" * len(hdr))

    for c in args.ckpt:
        p = c if os.path.isabs(c) else os.path.join(HERE, c)
        probe = _fno.load_any(p, dev)
        branch = ("head_osc" if getattr(probe, "head_osc", None) is not None
                  else "head_tb" if getattr(probe, "head_tb", None) is not None else None)
        if branch is None:
            print(f"{os.path.basename(c):22s} no temporal branch, skipped"); continue
        for tag in ("with " + branch[5:], branch[5:] + " OFF"):
            m = _fno.load_any(p, dev)
            if tag.endswith("OFF"):
                # Zero the branch's OUTPUT conv, which is exactly how it was initialized.
                getattr(m, branch)[2].weight.data.zero_()
                getattr(m, branch)[2].bias.data.zero_()
            P = infer([m], X, dev, args.alpha)
            r, mn, fl = split(P[..., :2], Y[..., :2])
            k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
            print(f"{os.path.basename(c):22s} {tag:>10s} {r:8.5f} {mn:8.5f} {fl:8.5f} {k:8.5f}")
    print("\nA large gap => the TERM carries the gain (pursue its form, e.g. a free temporal\n"
          "basis). A small gap against §22's baselines (rel-L2 0.08513, fluc 0.0707, TKE\n"
          "0.49123) => the HEAD regularized the trunk and belongs with --dropout instead.")


if __name__ == "__main__":
    main()
