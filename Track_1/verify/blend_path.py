"""Gate the GATED MEAN-BLEND before it ships. Nothing local exercises it, which is the problem.

    python verify/blend_path.py --dir submit_fno/_package8_blend

`submission.BLEND_BETA` only fires for windows whose mean streamwise speed falls OUTSIDE the
training range [0.03942, 0.29207] m/s.  Every local validation window comes from `train_real`,
so **the gate never fires locally and `validate_strict.py` would score the package as if the
feature did not exist**.  A green strict gate is therefore not evidence that this works, and
shipping on it would be the "silence is not success" failure this project has hit before.

Five checks, and an un-runnable one is a FAILURE, not a skip:

  A  gate is exactly 0 on all 336 local windows  -> in distribution the package is
     BIT-IDENTICAL to the same package with the blend off. This is the safety claim.
  B  on a synthetic out-of-range window the gate fires at the value the formula predicts.
  C  the blend moves the prediction's time-mean by exactly beta*gate*(mean_in - mean_pred),
     and moves nothing else: the fluctuation about the time-mean is untouched to float32.
  D  the pressure channel stays exactly zero and the band still brackets the BLENDED
     prediction (lower <= pred <= upper) -- the blend runs before the band for this reason.
  E  order-independence is not made WORSE. The forward pass itself is not bit-exact under a
     permutation -- batching changes which windows share a cuFFT/matmul reduction -- and out
     of distribution that jitter is larger than the 5.6e-08 `validate_strict` sees in it. So
     the test is the blend's drift against the SAME package's drift with the blend off, on
     the same input, not against zero.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np


def load_submission(d):
    spec = importlib.util.spec_from_file_location("_sub_blend", os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_sub_blend"] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="an EXTRACTED package directory")
    ap.add_argument("--beta", type=float, default=0.15)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub = load_submission(args.dir)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    print(f"package {args.dir}   BLEND_BETA={sub.BLEND_BETA}  BLEND_REF={sub.BLEND_REF}  "
          f"TRAIN_SPEED={sub.TRAIN_SPEED}")

    # --- A: the gate on real local windows -------------------------------------------------
    Xs = []
    for name in ("val_seen.npz", "val_unseen.npz"):
        p = os.path.join(here, name)
        if not os.path.exists(p):
            check(f"A {name} present", False, "cannot test the safety claim without it")
            return
        Xs.append(np.load(p)["x"])
    X = np.concatenate(Xs).astype(np.float32)
    sub.BLEND_BETA = args.beta
    g = sub._blend_gate(X)
    check("A gate is 0 on every local window", float(g.max()) == 0.0,
          f"max gate {float(g.max()):.6f} over {len(X)} windows; "
          f"speeds {float(X[..., 0].mean((1,2,3)).min()):.5f}-"
          f"{float(X[..., 0].mean((1,2,3)).max()):.5f} m/s")

    n = 24
    p_off = None
    for beta in (0.0, args.beta):
        sub.BLEND_BETA = beta
        out = sub.predict(X[:n])
        if beta == 0.0:
            p_off = out["prediction"].copy()
        else:
            d = float(np.abs(out["prediction"] - p_off).max())
            check("A in distribution the blend is bit-identical to blend-off", d == 0.0,
                  f"max |diff| {d:.3e}")

    # --- B, C: a synthetic OUT-OF-RANGE window ---------------------------------------------
    lo, hi = sub.TRAIN_SPEED
    tgt = 0.03265                                    # Re 2968 at nu/c = 1.1e-5
    Xo = (X[:n] * np.float32(tgt / float(X[:n, ..., 0].mean()))).astype(np.float32)
    Xo[..., 2] = 0.0
    g = sub._blend_gate(Xo)
    want = min(max(lo - float(Xo[0, ..., 0].mean()), 0.0) / ((hi - lo) * sub.BLEND_REF), 1.0)
    check("B gate fires at the predicted strength out of range",
          abs(float(g[0]) - want) < 1e-5, f"gate {float(g[0]):.5f} vs formula {want:.5f}")

    sub.BLEND_BETA = 0.0
    off = sub.predict(Xo)["prediction"].copy()
    sub.BLEND_BETA = args.beta
    res = sub.predict(Xo)
    on = res["prediction"]

    b = (np.float32(args.beta) * g).reshape(-1, 1, 1, 1, 1)
    want_shift = b * (Xo[..., :2].mean(1, keepdims=True) - off[..., :2].mean(1, keepdims=True))
    got_shift = on[..., :2].mean(1, keepdims=True) - off[..., :2].mean(1, keepdims=True)
    check("C the time-mean moves by exactly beta*gate*(mean_in - mean_pred)",
          float(np.abs(got_shift - want_shift).max()) < 2e-6,
          f"max |diff| {float(np.abs(got_shift - want_shift).max()):.3e}")
    fl_off = off[..., :2] - off[..., :2].mean(1, keepdims=True)
    fl_on = on[..., :2] - on[..., :2].mean(1, keepdims=True)
    check("C the fluctuation about the time-mean is untouched",
          float(np.abs(fl_on - fl_off).max()) < 2e-6,
          f"max |diff| {float(np.abs(fl_on - fl_off).max()):.3e}")
    check("C the blend actually changed something",
          float(np.abs(on - off).max()) > 1e-5,
          f"max |change| {float(np.abs(on - off).max()):.3e}")

    # --- D: contract still holds on the blended prediction ---------------------------------
    check("D pressure channel exactly zero", float(np.abs(on[..., 2]).max()) == 0.0)
    check("D lower <= prediction <= upper",
          bool((res["lower"] <= on).all() and (on <= res["upper"]).all()))
    check("D everything finite", bool(np.isfinite(on).all()
                                      and np.isfinite(res["lower"]).all()
                                      and np.isfinite(res["upper"]).all()))

    # --- E: order-independence -------------------------------------------------------------
    perm = np.random.default_rng(0).permutation(n)
    onp = sub.predict(Xo[perm])["prediction"]
    d_on = float(np.abs(onp - on[perm]).max())
    sub.BLEND_BETA = 0.0
    d_off = float(np.abs(sub.predict(Xo[perm])["prediction"] - off[perm]).max())
    sub.BLEND_BETA = args.beta
    rms = float(np.sqrt((on.astype(np.float64) ** 2).mean()))
    # The bound is RELATIVE TO THE BLEND-OFF DRIFT, and only that. An absolute bound would be
    # measuring the forward pass: this input is scaled ~7x out of distribution, where float32
    # accumulation drifts 3.6e-04 of rms under a permutation against the 5.6e-08 that
    # `validate_strict` sees on real windows. The question here is only whether the blend adds
    # to it, and it does not -- the blend is a per-window affine shift of the time-mean.
    check("E the blend does not worsen order-independence",
          d_on <= max(1.5 * d_off, 1e-7),
          f"blend-on drift {d_on:.3e} vs blend-off {d_off:.3e} ({d_on/max(d_off,1e-30):.3f}x) "
          f"on the same permutation; {d_on/rms:.2e} of field rms")

    print("\n" + ("PASS" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
