"""Does an --int8 archive predict the same thing as its fp32 source?  Archive against source.

    python verify/quant_identity.py --ckpt unet/fno_all_w96L8do45_s1.pth ...

build_zip.py --int8 rewrites the shipped state_dict and _fno.dequant8 reads it back.  That pair
is the only place in the project where the bytes that ship are not the bytes that were trained,
and round 13 earned the rule that a feature nothing local exercises needs its own verifier: the
gated mean-blend passed validate_strict twice while being wrong, because no validation window
ever fired its gate.  This file runs the REAL packer and the REAL loader -- not a
reimplementation of either -- and scores both ensembles on val_unseen with the official kit.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
sys.path.insert(0, TRACK1)
sys.path.insert(0, os.path.join(TRACK1, "submit_fno"))

import _fno    # noqa: E402
import opoint  # noqa: E402
import sps9    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    # 2026-09-28: defaults from opoint (were 0.60 and submit_fno/band.npz, an untracked build
    # by-product). The band is only packed into the staging archive; nothing here scores sps.
    ap.add_argument("--alpha", type=float, default=opoint.ALPHA,
                    help=f"compose alpha on both sides (default {opoint.ALPHA}, opoint.ALPHA)")
    ap.add_argument("--joint-gain", action=argparse.BooleanOptionalAction, default=opoint.JOINT_GAIN,
                    help="compose with _fno.JOINT_GAIN on both sides (default "
                         f"{opoint.JOINT_GAIN}, opoint.JOINT_GAIN)")
    ap.add_argument("--band", default=opoint.path(opoint.FINAL_BAND),
                    help=f"band packed into the staging archive (default {opoint.FINAL_BAND})")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _fno.JOINT_GAIN = bool(args.joint_gain)
    du = np.load(os.path.join(TRACK1, "val_unseen.npz"))
    X, Y = du["x"], du["y"]

    tmp = tempfile.mkdtemp(prefix="q8ver_")
    try:
        stage = os.path.join(tmp, "stage")
        zp = os.path.join(tmp, "s.zip")
        cmd = [sys.executable, os.path.join(TRACK1, "submit_fno", "build_zip.py"),
               "--int8", "--band", args.band, "--stage", stage, "--out", zp,
               "--ckpt", *args.ckpt]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:] or r.stderr[-2000:])
        if r.returncode != 0:
            raise SystemExit("build_zip --int8 failed")

        names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(args.ckpt) + 1)]
        packed = [os.path.join(stage, n) for n in names]
        mb = sum(os.path.getsize(p) for p in packed) / 1e6
        print(f"\npacked members: {mb:.1f} MB for {len(packed)}  "
              f"({mb/len(packed):.1f} MB each) -- the 256 MB cap is on the whole extracted tree")

        ref = [_fno.load_any(c, dev) for c in args.ckpt]
        q8 = [_fno.load_any(p, dev) for p in packed]
        P0 = _fno.predict_fno(ref, X, dev, alpha=args.alpha)
        P1 = _fno.predict_fno(q8, X, dev, alpha=args.alpha)
        d = float(np.linalg.norm(P1 - P0) / np.linalg.norm(P0))
        print(f"prediction difference: rel-L2 {d:.3e}   max|Δ| {float(np.abs(P1-P0).max()):.3e}")

        def sc3(P):
            Q = np.concatenate([P, np.zeros_like(P[..., :1])], -1).astype(np.float32)
            return (float(np.mean(sps9.sc.rel_l2_per_sample(Q, Y, 2))),
                    float(np.mean(sps9.sc.tke_rel_l2_per_sample(Q, Y, 2))),
                    float(np.mean(sps9.sc.mvpe_rel_l2_per_sample(Q, Y))))
        a, b = sc3(P0), sc3(P1)
        print(f"{'':10s} {'rel_l2':>10s} {'tke':>10s} {'mvpe':>10s}")
        print(f"{'fp32':10s} {a[0]:10.6f} {a[1]:10.6f} {a[2]:10.6f}")
        print(f"{'int8':10s} {b[0]:10.6f} {b[1]:10.6f} {b[2]:10.6f}")
        print(f"{'Δ %':10s} " + " ".join(f"{100*(y-x)/x:10.4f}" for x, y in zip(a, b)))
        ok = d < 2e-3 and all(abs(y - x) / x < 2e-3 for x, y in zip(a, b))
        print("\nINT8 PACKING OK" if ok else "\nINT8 PACKING FAILED -- do not ship")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
