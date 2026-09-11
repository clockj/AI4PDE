"""Can eight width-96 8-layer members fit under the 256 MB cap?  Measured, scheme by scheme.

    python quant8.py --ckpt unet/fno_all_w96L8do45_s1.pth ...

WHY THIS IS SUDDENLY WORTH DOING.  Round 17's Q1f priced 8 members against 4 at "+0.086 of
accuracy for a 0.09-0.10 latency bill", using the board-time model's 0.803 ms-per-local-ms
slope, and concluded it was a wash.  R18 and T2R6 then measured that slope directly, by shipping
the SAME prediction array through a faster path on each board:

    R17  -> R18   local -0.352 ms (4060-eq)   board -0.089 ms   slope +0.25
    T2R5 -> T2R6  local -0.890 ms             board +0.039 ms   slope -0.04
    T2R3 -> T2R5  local +2.353 ms             board +0.114 ms   slope +0.05

and refitting the eleven one-invocation rounds gives `board = 3.274 + 0.336*local`, rms 0.337 ms,
against the old model's 0.431 slope on top of a 4.447 intercept that misses every modern round
by 1.3-2.5 ms.  At 0.336 the extra four members cost 0.52 board ms = 0.037 `final` on Track 1
and ~0 on Track 2, so 8 members is worth about +0.05 / +0.09 -- as much as any accuracy arm on
the table, and it needs no training at all.

WHAT BLOCKS IT IS SIZE, AND ONLY SIZE.  A w96 L8 member is 108.5 MB fp32 / 54.3 MB fp16, so
four fill 217 MB of the 256 MB extracted cap and eight need 434.  The spectral weights are
~97 % of that.  fp16 is already the shipped packing and costs rel-L2 5.5e-4 of the field
against a model rel-L2 of 0.078, i.e. 2e-6 of the error in quadrature; this file asks what
int8 costs, under scaling schemes of increasing granularity, and whether any of them is cheap
enough to be free the way fp16 is.  The decision rule is the same one build_zip.py used for
fp16: the perturbation must be small enough that it adds in quadrature to nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import _fno   # noqa: E402


def q_int8(w, axes):
    """Symmetric int8 with a scale per index of `axes` (a tuple of kept axes, () = per tensor).

    Returns (dequantized tensor, bytes stored).  Scales are fp16, which is what a real packer
    would write; the count is tiny either way and is included so the size column is honest.
    """
    import torch
    if not w.is_floating_point() or w.numel() < 4096:
        return w.clone(), w.numel() * 2                      # small tensors stay fp16
    red = tuple(i for i in range(w.dim()) if i not in axes)
    s = w.abs().amax(dim=red, keepdim=True).clamp(min=1e-12) / 127.0
    s = s.half().float()                                     # the scale is stored in fp16 too
    q = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)
    n_scale = int(np.prod([w.shape[i] for i in axes])) if axes else 1
    return q.float() * s, w.numel() + 2 * n_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--n", type=int, default=96, help="val_unseen windows for the perturbation")
    ap.add_argument("--score", metavar="BAND.npz", default=None,
                    help="also SCORE every scheme through the official scorer with this band "
                         "held FIXED -- the band the package would actually ship. Forces "
                         "--n to the whole split.")
    args = ap.parse_args()
    if args.score:
        args.n = 10 ** 9

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    X = du["x"][:args.n]

    blobs = [torch.load(c, map_location="cpu") for c in args.ckpt]
    fp32_bytes = sum(sum(v.numel() * v.element_size() for v in b["model_state_dict"].values())
                     for b in blobs)
    print(f"{len(blobs)} member(s), fp32 state_dict = {fp32_bytes/1e6:.1f} MB "
          f"({fp32_bytes/1e6/len(blobs):.1f} MB each)")

    def build(fn):
        """Rebuild the ensemble from transformed state dicts, through _fno's own loader."""
        ms, nb = [], 0
        for c, b in zip(args.ckpt, blobs):
            sd = {}
            for k, v in b["model_state_dict"].items():
                nv, n = fn(v)
                sd[k] = nv
                nb += n
            tmp = dict(b); tmp["model_state_dict"] = sd
            path = os.path.join("/tmp", f"_q8_{os.path.basename(c)}")
            torch.save(tmp, path)
            ms.append(_fno.load_any(path, dev))
            os.remove(path)
        return ms, nb

    ref, _ = build(lambda v: (v.clone(), v.numel() * v.element_size()))
    P0 = _fno.predict_fno(ref, X, dev, alpha=args.alpha)
    nrm = float(np.linalg.norm(P0))
    del ref
    print(f"\n{'scheme':<26s} {'MB/member':>10s} {'8 memb MB':>10s} {'relL2(dP)':>11s} {'maxabs':>10s}")

    schemes = [
        ("fp16 (shipped)", lambda v: ((v.half().float() if v.is_floating_point() else v.clone()),
                                      v.numel() * (2 if v.is_floating_point() else
                                                   v.element_size()))),
        ("int8 per tensor", lambda v: q_int8(v, ())),
        ("int8 per out-channel", lambda v: q_int8(v, (0,))),
        ("int8 per out x in", lambda v: q_int8(v, (0, 1))),
        ("int8 per mode (ky,kx)", lambda v: q_int8(v, tuple(range(2, v.dim())) if v.dim() >= 4
                                                   else (0,))),
    ]
    for name, fn in schemes:
        ms, nb = build(fn)
        P = _fno.predict_fno(ms, X, dev, alpha=args.alpha)
        d = float(np.linalg.norm(P - P0)) / nrm
        mb = nb / 1e6 / len(blobs)
        print(f"{name:<26s} {mb:>10.1f} {8*mb:>10.1f} {d:>11.3e} "
              f"{float(np.abs(P - P0).max()):>10.3e}")
        del ms

    if args.score:
        score_schemes(args, blobs, X, du, dev, schemes)

    print("\nreference: the model's own rel-L2 on val_unseen is ~0.077, and errors add in")
    print("quadrature -- a perturbation of 1e-3 costs 0.0008 % of the error, 1e-2 costs 0.8 %.")
    print("The 256 MB cap is on the EXTRACTED archive: members + submission.py + _fno.py + band")
    print("(the band is 20*32*64*3 fp32 = 0.5 MB, the code ~0.2 MB).")


def score_schemes(args, blobs, X, du, dev, schemes) -> None:
    """What each packing scheme costs the SCORE, with the shipped band held fixed.

    The rest of this file measures the field perturbation, which is the right quantity for
    deciding whether a scheme is admissible -- but it is NOT the quantity the board pays in.
    `sps` scores per-element interval CONTAINMENT, and a containment score is a TAIL statistic:
    a perturbation that adds 0.0008 % of the error in quadrature can still push elements that
    were sitting just inside their interval out of it, and every one of those forfeits its
    whole contribution.  Round 21 shipped int8 for the first time (4 fp16 FNOs + an fp16 ViT is
    257 MB against a 256 MB cap) bundled with two other changes, so its cost has never been
    separated, and every package bigger than four members needs int8 to exist at all.

    The band is held FIXED and is the one the package ships, because that is the deployed
    situation: `fit_band.py` runs on fp32 checkpoints and the archive is quantized afterwards.
    A refit column would answer a different and less useful question.

    Sigma comes from the SAME packed models as the field -- quantization moves the variance
    head too, and the band's per-window scaling reads it.
    """
    import numpy as np
    import sps9
    from sps9 import prep, sc, sps_score
    import final_v9

    Y = du["y"]
    z = np.load(args.score)
    base, ref, pw = z["base"][..., :2], z["sigma_ref"][..., :2], float(z["p"])

    def scored(ms):
        # `sigma_out` is an OUT parameter, not a flag: predict_fno writes into it and returns
        # only the u,v field, which then needs the zero pressure channel the scorer expects.
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(ms, X, dev, alpha=args.alpha, sigma_out=sig)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mv = float(sc.mvpe_rel_l2(P, Y))
        half = (base * np.clip(sig / ref, 0.3, 3.0) ** np.float32(pw)).astype(np.float32)
        A, _, sc_, n = prep(P, Y)
        dm, tk, mvv = sps9.sample_metrics(P, Y)
        a = sps9.weights_from_metrics(dm, tk, mvv)
        s = float(sps_score(A, a, sc_, n, half))
        w = final_v9.predict([sc.score_error(r), sc.score_error(k), sc.score_error(mv),
                              95.92, s])[0]
        return r, k, mv, s, w

    print(f"\nSCORED, band {os.path.basename(args.score)} held FIXED, "
          f"{len(X)} val_unseen windows, ALPHA {args.alpha}")
    hdr = (f"{'scheme':<26s} {'relL2':>8s} {'TKE':>8s} {'MVPE':>8s} {'sps':>7s} {'wfinal':>8s} "
           f"{'d sps':>7s}")
    print(hdr); print("-" * len(hdr))

    def build_local(fn):
        import torch
        ms = []
        for c, b in zip(args.ckpt, blobs):
            sd = {k: fn(v)[0] for k, v in b["model_state_dict"].items()}
            tmp = dict(b); tmp["model_state_dict"] = sd
            path = os.path.join("/tmp", f"_q8s_{os.path.basename(c)}")
            torch.save(tmp, path)
            ms.append(_fno.load_any(path, dev))
            os.remove(path)
        return ms

    ref_ms = build_local(lambda v: (v.clone(), 0))
    r0 = scored(ref_ms)
    print(f"{'fp32 (reference)':<26s} {r0[0]:>8.4f} {r0[1]:>8.4f} {r0[2]:>8.4f} "
          f"{r0[3]:>7.3f} {r0[4]:>8.3f} {'--':>7s}")
    del ref_ms
    for name, fn in schemes:
        ms = build_local(fn)
        rr = scored(ms)
        print(f"{name:<26s} {rr[0]:>8.4f} {rr[1]:>8.4f} {rr[2]:>8.4f} "
              f"{rr[3]:>7.3f} {rr[4]:>8.3f} {rr[3]-r0[3]:>+7.3f}")
        del ms
    print("\nDECISION RULE: the seed floor at w96 L8 is ~0.05 full-split sps. A scheme inside")
    print("it is free and can be shipped; a scheme outside it has to be subtracted from")
    print("whatever the extra members it buys are worth.")


if __name__ == "__main__":
    main()
