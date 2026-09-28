"""Round 17 Q1a/Q1b: what the fast paths cost, what they change, and whether they are shippable.

Both shipped packages run fp32 with TF32 OFF, because round 15b measured a window's output
depending on its POSITION in the batch once TF32 was on (9.1e-6 on the fused w96 FNO group,
1.8e-4 on the U-FNO -- the latter fails the submission gate's 1e-4 permutation bound outright).
The plan's hypothesis is that this is really a dependence on the batch SHAPE: only the last,
smaller batch gets a different kernel, so padding every batch to one fixed size should remove
it and buy back the tensor cores.  At w96 L8 the 1x1 convolutions are ~70 % of the FLOPs, so
this is the whole latency bet.

For each (member set, variant) this measures, on the 336 val_unseen windows:

  * end-to-end ms/window, the same timed section `lat_r15.py` measures (predict_fno plus the
    band arithmetic), median of --rep passes;
  * REPEAT determinism -- the same call twice, which must be 0.0 for any of the rest to mean
    anything;
  * PERMUTATION dependence -- the 336 windows shuffled, predicted, un-shuffled, compared to
    the natural-order run.  Run twice: once on all 336 (which every candidate batch size
    divides exactly, so there is no partial tail) and once on 333 (which none of them divides,
    so there IS one).  The second is the test the padding is for; the first is the control that
    says whether a full batch is position-dependent on its own.
  * DEVIATION from the fp32/TF32-off reference, as rel-L2 over the whole field;
  * the four scored error subscores under the package's own shipped band.

Shippable = permutation-invariant to <= 1e-5 relative AND rel-L2 deviation <= 1e-3 from the
reference.  The subscores say what it is worth; those two say whether it is allowed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import final_v9                                                        # noqa: E402
import opoint                                                          # noqa: E402
import sps9                                                            # noqa: E402
from sps9 import sc                                                    # noqa: E402

R_4060 = 1.850
BOARD_C, BOARD_INV, BOARD_LOC = 2.616, 0.435, 0.803
FINAL_PER_BOARD_MS = -0.070

# name -> (TF32, pad the tail batch, autocast dtype)
VARIANTS = {
    "fp32":        (False, False, None),          # the reference: exactly what ships today
    "fp32pad":     (False, True,  None),
    "tf32":        (True,  False, None),
    "tf32pad":     (True,  True,  None),
    "fp16pad":     (False, True,  torch.float16),
    "fp16tf32pad": (True,  True,  torch.float16),
    "bf16pad":     (False, True,  torch.bfloat16),
}


def run(models, x, band, ref, p, alpha, batch, pad, amp):
    """A byte-for-byte replica of submission.predict()'s timed section, plus the sigma."""
    import _fno
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    sig = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32) if p else None
    _fno.predict_fno(models, x, "cuda", alpha=alpha, batch=batch, out=pred, sigma_out=sig,
                     pad_batch=pad, amp=amp)
    if p and sig is not None and band is not None:
        r3 = np.zeros((len(x), 1, 32, 64, 3), dtype=np.float32)
        np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        lower, upper = np.empty_like(pred), np.empty_like(pred)
        t_pred, t_lo, t_hi = (torch.from_numpy(a) for a in (pred, lower, upper))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=t_hi)
        torch.sub(t_pred, t_hi, out=t_lo)
        torch.add(t_pred, t_hi, out=t_hi)
        return pred, sig, lower, upper
    return pred, sig, pred, pred


def pick_batch(models, x, band, ref, p, alpha, pad, amp):
    best_t, best_b = None, 64
    for b in (14, 28, 56, 112):
        try:
            run(models, x[:b], band, ref, p, alpha, b, pad, amp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(models, x[:112], band, ref, p, alpha, b, pad, amp)
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
        except RuntimeError as e:                        # OOM at the widest fused group
            print(f"    batch {b}: {type(e).__name__}", flush=True)
            torch.cuda.empty_cache()
            continue
        if best_t is None or t < best_t:
            best_t, best_b = t, b
    return best_b


def rel(a, b):
    d = np.linalg.norm((a - b).ravel())
    n = np.linalg.norm(b.ravel())
    return float(d / n) if n else float(d)


def subscores(P, Y, lo, hi, tms):
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    mv = float(sc.mvpe_rel_l2(P, Y))
    # 2026-09-28: kit per-element rule on the given interval (band_io's, not necessarily P -/+ half)
    s = sps9.sps_interval(sps9.sample_weights(P, Y), Y, lo, hi)[0]
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), sc.score_time(tms / 1e3), s]
    return (r, k, mv), five


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="NAME=a.pth,b.pth ...")
    ap.add_argument("--band", required=True,
                    help="band npz, read through band_io.load_band (a SIGBLEND npz needs its sidecar); "
                         f"the package of record's is {opoint.FINAL_BAND}")
    # 2026-09-28: default from opoint (was 0.60).
    ap.add_argument("--alpha", type=float, default=opoint.ALPHA,
                    help=f"compose alpha (default {opoint.ALPHA}, opoint.ALPHA)")
    ap.add_argument("--joint-gain", action=argparse.BooleanOptionalAction, default=None,
                    help="compose with _fno.JOINT_GAIN (same cost either way). Neither flag (default) = "
                         f"opoint.JOINT_GAIN ({opoint.JOINT_GAIN}) when --band is centred on y0, else off")
    ap.add_argument("--rep", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--batch", type=int, default=0, help="0 = measure it, per variant")
    ap.add_argument("--verify-baseline", action="store_true",
                    help="check the default path against git HEAD's _fno.py, bit for bit")
    args = ap.parse_args()

    import _fno
    import band_io
    bd = band_io.load_band(args.band)
    band, ref, p = bd.base, bd.sigma_ref, bd.p
    _fno.JOINT_GAIN = band_io.joint_gain_for(bd, args.joint_gain)
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    x, Y = du["x"].astype(np.float32), du["y"]
    rng = np.random.default_rng(17)
    perm = rng.permutation(len(x))
    inv = np.argsort(perm)
    RAG = 333                       # no candidate batch size divides it -> a partial tail
    perm_r = rng.permutation(RAG)
    inv_r = np.argsort(perm_r)
    print(f"band {bd.describe()}  joint_gain {_fno.JOINT_GAIN}", flush=True)
    print(f"windows={len(x)} band={os.path.basename(args.band)} alpha={args.alpha} p={p} "
          f"ragged_n={RAG}", flush=True)

    for spec in args.sets:
        name, _, lst = spec.partition("=")
        paths = [q if os.path.isabs(q) else os.path.join(HERE, q) for q in lst.split(",")]
        models = _fno.fuse([_fno.load_any(q, "cuda") for q in paths])
        npar = sum(sum(t.numel() for t in m.parameters()) for m in models)
        print(f"\n=== {name}: {len(paths)} members -> {len(models)} invocation(s), "
              f"{npar/1e6:.2f}M params ===", flush=True)
        print(f"{'variant':<12} {'bs':>4} {'A100ms':>7} {'4060ms':>7} {'board':>6} {'dfin':>6} "
              f"{'repeat':>9} {'perm336':>9} {'perm333':>9} {'vs fp32':>9} | "
              f"{'relL2':>7} {'TKE':>7} {'MVPE':>7} {'sps':>6} {'wfinal':>7}")

        reference = None
        base_board = None
        for v in args.variants:
            tf32, pad, amp = VARIANTS[v]
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            b = args.batch or pick_batch(models, x, band, ref, p, args.alpha, pad, amp)
            try:
                P, _, lo, hi = run(models, x, band, ref, p, args.alpha, b, pad, amp)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.rep):
                    t0 = time.perf_counter()
                    run(models, x, band, ref, p, args.alpha, b, pad, amp)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) / len(x) * 1e3)
                a100 = float(np.median(ts))

                P2 = run(models, x, band, ref, p, args.alpha, b, pad, amp)[0]
                Pp = run(models, x[perm], band, ref, p, args.alpha, b, pad, amp)[0][inv]
                Pr = run(models, x[:RAG], band, ref, p, args.alpha, b, pad, amp)[0]
                Prp = run(models, x[:RAG][perm_r], band, ref, p, args.alpha, b, pad,
                          amp)[0][inv_r]
            except RuntimeError as e:
                print(f"{v:<12} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
                torch.cuda.empty_cache()
                continue
            if reference is None:
                reference = P.copy()
            loc = a100 * R_4060
            boardms = BOARD_C + BOARD_INV * len(models) + BOARD_LOC * loc
            if base_board is None:
                base_board = boardms
            scale = float(np.abs(reference).max())
            if bd.centre_alpha is not None or bd.kind != "static":
                # The timed replica is the pre-09-21 host path (pred -/+ base*ratio^p). Score the
                # package's interval instead (y0 centre, SIGBLEND factor), from one UNTIMED pass.
                c0 = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
                s0 = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32)
                _fno.predict_fno(models, x, "cuda", alpha=args.alpha, batch=b, sigma_out=s0,
                                 centre_out=c0, pad_batch=pad, amp=amp)
                lo, hi = band_io.interval(bd, c0, P, s0, alpha=args.alpha, joint_gain=_fno.JOINT_GAIN)
            (r, k, mv), five = subscores(P, Y, lo, hi, loc)
            print(f"{v:<12} {b:>4} {a100:>7.3f} {loc:>7.3f} {boardms:>6.2f} "
                  f"{(boardms - base_board) * FINAL_PER_BOARD_MS:>+6.3f} "
                  f"{np.abs(P - P2).max():>9.2e} {np.abs(P - Pp).max() / scale:>9.2e} "
                  f"{np.abs(Pr - Prp).max() / scale:>9.2e} {rel(P, reference):>9.2e} | "
                  f"{r:>7.4f} {k:>7.4f} {mv:>7.4f} {five[4]:>6.2f} "
                  f"{final_v9.predict(five)[0]:>7.3f}", flush=True)
        del models
        torch.cuda.empty_cache()

    print("\nrepeat/perm columns are max|delta| (perm ones divided by max|pred|, i.e. relative);"
          "\n'vs fp32' is rel-L2 of the whole field against the first variant listed."
          "\nSHIPPABLE = perm336 and perm333 <= 1e-5 AND 'vs fp32' <= 1e-3.")

    if args.verify_baseline:
        print("\n=== default path vs git HEAD's _fno.py, bit for bit ===", flush=True)
        import importlib.util
        import subprocess
        ref_src = subprocess.run(["git", "-C", os.path.dirname(HERE), "show",
                                  "HEAD:Track_1/submit_fno/_fno.py"],
                                 capture_output=True, text=True, check=True).stdout
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "_fno_head.py")
        open(tmp, "w").write(ref_src)
        spec = importlib.util.spec_from_file_location("_fno_head", tmp)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for s in args.sets:
            nm, _, lst = s.partition("=")
            paths = [q if os.path.isabs(q) else os.path.join(HERE, q) for q in lst.split(",")]
            a = _fno.fuse([_fno.load_any(q, "cuda") for q in paths])
            b = old.fuse([old.load_any(q, "cuda") for q in paths])
            pa = _fno.predict_fno(a, x, "cuda", alpha=args.alpha, batch=56)
            pb = old.predict_fno(b, x, "cuda", alpha=args.alpha, batch=56)
            print(f"{nm:<24} max|new - HEAD| = {np.abs(pa - pb).max():.3e} "
                  f"({'BIT-IDENTICAL' if np.array_equal(pa, pb) else 'DIFFERS'})", flush=True)
            del a, b
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
