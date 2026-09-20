"""WP-1 screen: do an FGN model's SAMPLES beat the log-MSE sigma head it was bolted onto?

    python eval_fgn.py --ckpt unet/fno_fgn1.pth --baseline unet/fno_vl_s1.pth --aoa 15 20

plan_20260810 WP-1 states the screen and the stopping rule in one line: **if no sigma candidate
beats the head's, stop the branch** -- do not tune it into working.  This script is that test.

The three sigma candidates come from the same K forward passes:

    head   exp(logstd) from the z = 0 pass -- the shipped estimator, rel-L2 ~0.358 on
           val_unseen (eval_pair.py's convention: PHYSICAL units, biased std, so it is
           reproduced here to the letter and the input-window-std anchor is printed beside it)
    s1     mean_k std_t(x_k)     -- the mean sample amplitude
    s2     sqrt(mean_k var_t(x_k)) -- the root-mean sample amplitude, >= s1 by Jensen

and two field candidates, because a generative model still has to submit ONE array:

    z0     the z = 0 field, which is bit-identical to the deterministic model
    smean  mean over the K samples -- lower L2 by construction, and the ensemble argument

`compose(field, sigma, alpha)` then puts them together exactly as `_fno.predict_fno` does at
submission time, so a winning row here is a row that can actually be shipped.  ALPHA is swept
over {0.75, 0.90} because that trade does not dissolve just because the sigma got better: the
scorer computes `tke` on `prediction`, so the submitted array must be simultaneously close in
L2 and correctly dispersed.

Conventions follow compare_v9.py: each row gets its OWN band, fitted on val_seen and reported
on val_unseen, and `--aoa 15 20` restricts the REPORTING split only (the board's rel-L2 is
1.06x that subset and 1.46x the full one).  `wfinal` is final_v9's board weighting applied to
local errors -- a ranking number, not a board prediction.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np

import final_v9
from sps9 import prep, sc, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "unet"))


# --------------------------------------------------------------------------- checkpoint


def load_alt(path: str, device: str):
    """-> (model, fgn_dim). Builds through models_alt.build from the blob's stored keys.

    Deliberately NOT an extension of `submit_fno/_fno.load`: that file is the submission's
    frozen copy of the architecture and packaging is deferred until this screen returns a
    verdict.  Same blob, same keys, but through the training tree, which is where `fgn_dim`
    and the `z` argument live.
    """
    import torch
    from models_alt import build

    blob = torch.load(path, map_location=device)
    n_ctx = max(int(blob.get("ctx", 20)) - 20, 0)
    if blob.get("hist", False) or n_ctx:
        raise SystemExit(f"{path}: --hist / --ctx checkpoints need extra input channels this "
                         f"screen does not build")
    m = build(blob.get("arch", "fno"), blob.get("width", 24), blob.get("var_head", True), 0,
              blob.get("modes_y", 8), blob.get("modes_x", 12), blob.get("n_layers", 4),
              blob.get("p", 256),
              var_mlp=blob.get("var_mlp", False), lift_mlp=blob.get("lift_mlp", False),
              point_mlp=blob.get("point_mlp", False), var_hidden=blob.get("var_hidden", 128),
              var_deep=blob.get("var_deep", False), var_in_std=blob.get("var_in_std", False),
              var_pred_std=blob.get("var_pred_std", False),
              fgn_dim=int(blob.get("fgn_dim", 0))).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    return m, int(blob.get("fgn_dim", 0))


# --------------------------------------------------------------------------- inference


def infer(m, X: np.ndarray, dev: str, zs, batch: int = 64):
    """-> (fields, sigmas), both dicts of numpy arrays in NORMALIZED units.

    X is raw physical (N, 20, 32, 64, >=2); `zs` is (K, N, fgn_dim) or None.

    Samples are ACCUMULATED, never stored.  K = 8 sample windows at N = 336 is 880 MB, while
    the four statistics they feed are one 110 MB field plus three 5.5 MB std maps -- and the
    accumulators are what every candidate below is actually built from.
    """
    import torch
    from model import C, H, MEAN, STD, T_OUT, W, from_ch, std_t, to_ch

    n = len(X)
    f0 = np.empty((n, T_OUT, H, W, C), dtype=np.float32)
    fm = np.empty_like(f0) if zs is not None else None
    sig = {}
    if m.head_logstd is not None:
        sig["head"] = np.empty((n, 1, H, W, C), dtype=np.float32)
    if zs is not None:
        sig["s1"] = np.empty((n, 1, H, W, C), dtype=np.float32)
        sig["s2"] = np.empty((n, 1, H, W, C), dtype=np.float32)

    mean = torch.from_numpy(MEAN).to(dev)
    std = torch.from_numpy(STD).to(dev)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)[..., :2].float()
            prev = to_ch((xb - mean) / std)
            raw, logstd = m(prev)                       # z = None: the deterministic model
            f0[i:i + batch] = from_ch(raw).cpu().numpy()
            if logstd is not None:
                s = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                sig["head"][i:i + batch] = s.cpu().numpy()
            if zs is not None:
                acc = accs = accv = None
                for k in range(len(zs)):
                    yk = from_ch(m(prev, z=zs[k, i:i + batch])[0])
                    sk = std_t(yk)
                    acc = yk if acc is None else acc + yk
                    accs = sk if accs is None else accs + sk
                    accv = sk * sk if accv is None else accv + sk * sk
                kf = float(len(zs))
                fm[i:i + batch] = (acc / kf).cpu().numpy()
                sig["s1"][i:i + batch] = (accs / kf).cpu().numpy()
                sig["s2"][i:i + batch] = (accv / kf).sqrt().cpu().numpy()

    fields = {"z0": f0}
    if fm is not None:
        fields["smean"] = fm
    return fields, sig


def composed(field: np.ndarray, sigma: np.ndarray | None, alpha: float, dev: str):
    """(N,T,H,W,2) normalized + (N,1,H,W,2) sigma -> (N,T,H,W,3) raw physical, p zero-filled.

    Uses model.compose, i.e. the identical rescaling `_fno.predict_fno` applies on the GPU
    before the host transfer, so nothing here is a re-derivation of the shipped path.
    """
    import torch
    from model import MEAN, STD, compose

    y = torch.from_numpy(field).to(dev)
    if sigma is not None and alpha > 0.0:
        y = compose(y, torch.from_numpy(sigma).to(dev), alpha)
    mean = torch.from_numpy(MEAN).to(dev)
    std = torch.from_numpy(STD).to(dev)
    uv = (y * std + mean).float().cpu().numpy()
    out = np.zeros(uv.shape[:-1] + (3,), dtype=np.float32)
    out[..., :2] = uv
    return out


# --------------------------------------------------------------------------- band


def fit_band(Ps: np.ndarray, Ys: np.ndarray, grid=np.linspace(0.1, 3.0, 30)):
    """`c * optimal_half(err_map)` fitted on the SEEN split -- eval_v9.main's band, verbatim.

    The scale search runs through sps9's factorized scorer instead of eval_v9.fit_scale's call
    into the kit's `aggregate_sps`: the same objective (sps9.self_check agrees with the kit to
    6.5e-8) at a fraction of the cost, which is what makes 12 field x sigma x alpha rows
    affordable in one pass.
    """
    from eval_v9 import err_map, optimal_half

    # Measured channels only: sps9's fast path scores |error| on (u, v), so the half-width has
    # to have that shape too. eval_v9 keeps the dead p channel because it forms P +- half and
    # lets the kit drop it; here the band and the error array are multiplied directly.
    w = optimal_half(err_map(Ps, Ys)[..., :2])
    A, a, scored, n = prep(Ps, Ys)
    best_v, best_c = -1.0, float(grid[0])
    for c in grid:
        v = sps_score(A, a, scored, n, (c * w).astype(np.float32))
        if v > best_v:
            best_v, best_c = v, float(c)
    return (best_c * w).astype(np.float32), best_c


# --------------------------------------------------------------------------- reporting


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[],
                    help="FGN checkpoints (blob fgn_dim > 0). Plain FNOs are accepted here too "
                         "and simply take the z-free path.")
    ap.add_argument("--baseline", nargs="+", default=[],
                    help="plain FNO checkpoints, scored through the SAME code path with the "
                         "sample candidates skipped, so the comparison is one table and not "
                         "two runs of two scripts.")
    ap.add_argument("--k", type=int, default=8,
                    help="samples drawn per window at eval. Larger than training's K = 4 "
                         "because eval has no memory pressure and the sample-spread estimators "
                         "converge as 1/sqrt(K); deployment cost is priced separately, by "
                         "compose_v9.py, only if this screen passes.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the torch.Generator the noise vectors are drawn from. The "
                         "whole (K, N, fgn_dim) block is drawn ONCE up front and sliced per "
                         "batch, so a row is reproducible and independent of --batch.")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.75, 0.90],
                    help="fluctuation rescaling in compose(). 0.75 is shipped; gain_v9.py puts "
                         "the optimum of the one-dimensional family at ~0.90 and three "
                         "readings have all been positive, so both are swept.")
    ap.add_argument("--aoa", type=int, nargs="+", default=None,
                    help="restrict the REPORTING split (val_unseen) to these angles of attack. "
                         "Error is governed by AoA and by almost nothing else, and the board "
                         "reads at or above our hardest single trajectory, so 15-20 is the "
                         "regime that most resembles it. The band is still fitted on all of "
                         "val_seen, as in compare_v9.py.")
    ap.add_argument("--ms", type=float, default=1.32,
                    help="end-to-end ms/window used for time_score. Identical across rows, so "
                         "it shifts the table rather than reordering it -- sample count is a "
                         "deployment question, priced with the board-time model, not here.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-dir", default=HERE,
                    help="directory holding val_seen.npz / val_unseen.npz. Only moved for "
                         "smoke tests on cut-down data.")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N windows of each split. Smoke-test knob; 0 = all.")
    args = ap.parse_args()
    if not args.ckpt and not args.baseline:
        raise SystemExit("nothing to score: pass --ckpt and/or --baseline")

    import torch

    from model import STD

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(args.val_dir, "val_seen.npz"))
    du = np.load(os.path.join(args.val_dir, "val_unseen.npz"))
    cut = slice(None) if args.limit <= 0 else slice(0, args.limit)
    Xs, Ys = ds["x"][cut], ds["y"][cut]
    Xu, Yu, ids = du["x"][cut], du["y"][cut], du["ids"][cut]

    sel = np.ones(len(Xu), dtype=bool)
    if args.aoa:
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])
        sel = np.isin(aoa, args.aoa)
        print(f"val_unseen restricted to AoA {args.aoa}: {int(sel.sum())}/{len(aoa)} windows")
    t_score = sc.score_time(args.ms / 1000.0)

    # The anchor for the sigma column: reusing the INPUT window's own per-pixel std is the
    # zero-parameter estimator every sigma head has had to beat since round 3.
    tgt_u = Yu[..., :2].std(1, keepdims=True)
    in_u = Xu[..., :2].std(1, keepdims=True)
    print(f"\nsigma reference (physical units, rel-L2 vs std_t(target) on val_unseen):")
    print(f"  {'input-window std (zero-parameter)':38s} all {rel(in_u, tgt_u):6.4f}   "
          f"aoa {rel(in_u[sel], tgt_u[sel]):6.4f}")
    print(f"  {'shipped variance head (reported)':38s} all 0.3580   aoa      -")

    rows = []
    for path in list(args.ckpt) + list(args.baseline):
        p = path if os.path.isabs(path) else os.path.join(HERE, path)
        tag = os.path.basename(p)[:-4] if p.endswith(".pth") else os.path.basename(p)
        if not os.path.exists(p):
            print(f"\n{tag}: (missing)")
            continue
        m, fgn_dim = load_alt(p, dev)

        # One fixed (K, N, fgn_dim) block per split, drawn up front from a generator re-seeded
        # per split. Slicing it per batch is what makes a row independent of --batch, and
        # re-seeding makes each split's draw independent of the other split's size.
        def draw(n):
            g = torch.Generator(device=dev).manual_seed(args.seed)
            return torch.randn(args.k, n, fgn_dim, generator=g, device=dev)

        zs_s = zs_u = None
        if fgn_dim > 0 and args.k > 0:
            zs_s, zs_u = draw(len(Xs)), draw(len(Xu))
        Fs, Ss = infer(m, Xs, dev, zs_s, args.batch)
        Fu, Su = infer(m, Xu, dev, zs_u, args.batch)

        print(f"\n{tag}  (fgn_dim {fgn_dim}, K {args.k if fgn_dim else 0}) "
              f"sigma rel-L2, physical:")
        for k in ("head", "s1", "s2"):
            if k in Su:
                sp = Su[k] * STD
                print(f"  {k:38s} all {rel(sp, tgt_u):6.4f}   "
                      f"aoa {rel(sp[sel], tgt_u[sel]):6.4f}")

        for fk in ("z0", "smean"):
            if fk not in Fs:
                continue
            for sk in ("head", "s1", "s2"):
                if sk not in Ss:
                    continue
                for alpha in args.alphas:
                    Ps = composed(Fs[fk], Ss[sk], alpha, dev)
                    Pu = composed(Fu[fk], Su[sk], alpha, dev)[sel]
                    Yv = Yu[sel]
                    band, c = fit_band(Ps, Ys)
                    r = float(np.mean(sc.rel_l2_per_sample(Pu, Yv, 2)))
                    kk = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yv, 2)))
                    mv = sc.mvpe_rel_l2(Pu, Yv)
                    five = [sc.score_error(r), sc.score_error(kk), sc.score_error(mv), t_score,
                            sps_score(*prep(Pu, Yv), band)]
                    rows.append((tag, fk, sk, alpha, c, r, kk, mv, five))

    hdr = (f"{'checkpoint':20s} {'field':6s} {'sig':5s} {'alph':>5s} {'c':>5s} | {'relL2':>7s} "
           f"{'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} {'mvpe':>6s} {'time':>6s} "
           f"{'sps':>6s} | {'mean5':>7s} {'wfinal':>7s}")
    print(f"\n{hdr}\n" + "-" * len(hdr))
    for tag, fk, sk, alpha, c, r, kk, mv, five in rows:
        print(f"{tag:20s} {fk:6s} {sk:5s} {alpha:5.2f} {c:5.2f} | {r:7.4f} {kk:7.4f} {mv:7.4f} "
              f"| {five[0]:6.2f} {five[1]:6.2f} {five[2]:6.2f} {five[3]:6.2f} {five[4]:6.2f} "
              f"| {np.mean(five):7.3f} {final_v9.predict(five)[0]:7.3f}")


if __name__ == "__main__":
    main()
