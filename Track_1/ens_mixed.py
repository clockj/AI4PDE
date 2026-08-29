"""Score a MIXED-family ensemble (FNO + U-Net members) under v9 — compare_v9 for member lists
that _fno.load alone cannot load.

    python ens_mixed.py --ckpt unet/fno_vl_s1.pth ... unet/unet_g1.pth --aoa 15 20 [--solo]

Members are averaged exactly the way `_fno.predict_fno` averages FNO members — raw field and
sigma in NORMALIZED space, on the GPU, before one compose() and one denormalization — so an
all-FNO list through this script reproduces compare_v9's ensemble semantics, and a U-Net
member simply joins the average. The WP-3 question this answers is decorrelation: does adding
unet_g1 to the 7-FNO ensemble beat adding an 8th FNO seed?

Band and scoring follow compare_v9 verbatim: per-list band fitted on val_seen with
sps9.optimal_band at --inflate, reported on val_unseen (optionally an AoA subset), wfinal by
final_v9. `--ms` is a fixed nominal latency, identical across rows: it shifts the table, not
the ordering.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

import final_v9
from sps9 import optimal_band, prep, sc, sps_score


def load_member(path: str, dev: str):
    import torch
    import _fno
    blob = torch.load(path, map_location=dev)
    if blob.get("arch", "fno") == "unet":
        from model import UNetTwoHead
        m = UNetTwoHead(blob.get("width", 64), var_head=blob.get("var_head", True)).to(dev)
        m.load_state_dict(blob["model_state_dict"])
        m.eval()
        return m
    return _fno.load(path, dev)


def infer_mixed(models, x_raw: np.ndarray, dev: str, alpha: float,
                batch: int = 64) -> np.ndarray:
    """Raw physical (N,20,32,64,>=2) -> (N,20,32,64,3), member-averaged then composed once."""
    import torch
    import _fno
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    T, Hh, Ww, C = _fno.T_OUT, _fno.H, _fno.W, _fno.C
    out = np.zeros((len(x_raw), T, Hh, Ww, 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x_raw), batch):
            xb = torch.from_numpy(x_raw[i:i + batch]).to(dev)[..., :2].float()
            prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), T * C, Hh, Ww)
            y = sig = None
            for m in models:
                raw, logstd = m(prev)
                yi = raw.reshape(len(xb), T, C, Hh, Ww).permute(0, 1, 3, 4, 2)
                y = yi if y is None else y + yi
                si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                sig = si if sig is None else sig + si
            y, sig = y / len(models), sig / len(models)
            if alpha > 0:
                mu = y.mean(1, keepdim=True)
                s = y.std(1, keepdim=True, unbiased=False)
                y = mu + (y - mu) * ((alpha * sig + (1 - alpha) * s) / (s + 1e-6))
            out[i:i + batch, ..., :2] = (y * std + mean).float().cpu().numpy()
    return out


def score_row(tag, models, ds, du, sel, alpha, f, t_score, dev):
    Ps, Ys = infer_mixed(models, ds["x"], dev, alpha), ds["y"]
    Pu, Yu = infer_mixed(models, du["x"], dev, alpha)[sel], du["y"][sel]
    S, U = prep(Ps, Ys), prep(Pu, Yu)
    band = optimal_band(np.float32(f) * S[0], S[1], S[2])
    r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
    mv = sc.mvpe_rel_l2(Pu, Yu)
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), t_score,
            sps_score(*U, band)]
    print(f"{tag:34s} {r:7.4f} {k:7.4f} {mv:7.4f} | {five[0]:6.2f} {five[1]:6.2f} "
          f"{five[2]:6.2f} {five[3]:6.2f} {five[4]:6.2f} | {np.mean(five):7.3f} "
          f"{final_v9.predict(five)[0]:7.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--inflate", type=float, default=1.5)
    ap.add_argument("--ms", type=float, default=1.32)
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    ap.add_argument("--solo", action="store_true", help="also score each member alone")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    t_score = sc.score_time(args.ms / 1000.0)
    sel = slice(None)
    if args.aoa:
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
        sel = np.isin(aoa, args.aoa)
        print(f"val_unseen restricted to AoA {args.aoa}: {int(sel.sum())}/{len(aoa)} windows")

    members = [(os.path.basename(c), load_member(
        c if os.path.isabs(c) else os.path.join(HERE, c), dev)) for c in args.ckpt]

    hdr = (f"{'ensemble':34s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'time':>6s} {'sps':>6s} | {'mean5':>7s} {'wfinal':>7s}")
    print(hdr)
    print("-" * len(hdr))
    if args.solo:
        for name, m in members:
            score_row(name, [m], ds, du, sel, args.alpha, args.inflate, t_score, dev)
    score_row(f"ALL {len(members)} members", [m for _, m in members], ds, du, sel,
              args.alpha, args.inflate, t_score, dev)


if __name__ == "__main__":
    main()
