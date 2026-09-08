"""MC dropout as a ZERO-STORAGE ensemble: how many members can one checkpoint stand in for?

    python mcdrop_v9.py --ckpt unet/fno_r16_w96_do45_L8.pth --mc 1 2 4 8 16

Round 18 measured two things that make this worth asking. First, the board's latency slope
collapsed: `board ms = 3.274 + 0.336*local` over the eleven one-invocation rounds, so an extra
forward pass costs ~0.004 `final` and members are nearly free in TIME. Second, the binding
constraint moved to the 256 MB extracted cap -- eight w96 L8 members need int8 packing to fit
at 232 MB, and a ninth does not fit at all. So the question is no longer "can we afford more
forwards" but "can we get ensemble diversity without storing more weights".

Every shipped member trains at dropout 0.45. Leaving the Dropout2d modules in TRAIN mode at
inference and averaging N stochastic passes is an ensemble drawn from one checkpoint: N forward
passes, ONE set of weights, no archive cost. The shipped `predict_fno` already averages members
in raw-field and sigma space before `compose`, so passing the same model N times gives exactly
the ensemble semantics the package deploys -- nothing here is a different code path.

What it CANNOT do is manufacture the diversity of independent training runs; MC-dropout samples
share every weight. The honest comparison is therefore against BOTH ends: the deterministic
single model (does averaging noise help at all?) and the real K-member ensemble (does it come
anywhere near?). `--ckpt` may list several checkpoints, in which case each is sampled `mc`
times and all N*mc passes are averaged together.

NOTE the ceiling this is fighting. `Track_1/plan/campaign_20260906.md` §3b fits the member
curve as e(K) = e_inf + c/K: 4 -> 8 members is +0.041 `final` and 8 -> 16 is +0.020. An MC
sample is a WORSE member than an independently seeded one, so +0.041 is the optimistic bound
on the whole idea, not a target.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

import final_v9
import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

F_BOARD = 1.39


def load_with_dropout(path: str, device: str, force_train: bool):
    """The checkpoint's own class, not the submission's -- `_fno.FNO2d` accepts `dropout` and
    IGNORES it (it holds no parameters and the shipped path is always eval), so a model loaded
    through `load_any` has no Dropout2d to switch on. `models_alt.build` constructs the real
    one. The two produce the same field to float32 noise, which is what makes the deterministic
    row here comparable to every other instrument in the repo."""
    import torch
    import _fno

    blob = torch.load(path if os.path.isabs(path) else os.path.join(HERE, path),
                      map_location=device)
    blob["model_state_dict"] = _fno.dequant8(blob["model_state_dict"])
    cfg = blob.get("config", {})
    p = float(cfg.get("dropout", 0.0))
    m = importlib.import_module("models_alt").build(
        blob.get("arch", "fno"), blob.get("width", cfg.get("width", 96)),
        blob.get("var_head", True), 0, blob.get("modes_y", 8), blob.get("modes_x", 12),
        blob.get("n_layers", 4), dropout=p).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    if force_train:
        if p <= 0.0:
            raise SystemExit(f"{path}: dropout 0 in its config -- MC dropout is a no-op on it")
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout2d):
                mod.train()
    m.self_norm = bool(cfg.get("norm", "real") == "self")
    return m, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--mc", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="stochastic passes PER checkpoint; 1 with --keep-det is the control")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-score", type=float, default=93.0586,
                    help="board time_score for the `final` projection; the default is R18's")
    args = ap.parse_args()

    import torch
    from _fno import predict_fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    f = np.float32(args.f)

    det = [load_with_dropout(c, dev, False)[0] for c in args.ckpt]
    sto, p = zip(*[load_with_dropout(c, dev, True) for c in args.ckpt])
    sto = list(sto)
    print(f"{len(args.ckpt)} checkpoint(s), dropout {p[0]:.2f}, alpha {args.alpha}, "
          f"band refit per row at {args.f}x")

    def infer(models, X):
        uv = predict_fno(models, X, dev, alpha=args.alpha)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def row(label, models, mask_s, mask_u, tag):
        torch.manual_seed(args.seed)
        Pf, Yf = infer(models, ds["x"])[mask_s], ds["y"][mask_s]
        Pu, Yu = infer(models, du["x"])[mask_u], du["y"][mask_u]
        Af, af, sf, _ = prep(Pf, Yf)
        Au, _, su, nu = prep(Pu, Yu)
        band = optimal_band(f * Af, af, sf)
        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        mv = sc.mvpe_rel_l2(Pu, Yu)
        dm, tk, mvv = sps9.sample_metrics(Pu, Yu)
        a_b = sps9.weights_from_metrics(f * dm, f * tk, f * mvv)
        s = sps_score(f * Au, a_b, su, nu, band)
        four = [sc.score_error(r * f), sc.score_error(k * f), sc.score_error(mv * f), s]
        fin = final_v9.predict(four[:3] + [args.time_score, four[3]])[0]
        print(f"{label:34s} {len(models):3d} | {r:7.4f} {k:7.4f} {mv:7.4f} {s:6.2f} | "
              f"{np.mean(four):7.3f} {fin:7.3f}")
        return fin

    for tag, mask_s, mask_u in (
            (f"AoA {args.aoa} proxy",
             np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa),
             np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)),
            ("full split", np.ones(len(ds["ids"]), bool), np.ones(len(du["ids"]), bool))):
        hdr = (f"{'configuration':34s} {'fwd':>3s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} "
               f"{'sps':>6s} | {'mean4':>7s} {'final':>7s}")
        print(f"\n--- {tag} ---")
        print(hdr)
        print("-" * len(hdr))
        base = row("deterministic (dropout OFF)", det, mask_s, mask_u, tag)
        for n in args.mc:
            fin = row(f"MC dropout x{n}", sto * n, mask_s, mask_u, tag)
            print(f"{'':34s} {'':3s} | vs deterministic: {fin - base:+.3f} final")


if __name__ == "__main__":
    main()
