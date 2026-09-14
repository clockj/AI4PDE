"""Re/AoA ORACLE diagnostic (plan_20260914.md section 4, wave 2). How much held-out error does
knowing the TRUE (Re, AoA) of a window remove? DIAGNOSTIC ONLY: predict() is not given the
condition, so nothing measured here can ship.

    python Track_1/screen_oracle.py --oracle SCREEN/w2/w2_oracle_fno_s1.pth \
        --ctrl SCREEN/w1/w1_fno_ctrl_s1.pth [--arm w2_oracle_fno] [--out JSON] [--no-ledger]

The two checkpoints are the SAME recipe and seed (the screen's FNO control: cmb3sim minus sim,
held out = model.VAL_TRAJ, seed 1) and differ only in `train2.py --oracle-cond`. Both are rebuilt
as `models_alt.FNO2d` and scored identically through `_fno.predict_fno` -- the real member-average
+ compose path at ALPHA 0.60 -- with the oracle's true condition fed in window order (a stateful
wrapper that advances with predict_fno's batches and asserts it consumed exactly N rows).
Metrics on val_unseen, full split and AoA 15-20: rel / tke / mvpe, and sps with ens_v9's band
(p = 0, solved on val_seen AoA 15-20 at the 1.39x stretch, errors and accuracy factors stretched
1.39x). Two further rows: the oracle with its conditions SHUFFLED across windows (if that is no
worse than the true condition, the model ignores it), and with a zero condition.

Closing rule (main 09-14, scout lit A.7): if knowing the condition removes under ~2 % of rel-L2,
the conditioning idea closes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

import _fno  # noqa: E402
import models_alt  # noqa: E402
import sps9  # noqa: E402
from sps9 import optimal_band, prep, sc, sps_score  # noqa: E402

ALPHA, F_BOARD, AOA_HI = 0.60, 1.39, (15, 20)
OUT_DEFAULT = os.path.join(HERE, "plan", "screen_oracle_20260914.json")
CLOSE_PCT = 2.0


class CondFeed(torch.nn.Module):
    """Hands predict_fno's successive batches their rows of a fixed (N, 2) condition table."""

    def __init__(self, model: torch.nn.Module, cond: torch.Tensor):
        super().__init__()
        self.m, self.cond, self.pos = model, cond, 0
        self.self_norm = False

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        c = self.cond[self.pos:self.pos + len(prev)]
        assert len(c) == len(prev), "condition feed out of step with predict_fno's batches"
        self.pos += len(prev)
        return self.m(prev, cond=c)


def load_fno(path: str, dev: str, cond_dim: int):
    blob = torch.load(path, map_location="cpu")
    cfg = blob["config"]
    assert cfg["arch"] == "fno", f"{path}: arch {cfg['arch']}"
    assert bool(cfg.get("oracle_cond", False)) == (cond_dim > 0), \
        f"{path}: oracle_cond={cfg.get('oracle_cond')} but cond_dim={cond_dim}"
    m = models_alt.build("fno", cfg["width"], cfg["var_head"], 0, cfg["modes_y"], cfg["modes_x"],
                         cfg["n_layers"], var_mlp=cfg["var_mlp"], lift_mlp=cfg["lift_mlp"],
                         point_mlp=cfg.get("point_mlp", False),
                         var_hidden=cfg.get("var_hidden", 128),
                         dropout=cfg.get("dropout", 0.0), cond_dim=cond_dim)
    m.load_state_dict(blob["model_state_dict"])                         # strict
    return m.to(dev).eval(), cfg


def cond_of(ids) -> np.ndarray:
    stems = [str(i).split(":")[0].split("_") for i in ids]
    return models_alt.oracle_cond_features([float(s[0]) for s in stems],
                                           [float(s[1]) for s in stems])


def infer(model, X: np.ndarray, dev: str, cond: np.ndarray | None) -> np.ndarray:
    m = model if cond is None else CondFeed(model, torch.from_numpy(cond).to(dev))
    uv = _fno.predict_fno([m], X, dev, alpha=ALPHA, pad_batch=False, amp=None)
    if cond is not None:
        assert m.pos == len(X), f"fed {m.pos} condition rows for {len(X)} windows"
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)


def score(Ps, Ys, hs, Pu, Yu, hu) -> dict:
    """ens_v9's band (p = 0, val_seen AoA 15-20, 1.39x) applied to the full split and the proxy."""
    Af, af, sf, _ = prep(Ps[hs], Ys[hs])
    band = optimal_band(np.float32(F_BOARD) * Af, af, sf)
    out = {}
    for tag, sel in (("full", slice(None)), ("aoa1520", hu)):
        P, Y = Pu[sel], Yu[sel]
        A, _, scored, n = prep(P, Y)
        dm, tk, mv = sps9.sample_metrics(P, Y)
        a_b = sps9.weights_from_metrics(F_BOARD * dm, F_BOARD * tk, F_BOARD * mv)
        out[tag] = dict(rel=float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                        tke=float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))),
                        mvpe=float(sc.mvpe_rel_l2(P, Y)),
                        sps=float(sps_score(np.float32(F_BOARD) * A, a_b, scored, n, band)),
                        n=int(len(P)))
    return out


def removal(ctrl: dict, arm: dict) -> dict:
    return {sp: {m: round(100.0 * (ctrl[sp][m] - arm[sp][m]) / ctrl[sp][m], 3)
                 for m in ("rel", "tke", "mvpe")} | {"sps": round(arm[sp]["sps"] - ctrl[sp]["sps"], 4)}
            for sp in ("full", "aoa1520")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--arm", default="w2_oracle_fno")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--gpu-h", type=float, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    aoa = lambda d: np.array([int(str(i).split(":")[0].split("_")[1]) for i in d["ids"]])  # noqa: E731
    hs, hu = np.isin(aoa(ds), AOA_HI), np.isin(aoa(du), AOA_HI)
    cs, cu = cond_of(ds["ids"]), cond_of(du["ids"])
    rng = np.random.default_rng(0)
    ps, pu = rng.permutation(len(cs)), rng.permutation(len(cu))

    oracle, ocfg = load_fno(args.oracle, dev, cond_dim=2)
    ctrl, ccfg = load_fno(args.ctrl, dev, cond_dim=0)
    for k in ("width", "n_layers", "modes_y", "modes_x", "dropout", "lam_cov", "lam_grad",
              "lam_tgrad", "lam_v", "lam_t", "epochs", "seed", "sim", "all_data", "exclude"):
        if ocfg.get(k) != ccfg.get(k):
            print(f"[warn] recipe differs on {k}: oracle {ocfg.get(k)} vs ctrl {ccfg.get(k)}")

    rows = {}
    rows["ctrl"] = score(infer(ctrl, ds["x"], dev, None), ds["y"], hs,
                         infer(ctrl, du["x"], dev, None), du["y"], hu)
    rows["oracle_true"] = score(infer(oracle, ds["x"], dev, cs), ds["y"], hs,
                                infer(oracle, du["x"], dev, cu), du["y"], hu)
    rows["oracle_shuffled"] = score(infer(oracle, ds["x"], dev, cs[ps]), ds["y"], hs,
                                    infer(oracle, du["x"], dev, cu[pu]), du["y"], hu)
    rows["oracle_zero"] = score(infer(oracle, ds["x"], dev, np.zeros_like(cs)), ds["y"], hs,
                                infer(oracle, du["x"], dev, np.zeros_like(cu)), du["y"], hu)
    rem = {k: removal(rows["ctrl"], rows[k]) for k in ("oracle_true", "oracle_shuffled", "oracle_zero")}
    film_norm = float(sum(f[-1].weight.detach().norm() ** 2 for f in oracle.cond_film) ** 0.5)
    best_rel = max(rem["oracle_true"]["full"]["rel"], rem["oracle_true"]["aoa1520"]["rel"])
    verdict = (f"CLOSE the conditioning idea: the true (Re, AoA) removes {best_rel:.2f} % of rel-L2 "
               f"(< {CLOSE_PCT} %)" if best_rel < CLOSE_PCT else
               f"OPEN: the true (Re, AoA) removes {best_rel:.2f} % of rel-L2 (>= {CLOSE_PCT} %); a "
               f"legal in-window estimator of the condition may be worth building")
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), oracle=os.path.abspath(args.oracle),
               ctrl=os.path.abspath(args.ctrl), alpha=ALPHA, rows=rows, removal_pct=rem,
               film_last_weight_norm=round(film_norm, 5), verdict_rule=verdict,
               features="models_alt.oracle_cond_features: ((ln Re - 9.3)/0.6, (AoA - 10)/7.07)",
               job=os.environ.get("SLURM_JOB_ID"), eval_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{'row':18s} {'split':8s} {'rel':>8s} {'tke':>8s} {'mvpe':>8s} {'sps':>7s}")
    for k, r in rows.items():
        for sp in ("full", "aoa1520"):
            print(f"{k:18s} {sp:8s} {r[sp]['rel']:8.5f} {r[sp]['tke']:8.5f} {r[sp]['mvpe']:8.5f} "
                  f"{r[sp]['sps']:7.3f}")
    print("error removed vs the control, % (sps: points):", json.dumps(rem, indent=1))
    print(f"FiLM last-layer weight norm {film_norm:.4f}")
    print(verdict)
    print(f"wrote {args.out}")

    if not args.no_ledger:
        import screen_eval
        screen_eval.upsert(dict(
            date=time.strftime("%Y-%m-%d"), wave=2, arm=args.arm,
            backbone="fno w96 L8 m8x12 do0.45 var_mlp lift_mlp + oracle FiLM(true Re, AoA)",
            family="fno", role="diagnostic", status="done", gpu_h=args.gpu_h,
            params_M=round(sum(p.numel() for p in oracle.parameters()) / 1e6, 4),
            epochs=ocfg.get("epochs"), dropout=ocfg.get("dropout"),
            ms_b64_fp16=None, ms_b64_fp32=None, ms_b1_fp16=None, ms_ratio_fno=None,
            single_full={k: round(v, 6) for k, v in rows["oracle_true"]["full"].items() if k != "n"},
            single_aoa1520={k: round(v, 6) for k, v in rows["oracle_true"]["aoa1520"].items() if k != "n"},
            proj_dfinal_t1_standalone=None, proj_dfinal_t1_partner=None,
            proj_dfinal_t2_partner=None, verdict="pending",
            note="DIAGNOSTIC ONLY (never ships): the FNO control recipe with FiLM on the true "
                 "(Re, AoA); compared with w1_fno_ctrl (same recipe and seed)",
            ctrl_full=rows["ctrl"]["full"], ctrl_aoa1520=rows["ctrl"]["aoa1520"],
            oracle_detail=rows, removal_pct=rem, verdict_rule=verdict,
            ckpt=[os.path.abspath(args.oracle)], ctrl_ckpt=os.path.abspath(args.ctrl),
            oracle_json=os.path.abspath(args.out), eval_job=os.environ.get("SLURM_JOB_ID")))
        print(f"upserted {args.arm} into {screen_eval.LEDGER}")


if __name__ == "__main__":
    main()
