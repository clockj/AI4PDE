"""Member-set evaluations for a screen arm (plan section 4; main's chunk-2b request for Swin).

    python Track_1/screen_sets.py --arm w1_swin_unet \
        --set "S1=BASE" "S2=BASE,SCREEN/w1/w1_swin_unet_s1.pth" "S4=SCREEN/w1/w1_swin_unet_s1.pth,VIT2" \
        [--mirror OTHER_ARM]

Set tokens, comma-separated: FNO3 (the three FNO cmb3sim twins), VIT2 (the two ViT twins), BASE
(= FNO3 + VIT2), or checkpoint paths. For every set, at ALPHA 0.60 and always through the real
instruments:
  ens_v9.py --aoa 15 20           its own band (p = 0, val_seen AoA 15-20 at 1.39x), AoA 15-20 report
  ens_v9.py --aoa 0 5 10 15 20    the same on the full split
  fit_band.py --band-p 0.45       the shipping band, scored on the full split and on AoA 15-20
                                  (screen_eval.eval_set: rel / tke / mvpe local, sps at 1.39x)
Results are merged into the arm's row under `ens_sets[<name>]` (and into --mirror's row) and printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import screen_eval as se  # noqa: E402


def expand(tokens: str) -> list[str]:
    out = []
    for t in (x.strip() for x in tokens.split(",") if x.strip()):
        if t == "FNO3":
            out += se.BASE_FNO
        elif t == "VIT2":
            out += se.BASE_VIT
        elif t == "BASE":
            out += se.BASE_FNO + se.BASE_VIT
        else:
            p = os.path.abspath(t)
            if not os.path.exists(p):
                raise SystemExit(f"missing checkpoint {p}")
            out.append(p)
    if len(set(out)) != len(out):
        raise SystemExit(f"duplicate member in {tokens}")
    return out


def row_of(arm: str) -> dict:
    if not os.path.exists(se.LEDGER):
        return {}
    with open(se.LEDGER) as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("arm") == arm:
                return json.loads(line)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--set", nargs="+", required=True, help="NAME=TOKENS")
    ap.add_argument("--mirror", default=None)
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {}
    for spec in args.set:
        name, tokens = spec.split("=", 1)
        sets[name] = expand(tokens)
    logdir = os.path.join(se.SCREEN, "eval", f"sets_{args.arm}")
    os.makedirs(logdir, exist_ok=True)
    t0 = time.time()

    ens = {}
    for tag, aoa in (("aoa1520", ["15", "20"]), ("full", ["0", "5", "10", "15", "20"])):
        txt = se.run_main("ens_v9", ["--sets", *[",".join(v) for v in sets.values()],
                                     "--alpha", se.ALPHA, "--aoa", *aoa],
                          log=os.path.join(logdir, f"ens_v9_{tag}_{'_'.join(sets)}.txt"))
        ens[tag] = se.parse_ens_v9(txt, len(sets))
    data = se.Data()
    res = {}
    for i, (name, paths) in enumerate(sets.items()):
        r = se.eval_set(paths, data, dev, se.ALPHA)
        res[name] = dict(members=[os.path.basename(p) for p in paths], n=len(paths),
                         ens_v9_aoa1520=ens["aoa1520"][i], ens_v9_full=ens["full"][i],
                         p045_full=r["full"], p045_aoa1520=r["aoa1520"], band=r["band"],
                         date=time.strftime("%Y-%m-%d %H:%M"),
                         job=os.environ.get("SLURM_JOB_ID"))
    for arm in [args.arm] + ([args.mirror] if args.mirror else []):
        merged = dict(row_of(arm).get("ens_sets", {}))
        merged.update(res)
        se.upsert({"arm": arm, "ens_sets": merged})

    print(f"{'set':6s} {'n':>2s} | {'ens_v9 AoA15-20: rel':>21s} {'tke':>7s} {'mvpe':>7s} {'sps':>6s} | "
          f"{'ens_v9 full: rel':>17s} {'tke':>7s} {'mvpe':>7s} {'sps':>6s} | "
          f"{'p0.45 full: rel':>16s} {'tke':>7s} {'mvpe':>7s} {'sps':>7s} | {'p0.45 AoA: sps':>14s}")
    for name, r in res.items():
        a, f, pf, pa = r["ens_v9_aoa1520"], r["ens_v9_full"], r["p045_full"], r["p045_aoa1520"]
        print(f"{name:6s} {r['n']:2d} | {a['rel']:21.4f} {a['tke']:7.4f} {a['mvpe']:7.4f} {a['sps']:6.2f} | "
              f"{f['rel']:17.4f} {f['tke']:7.4f} {f['mvpe']:7.4f} {f['sps']:6.2f} | "
              f"{pf['rel']:16.5f} {pf['tke']:7.5f} {pf['mvpe']:7.5f} {pf['sps']:7.3f} | {pa['sps']:14.3f}")
        print(f"         members: {', '.join(r['members'])}")
    print(f"merged into {se.LEDGER} under {args.arm}{' and ' + args.mirror if args.mirror else ''} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
