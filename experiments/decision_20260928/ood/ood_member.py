"""Single twins against their seed-matched twins on val_unseen (gate R1, one level down). GPU.

    python ood_member.py OUT.json A.pth:B.pth ...

Each checkpoint alone through the repository's `_fno.predict_fno` at alpha 0.80 (the shipped compose; JOINT_GAIN at the
tree's default, as fit_band.py runs it), on Track_1/val_unseen.npz (the 8 VAL_TRAJ trajectories, 336 windows); the kit's
per-window rel, tke, mvpe errors, averaged, for all 8 and for the AoA 15-20 slice.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_lib  # noqa: E402
import ood_score  # noqa: E402

sys.path.insert(0, os.path.join(ood_lib.WT, "Track_1", "submit_fno"))


def main():
    import torch
    import _fno
    out = sys.argv[1]
    d = np.load(os.path.join(ood_lib.WT, "Track_1", "val_unseen.npz"))
    X, Y, ids = d["x"], d["y"], d["ids"]
    hi = np.array([int(str(i).split(":")[0].split("_")[1]) in (15, 20) for i in ids])
    sc = ood_score.sc
    res = {}
    for pair in sys.argv[2:]:
        for p in pair.split(":"):
            if p in res:
                continue
            m = _fno.load_any(p, "cuda")
            uv = _fno.predict_fno([m], X, "cuda", alpha=0.80)
            P = np.concatenate([uv[..., :2], np.zeros_like(uv[..., :1])], -1).astype(np.float32)
            r, t, v = sc.rel_l2_per_sample(P, Y, 2), sc.tke_rel_l2_per_sample(P, Y, 2), sc.mvpe_rel_l2_per_sample(P, Y)
            res[p] = {"all8": dict(rel=float(r.mean()), tke=float(t.mean()), mvpe=float(v.mean())),
                      "a15_20": dict(rel=float(r[hi].mean()), tke=float(t[hi].mean()), mvpe=float(v[hi].mean()))}
            del m
            torch.cuda.empty_cache()
            print(os.path.basename(p), res[p], flush=True)
    rows = []
    for pair in sys.argv[2:]:
        a, b = pair.split(":")
        rows.append(dict(a=a, b=b, **{f"{s}_{k}": res[a][s][k] - res[b][s][k] for s in ("all8", "a15_20") for k in ("rel", "tke", "mvpe")}))
    json.dump(dict(members=res, pairs=rows), open(out, "w"), indent=1)
    for r in rows:
        print(f"{os.path.basename(r['a'])} - {os.path.basename(r['b'])}: all8 rel {r['all8_rel']:+.4f} tke {r['all8_tke']:+.4f} "
              f"mvpe {r['all8_mvpe']:+.4f} | a15_20 rel {r['a15_20_rel']:+.4f} tke {r['a15_20_tke']:+.4f} mvpe {r['a15_20_mvpe']:+.4f}")


if __name__ == "__main__":
    main()
