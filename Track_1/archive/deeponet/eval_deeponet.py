"""Score the trained DeepONet on the same 336-window validation set as the official baselines."""
from __future__ import annotations

import importlib.util
import sys
import time

import numpy as np
import torch

from deeponet_baseline import MEAN, STD, DeepONet

SCORING = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7/scoring.py"
spec = importlib.util.spec_from_file_location("sc", SCORING)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def main() -> None:
    val = sys.argv[1] if len(sys.argv) > 1 else "val.npz"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "deeponet.pth"
    d = np.load(val)
    X, Y = d["x"], d["y"]

    dev = "cuda"
    model = DeepONet().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state_dict"])
    model.eval()

    mean = torch.from_numpy(MEAN).to(dev)
    std = torch.from_numpy(STD).to(dev)
    outs, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 16, ..., :2])).to(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            yb = model((xb - mean) / std) * std + mean
            torch.cuda.synchronize()
            t += time.perf_counter() - t0
            outs.append(yb.float().cpu().numpy())
    uv = np.concatenate(outs)
    P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
    tstep = t / len(X)

    c = 2
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, c)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, c)))
    m = sc.mvpe_rel_l2(P, Y)
    sps = sc.score_sps(sc.aggregate_sps(P, Y, c)[0])
    sps_tuned = max(sc.score_sps(sc.aggregate_sps(P, Y, c, lower=P - h, upper=P + h)[0])
                    for h in [0.003, 0.005, 0.0075, 0.01, 0.015])
    s = [sc.score_error(r), sc.score_error(k), sc.score_error(m), sc.score_time(tstep), sps]
    print(f"DeepONet  relL2={r:.3f} TKE={k:.3f} MVPE={m:.3f} | "
          f"rel={s[0]:.1f} tke={s[1]:.1f} mvpe={s[2]:.1f} time={s[3]:.1f} sps={sps:.1f} "
          f"| FINAL={np.mean(s):.2f}  +band={np.mean(s[:4] + [sps_tuned]):.2f}  ({tstep*1000:.2f} ms/win)")


if __name__ == "__main__":
    main()
