"""Teacher labels for knowledge distillation (forge, 2026-09-17).

Runs an ENSEMBLE of trained members over every training window and writes what the shipping
inference path averages -- the member-averaged raw field and the member-averaged sigma, BEFORE
`compose` -- so a single student can be trained to reproduce the ensemble instead of the data.

Why this exists.  The board charges FLOPs (1.75 board ms per 4060-equivalent local ms on Track 1)
and ~1.15 ms per invocation.  R25 is six members in two invocations at 5.076 board ms; a single
student that reproduced the ensemble would ship as ONE invocation at a quarter of the FNO group's
FLOPs.  The ceiling the student has to climb is the held-out gain from K=1 to the full set
(`Track_1/plan/screen_kcurve_20260914.json` and screen.jsonl's S1 set); the time credit is
worth about +0.3 `final` on Track 1, so the student may lose some of it and still win.

WHAT IS STORED, and in WHICH SPACE.  `_fno.predict_fno` normalizes with the fixed real
GaussianNormalizer, runs every member, and averages the raw field and sigma ON THE GPU with the
`n_members` weighting a fused stack carries, all of that BEFORE the fluctuation rescaling
(`compose`) and before the denormalization.  This script reproduces exactly that arithmetic and
stores the result in NORMALIZED space, which is the space the student's own head predicts in:

    mean  (N, 20, 32, 64, 2) float16   the averaged raw field  (`y` in predict_fno)
    sig   (N,  1, 32, 64, 2) float16   the averaged sigma      (`sig` in predict_fno)

Sigma is one map per window, not per frame: the var head emits (B, 2, H, W) and predict_fno
broadcasts it over time.  Members run in fp32 with TF32 OFF -- labels are generated once, and
the fp16 inference path is a shipping optimization, not a label-quality one.

ALIGNMENT.  Rows are in `model.window_index(trajs, stride)` order, which is the row order the
trainer builds from the same (data dir, exclude, phases, stride).  The index itself is saved
next to the labels and `train2.py --teacher` asserts it matches the index it built, so a stride
or split mismatch fails loudly instead of training against shuffled labels.

HELD-OUT DISCIPLINE.  The teacher must not have seen the student's validation trajectories.  The
split is NOT typed here: it is `[]` for `--all-data` and `model.VAL_TRAJ` otherwise, exactly the
rule train2.py applies, and every member's `config` is checked against it before anything runs
(all five held-out twins carry `exclude=None, all_data=False`, i.e. VAL_TRAJ).

    python distill_labels.py --members A.pth B.pth ... --out DIR --stride 2 \
        [--all-data] [--sim <train_sim dir> --sim-stride 5 --sim-exclude-held] [--batch 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unet")   # moved into closed/ 2026-09-27: HERE is still Track_1/unet
SUBMIT = os.path.join(os.path.dirname(HERE), "submit_fno")
sys.path.insert(0, HERE)
sys.path.insert(0, SUBMIT)

import _fno  # noqa: E402  (the shipping inference module: the path of record)
from model import VAL_TRAJ, load_trajectories, trajectory_names, window_index  # noqa: E402

PHASES_ALL = ((0, 0), (0, 1), (1, 0), (1, 1))
T_IN, T_OUT, H, W, C = 20, 20, 32, 64, 2


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def label_set(models, trajs, idx, batch, device, tag):
    """The member-averaged field and sigma, in normalized space, for every row of `idx`.

    Mirrors `_fno.predict_fno` line for line up to but not including compose: the same
    normalizer, the same (B, CH, H, W) time-major layout, the same `n_members` weighting, and
    the same `logstd.clamp(-8, 2).exp()` sigma.  `trajs` are RAW PHYSICAL, as train2.py loads
    them; the normalization happens here.
    """
    n = len(idx)
    mean_a = np.empty((n, T_OUT, H, W, C), np.float16)
    sig_a = np.empty((n, 1, H, W, C), np.float16)
    mean = torch.from_numpy(_fno.MEAN).to(device)
    std = torch.from_numpy(_fno.STD).to(device)
    trajs_t = [torch.from_numpy(d) for d in trajs]
    t0 = time.time()
    for i in range(0, n, batch):
        sel = idx[i:i + batch]
        xb = torch.stack([trajs_t[a][b:b + T_IN] for a, b in sel]).to(device).float()
        prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), T_IN * C, H, W)
        y = sig = None
        total = 0
        for m in models:
            nm = getattr(m, "n_members", 1)
            total += nm
            raw, logstd = m(prev)
            yi = raw.reshape(len(xb), T_OUT, C, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            if logstd is None:
                raise SystemExit("a teacher member has no sigma head; the student's sigma "
                                 "loss would have no target")
            si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
            sig = si if sig is None else sig + si
        y, sig = y / total, sig / total
        mean_a[i:i + batch] = y.cpu().numpy().astype(np.float16)
        sig_a[i:i + batch] = sig.cpu().numpy().astype(np.float16)
        if i % (batch * 50) == 0:
            done = i + len(sel)
            print(f"  {tag} {done}/{n}  {time.time() - t0:.0f}s", flush=True)
    print(f"  {tag} done {n} windows in {time.time() - t0:.0f}s   "
          f"mean |y| {np.abs(mean_a[::97].astype(np.float32)).mean():.5f}  "
          f"sigma median {np.median(sig_a[::97].astype(np.float32)):.5f}", flush=True)
    return mean_a, sig_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--phases", type=int, default=1)
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="NEVER type this; it exists only so the rule can be overridden in a "
                         "test. The split is VAL_TRAJ (or [] with --all-data).")
    ap.add_argument("--sim", default="")
    ap.add_argument("--sim-stride", type=int, default=5)
    ap.add_argument("--sim-exclude-held", action="store_true")
    ap.add_argument("--sim-map", default="scalar")
    ap.add_argument("--sim-frames", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--verify", type=int, default=8,
                    help="check this many windows against _fno.predict_fno(alpha=0); 0 skips")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    excl = [] if a.all_data else (a.exclude if a.exclude is not None else list(VAL_TRAJ))
    os.makedirs(a.out, exist_ok=True)

    print(f"teacher members ({len(a.members)}):", flush=True)
    models, meta = [], []
    for p in a.members:
        ck = torch.load(p, map_location="cpu")
        cfg = ck.get("config", {}) or {}
        mex = [] if cfg.get("all_data") else (cfg.get("exclude") or list(VAL_TRAJ))
        # A teacher that trained on the student's validation trajectories would launder
        # memorized targets into the student's labels; refuse rather than warn.
        if sorted(mex) != sorted(excl):
            raise SystemExit(f"REFUSING: {os.path.basename(p)} was trained excluding {mex}, "
                             f"but these labels are for a run excluding {excl}")
        m = _fno.load_any(p, dev).eval()
        models.append(m)
        meta.append(dict(path=os.path.abspath(p), md5=md5(p), arch=ck.get("arch"),
                         width=ck.get("width"), n_layers=ck.get("n_layers"),
                         seed=cfg.get("seed"), all_data=bool(cfg.get("all_data")),
                         n_members=int(getattr(m, "n_members", 1))))
        print(f"  {os.path.basename(p):34s} {ck.get('arch')} w{ck.get('width')} "
              f"L{ck.get('n_layers')} seed {cfg.get('seed')} all_data {bool(cfg.get('all_data'))}",
              flush=True)

    man = dict(created=time.strftime("%F %T"), members=meta, stride=a.stride,
               phases=a.phases, all_data=bool(a.all_data), exclude=sorted(excl),
               data=os.path.abspath(a.data), space="normalized (MEAN/STD of _fno)",
               note="mean = member-averaged raw field before compose; sig = member-averaged "
                    "exp(clamp(logstd,-8,2)); both float16, rows in window_index order")

    trajs = load_trajectories(a.data, exclude=excl, normalize=False,
                              phases=PHASES_ALL[:a.phases])
    idx = window_index(trajs, a.stride)
    names = trajectory_names(a.data, exclude=excl, phases=PHASES_ALL[:a.phases])
    print(f"real: {len(trajs)} trajectory-phases -> {len(idx)} windows at stride {a.stride}, "
          f"{len(excl)} held out; labels {len(idx) * (T_OUT + 1) * H * W * C * 2 / 1e9:.2f} GB",
          flush=True)
    mean_a, sig_a = label_set(models, trajs, idx, a.batch, dev, "real")
    if a.verify:
        # The labels claim to be `predict_fno`'s pre-compose average, so check them against
        # `predict_fno` itself with ALPHA 0 (which short-circuits compose) on the first few
        # windows. Denormalizing the label must reproduce its output; sigma_out is already
        # physical, so the label times STD must reproduce that.
        k = min(a.verify, len(idx))
        xr = np.stack([np.concatenate(
            [trajs[i][t:t + T_IN], np.zeros((T_IN, H, W, 1), np.float32)], -1)
            for i, t in idx[:k]])
        so = np.empty((k, 1, H, W, C), np.float32)
        yr = _fno.predict_fno(models, xr, dev, alpha=0.0, batch=a.batch, sigma_out=so, amp=None)
        dy = np.abs(yr[..., :C] - (mean_a[:k].astype(np.float32) * _fno.STD + _fno.MEAN)).max()
        ds = np.abs(so - sig_a[:k].astype(np.float32) * _fno.STD).max()
        rms = float(np.sqrt((yr[..., :C] ** 2).mean()))
        print(f"verify vs predict_fno(alpha=0) on {k} windows: field max |diff| {dy:.3e} "
              f"(rms {rms:.4f}), sigma max |diff| {ds:.3e}  "
              f"[fp16 label quantum is ~{rms * 2 ** -11:.1e}]", flush=True)
        if dy > 5e-3 * rms or ds > 5e-3 * float(np.abs(so).mean()):
            raise SystemExit("REFUSING: the labels do not reproduce the inference path")
    np.save(os.path.join(a.out, "real_mean.npy"), mean_a)
    np.save(os.path.join(a.out, "real_sig.npy"), sig_a)
    np.save(os.path.join(a.out, "real_idx.npy"), idx)
    man["real"] = dict(n=int(len(idx)), trajs=len(trajs), names=names)
    del mean_a, sig_a, trajs

    if a.sim:
        from simreal import load_sim_mapped
        sim_trajs = load_sim_mapped(a.sim, a.data, phases=PHASES_ALL[:a.phases],
                                    frames=a.sim_frames,
                                    exclude=excl if a.sim_exclude_held else None,
                                    real_exclude=excl, mode=a.sim_map, with_p=False)
        sidx = window_index(sim_trajs, a.sim_stride)
        print(f"sim: {len(sim_trajs)} trajectories -> {len(sidx)} windows at stride "
              f"{a.sim_stride} (map '{a.sim_map}', sim_exclude_held {a.sim_exclude_held})",
              flush=True)
        smean, ssig = label_set(models, sim_trajs, sidx, a.batch, dev, "sim")
        np.save(os.path.join(a.out, "sim_mean.npy"), smean)
        np.save(os.path.join(a.out, "sim_sig.npy"), ssig)
        np.save(os.path.join(a.out, "sim_idx.npy"), sidx)
        man["sim"] = dict(n=int(len(sidx)), trajs=len(sim_trajs), stride=a.sim_stride,
                          sim=os.path.abspath(a.sim), sim_map=a.sim_map,
                          sim_exclude_held=bool(a.sim_exclude_held), frames=a.sim_frames)

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("wrote", a.out, flush=True)
    for f in sorted(os.listdir(a.out)):
        print(f"  {f:16s} {os.path.getsize(os.path.join(a.out, f)) / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
