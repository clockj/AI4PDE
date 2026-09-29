"""Stream one regime's trajectories through ONE extracted Track-2 package, on the real code path, and record every window.

    python ood_stream.py --pkg RUN/pkg/control --regime A20 --out RUN/json/control.json [--causal-check]

The loop is `Track_2/stream_eval.run` from the scratch worktree (the platform's protocol: batch 1, time order,
reset_ttt_state at each trajectory, the revealed target cached after the timed call and normalised with the TARGET
statistics unless --prev-space input). One package per process (several FrozenEnsemble objects crash).

Output json: the package's baked knobs and the live state flags (r1 / d1c / vc / fw actually active), the load line,
and per window: trajectory, time id, index k in its stream, the kit's per-window rel / tke / mvpe errors, the SPS accuracy
factor a, containment sum con, scored and inside counts, step time, and sha1 of the window's pred / lower / upper
(bit-identity checks between variants). `ood_report.py` scores any subset from these.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_lib  # noqa: E402
import ood_score  # noqa: E402

sys.path.insert(0, os.path.join(ood_lib.WT, "Track_2"))
import stream_eval  # noqa: E402  (the worktree's harness; imports t2_common)
import t2_common as t2  # noqa: E402

KNOBS = ["ALPHA", "JOINT_GAIN", "CENTRE_ALPHA", "AMP", "FUSE", "SIGPOW", "R1", "R1_N0", "R1_ASYM", "R1_TSM", "R1_BLK_N0",
         "R1_BLK", "D1C_N0", "D1C_EB", "D1C_EB_N0", "D1C_LEAD_N0", "VC_N0", "VC_CLIP", "FW_N0", "PREV_SPACE", "STATS_MASK",
         "HIST_BETA", "ARENA_STEPS", "NUMPY_OUT", "FFNO_PATH", "SPEC_PATH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--causal-check", action="store_true")
    ap.add_argument("--prev-space", choices=["target", "input"], default="target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None,
                    help="directory: write P, L, U, Y (physical, u and v, float32 .npy), the input's zero state per element "
                         "(ztrail = number of trailing input frames that are the exact physical zero, 0..20; zany = zero in "
                         "any input frame; uint8 / bool, per channel), the stream's centre field C when the package keeps one "
                         "(gC, after D1c), and the window index -- for the arm-I7 screen (ood_i7.py)")
    ap.add_argument("--reblank", default=None, choices=["m1", "m2", "m3"],
                    help="re-blank every trajectory with a synthetic time-constant mask (ood_reblank.py) in inputs and targets")
    a = ap.parse_args()

    import torch
    ev, groups, _ = ood_lib.regime(a.regime)
    t0 = time.time()
    stream = t2.build_stream(ev)
    reblank = None
    if a.reblank:
        import ood_reblank
        stream, reblank = ood_reblank.apply(stream, a.reblank)
        mp = [v["mask_px"] for v in reblank.values()]; nw = [v["mask_new_px"] for v in reblank.values()]
        print(f"[ood_stream] re-blanked with {a.reblank}: mask {min(mp)}-{max(mp)} px, newly blanked {min(nw)}-{max(nw)} px "
              f"of {32 * 64}", flush=True)
    print(f"[ood_stream] {a.regime}: {len(stream)} steps over {len(ev)} trajectories  (read {time.time() - t0:.0f}s)",
          flush=True)

    pkg = Path(a.pkg).resolve()
    mod = stream_eval.import_submission(pkg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = mod.get_ttt_model(str(pkg), a.device)
    load_line = buf.getvalue().strip()
    print(load_line, flush=True)
    knobs = {k: repr(getattr(mod, k, None)) for k in KNOBS}
    active = dict(r1=getattr(model, "r1", None) is not None, d1c=bool(getattr(model, "d1c", False)),
                  d1x=bool(getattr(model, "d1x", False)), vc=bool(getattr(model, "vc", False)),
                  fw=bool(getattr(model, "fw", False)), groups=len(model.models),
                  graph=getattr(model, "graph", None) is not None, amp=repr(getattr(mod, "AMP_ACTIVE", None)))
    print(f"[ood_stream] active {active}", flush=True)
    for knob, flag in (("FW_N0", "fw"), ("VC_N0", "vc"), ("D1C_N0", "d1c")):
        if float(getattr(mod, knob, 0.0) or 0.0) > 0 and not active[flag]:
            print(f"[ood_stream] WARNING: {knob} > 0 but the correction is NOT active (FAMW needs exactly two fused groups; "
                  f"this package has {active['groups']})", flush=True)

    causal = None
    if a.causal_check:
        n, worst = stream_eval.causal_check(model, stream, a.device, prev_space=a.prev_space)   # SystemExit on failure
        causal = dict(steps=n, max_abs_diff=worst)
        print(f"[ood_stream] CAUSALITY OK: {n} steps bit-identical up to each cut (max |diff| {worst:.1e})", flush=True)

    cap = []
    if a.dump and getattr(model, "gC", None) is not None:
        # the centre the package used for this window (target-normalised, after D1c): read right after each step.
        # Only for the dump run; the extra synchronize and copy add to the recorded step time.
        _orig = model.ttt_step

        def _step(xn, prev):
            out = _orig(xn, prev)
            torch.cuda.synchronize()
            cap.append(model.gC.detach().float().cpu().numpy()[0].copy())
            return out
        model.ttt_step = _step
    fwc = []
    if active["fw"]:
        # FAMW's weight of family B (the second fused group) used for THIS window, read right after the step (gW is set
        # inside the step from past windows only). Mean over pixels, per channel. The extra synchronize adds to step time.
        _orig_fw = model.ttt_step

        def _step_fw(xn, prev):
            out = _orig_fw(xn, prev)
            torch.cuda.synchronize()
            fwc.append(model.gW.detach().double().mean((0, 1, 2, 3)).cpu().numpy().tolist())
            return out
        model.ttt_step = _step_fw
    P, L, U, Y, T = stream_eval.run(model, stream, a.device, prev_space=a.prev_space)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        X = np.stack([s[2][..., :2] for s in stream])                     # the PHYSICAL input the harness normalised
        # The test a package can run: |x_norm - z_c| < 1e-6 in INPUT-normalised units, i.e. |x| < 1e-6 * STD_IN[c] physical
        # (9.7e-8 m/s for u, 1.6e-8 for v). The release holds ~1100 of 146 M values per channel in (0, 1e-12) and none in
        # [1e-12, 1e-8), so this equals the exact test up to those denormal-size values (counted and saved below).
        tolp = (1e-6 * t2.STD_IN[:2]).astype(np.float32)
        z = np.abs(X) < tolp
        n_diff = int((z & (X != 0.0)).sum())
        ztrail = np.zeros(z.shape[:1] + z.shape[2:], np.uint8)            # (N, H, W, 2)
        run_ = np.ones(ztrail.shape, bool)
        for f in range(z.shape[1] - 1, -1, -1):                           # count trailing zero frames
            run_ &= z[:, f]
            ztrail += run_
        np.save(os.path.join(a.dump, "P.npy"), P[..., :2]); np.save(os.path.join(a.dump, "L.npy"), L[..., :2])
        np.save(os.path.join(a.dump, "U.npy"), U[..., :2]); np.save(os.path.join(a.dump, "Y.npy"), Y[..., :2])
        np.save(os.path.join(a.dump, "ztrail.npy"), ztrail); np.save(os.path.join(a.dump, "zany.npy"), z.any(1))
        if cap:
            C = np.stack(cap)[..., :2].astype(np.float64) * t2.STD_TGT[:2] + t2.MEAN_TGT[:2]
            np.save(os.path.join(a.dump, "C.npy"), C.astype(np.float32))
        names_ = [s[0] for s in stream]
        np.save(os.path.join(a.dump, "traj.npy"), np.array(names_))
        json.dump({"input_zero_tol_normalised": 1e-6, "input_zero_tol_physical": tolp.tolist(),
                   "input_elements_zero_by_tol_but_not_exact": n_diff}, open(os.path.join(a.dump, "meta.json"), "w"))
        print(f"[ood_stream] dumped {len(P)} windows to {a.dump} (centre field {'yes' if cap else 'not kept by this package'}; "
              f"{n_diff} input elements zero by tolerance but not exactly)",
              flush=True)
    if L is None:
        raise SystemExit("the package supplied no bounds")
    h = lambda x: [hashlib.sha1(np.ascontiguousarray(w).tobytes()).hexdigest()[:16] for w in x]   # noqa: E731
    rec = ood_score.window_records(P, L, U, Y)
    chk = ood_score.selfcheck(P, L, U, Y, rec)
    bounds_ok = bool(np.isfinite(L).all() and np.isfinite(U).all() and (L <= P).all() and (P <= U).all())
    names = [s[0] for s in stream]
    k = []
    for i, nm in enumerate(names):
        k.append(0 if (i == 0 or names[i - 1] != nm) else k[-1] + 1)
    out = dict(pkg=str(pkg), variant=pkg.name + (f"__{a.reblank}" if a.reblank else ""), reblank=a.reblank,
               reblank_stats=reblank, regime=a.regime.upper(), groups=groups, prev_space_harness=a.prev_space,
               knobs=knobs, active=active, load_line=load_line, causal=causal, selfcheck=chk, bounds_ok=bounds_ok,
               step_ms_mean=float(np.mean(T) * 1e3), step_ms_median=float(np.median(T) * 1e3),
               gpu=torch.cuda.get_device_name() if a.device.startswith("cuda") else "cpu",
               slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
               windows=dict(traj=names, t=[int(s[1]) for s in stream], k=k,
                            **{key: np.asarray(v).tolist() for key, v in rec.items()},
                            ms=(np.asarray(T) * 1e3).tolist(), hP=h(P), hL=h(L), hU=h(U),
                            **({"fw_w": fwc, "fw_w0": float(model.fw0)} if fwc else {})))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    for g, ts in groups.items():
        s = ood_score.aggregate(rec, np.isin(names, ts))
        print(f"  {g:7s} n {s['n']:4d}  rel {s['rel']:.3f} tke {s['tke']:.3f} mvpe {s['mvpe']:.3f} sps {s['sps']:.3f} "
              f"C {s['C']:.3f} cov {s['cov']:.2f}  composite {s['composite']:.4f}", flush=True)
    print(f"  selfcheck vs kit {chk}  bounds_ok {bounds_ok}  step {out['step_ms_mean']:.3f} ms (median "
          f"{out['step_ms_median']:.3f})\n  wrote {a.out}", flush=True)
    if max(chk.values()) > 1e-6 or not bounds_ok:     # kit metrics are float32 means: ~1e-9 rounding
        raise SystemExit("SELFCHECK FAILED")


if __name__ == "__main__":
    main()
