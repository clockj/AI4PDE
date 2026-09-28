"""CPU gate for `train2.py --aug-resim` (Decision-phase arm T4, patch `t4_aug_resim.patch`).

Modelled on `Track_1/verify/aug_mask_check.py`. Runs against a tree that carries the patch
(default: the scratch worktree `dec0928/wt_t4`). Two modes:

    python t4_aug_resim_check.py [--tree WT]              # the checks below, CPU, ~2-4 min (reads real data)
    python t4_aug_resim_check.py --cpu-train WT -- ARGS    # run WT's train2.py on the CPU with ARGS

`--cpu-train` executes the tree's own train2.py source with its single `dev = "cuda"` line
replaced by `dev = "cpu"` and nothing else, so an unmodified tree and a patched one can be
compared step for step on a machine without a GPU (the default-path identity check).

Checks:
  1. the parser defaults are --aug-resim 0.0 / --aug-resim-p 0.5, and every call site in main()
     sits under `args.aug_resim > 0.0` (validation under `!= 0.0`) -- a source scan;
  2. the augmentation draws only from its own generator: the global CPU RNG state is
     byte-identical after a call at S = 0.25;
  3. s = 1 reproduces the stored window bit for bit, on real data;
  4. synthetic travelling wave u = U f(x - cUt): the augmented sample at factor s equals the
     analytic field at U -> sU to interpolation error (the number is printed), and is far
     from what an amplitude-only scale (--aug-scale) gives;
  5. a field linear in time is reproduced EXACTLY, with a constant frame step across the
     input/target boundary: input and target are contiguous in resampled time;
  6. the physical zero: whether the zero set of the real data is static in time (reported;
     it is not), how much the union rule grows it, stays exactly 0 under the augmentation, normalizes to exactly -MEAN/STD, and
     an element that is 0 in only one weighted neighbour comes out 0;
  7. windows that would run past the trajectory's end are left un-augmented and counted; the
     fraction of stride-5 training windows that cannot be augmented at s = 1.25 is reported,
     on all 82 trajectories and on the RE regime's training list;
  8. the loss's variance / TKE targets are formed from the augmented `y` (source order in
     main(): the aug block precedes self_norm, compute_loss, std_t(y), tke(y, mul)); and
     std_t of the augmented physical target equals s times that of the resampled one;
  9. every refused combination raises SystemExit with the refusal message, before any data
     is read.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

DEF_TREE = "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/wt_t4"
REAL = "/users/PAS1187/qwu/RealPDE/train_real/train_real"


def cpu_train(tree: str, argv: list[str]) -> None:
    unet = os.path.join(tree, "Track_1", "unet")
    path = os.path.join(unet, "train2.py")
    src = open(path).read()
    line = '    dev = "cuda"\n'
    if src.count(line) != 1:
        raise SystemExit(f"{path}: expected exactly one `dev = \"cuda\"` line")
    src = src.replace(line, '    dev = "cpu"\n')
    sys.path.insert(0, unet)
    os.chdir(unet)
    sys.argv = [path] + argv
    torch_threads = os.environ.get("OMP_NUM_THREADS")
    import torch
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})


def checks(tree: str) -> int:
    import numpy as np
    import torch

    unet = os.path.join(tree, "Track_1", "unet")
    sys.path.insert(0, unet)
    sys.path.insert(0, os.path.join(tree, "Track_1"))
    from model import MEAN, STD, T_IN, T_OUT, load_trajectories, trajectory_names, window_index, std_t
    import train2
    from train2 import aug_resim, resim_frames

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}", flush=True)

    N = T_IN + T_OUT

    # ------------------------------------------------------------ 1. defaults and guards
    src = open(train2.__file__).read()
    m_def = re.search(r'"--aug-resim", type=float, default=([0-9.]+)', src)
    p_def = re.search(r'"--aug-resim-p", type=float, default=([0-9.]+)', src)
    check("parser defaults --aug-resim 0.0, --aug-resim-p 0.5",
          m_def and float(m_def.group(1)) == 0.0 and p_def and float(p_def.group(1)) == 0.5)
    main_src = src[src.index("def main()"):]
    lines = main_src.splitlines()
    uses = [i for i, ln in enumerate(lines)
            if re.search(r"resim_gen|aug_resim\(|rs_hit|rs_short|rs_s\b|rs_rows|rs_fr", ln)]
    unguarded = []
    for i in uses:
        # the chain of enclosing statements: walk up, keeping each line less indented than the last
        ind = len(lines[i]) - len(lines[i].lstrip())
        chain = []
        for k in range(i - 1, -1, -1):
            lk = lines[k]
            if not lk.strip() or lk.lstrip().startswith("#"):
                continue
            indk = len(lk) - len(lk.lstrip())
            if indk < ind:
                chain.append(lk.strip())
                ind = indk
        if not any(c.startswith("if args.aug_resim > 0.0") for c in chain):
            unguarded.append(lines[i].strip())
    check("every use of the augmentation's state in main() is under `if args.aug_resim > 0.0`",
          not unguarded, f"{len(uses)} lines scanned; unguarded: {unguarded}")

    # ------------------------------------------------------------ 2. global RNG untouched
    torch.manual_seed(0)
    fake = [torch.randn(300, 32, 64, 2).abs() + 0.01]
    sel = torch.tensor([[0, t] for t in range(0, 250, 7)])
    st = torch.get_rng_state()
    g = torch.Generator().manual_seed(1)
    rows, fr, sv, ns = aug_resim(fake, sel, 0.25, 0.5, g)
    check("S=0.25 draws nothing from the global CPU generator", torch.equal(st, torch.get_rng_state()),
          f"{len(rows)}/{len(sel)} rows replaced")
    g1 = torch.Generator().manual_seed(1)
    rows1, fr1, sv1, _ = aug_resim(fake, sel, 0.25, 0.5, g1)
    check("the dedicated generator makes the draw reproducible", rows == rows1 and torch.equal(fr, fr1))
    s_on = sv[sv != 1.0]
    check("drawn s lies in [1/1.25, 1.25]", bool(((s_on >= 0.8 - 1e-12) & (s_on <= 1.25 + 1e-12)).all()))

    # ------------------------------------------------------------ 3. s = 1 on real data
    names_all = trajectory_names(REAL, exclude=[])
    trajs = load_trajectories(REAL, exclude=[], normalize=False)
    tt = [torch.from_numpy(d) for d in trajs]
    bad = 0
    for i in range(0, len(tt), 7):
        for t0 in (0, 137, len(tt[i]) - N):
            if not torch.equal(resim_frames(tt[i], t0, 1.0), tt[i][t0:t0 + N]):
                bad += 1
    check("s = 1 reproduces the stored window bit for bit (real data)", bad == 0)
    g = torch.Generator().manual_seed(3)
    sel = torch.tensor([[0, 100], [1, 5]])
    rows, fr, sv, ns = aug_resim(tt, sel, 0.25, 1e-12, g)
    check("rows not drawn (p ~ 0) are not replaced", rows == [] and fr is None and bool((sv == 1).all()))

    # ------------------------------------------------------------ 4. travelling wave
    Tn, Hh, Ww = 400, 32, 64
    xg = torch.arange(Ww, dtype=torch.float64).view(1, 1, Ww)
    yg = torch.arange(Hh, dtype=torch.float64).view(1, Hh, 1)
    c = 0.8                                   # grid cells per frame per unit U
    U = 1.0
    Lx = 40.0

    def field(Uv, tvec, x0=0.0):
        ph = 2 * np.pi * (xg - x0 - c * Uv * tvec.view(-1, 1, 1)) / Lx
        u = Uv * (1.0 + 0.3 * torch.sin(ph) + 0.1 * torch.cos(2 * ph + yg / 5.0))
        v = Uv * 0.2 * torch.cos(ph + yg / 7.0)
        return torch.stack([u, v], -1)

    traj = field(U, torch.arange(Tn, dtype=torch.float64)).float()
    j = torch.arange(N, dtype=torch.float64)
    t0 = 50
    for s in (1.25, 0.8, 1.1):
        got = resim_frames(traj, t0, s).double()
        ana = field(s * U, j, x0=c * U * t0)          # U -> sU, phase origin at t0
        amp_only = s * field(U, t0 + j)               # --aug-scale: amplitude, no time scale
        e = (got - ana).abs().max().item() / ana.abs().max().item()
        e_rms = ((got - ana).pow(2).mean().sqrt() / ana.pow(2).mean().sqrt()).item()
        e_amp = ((amp_only - ana).pow(2).mean().sqrt() / ana.pow(2).mean().sqrt()).item()
        # linear interpolation of a sinusoid of period P frames has max error ~ (pi/P)^2/2 of
        # its amplitude; here P = Lx/(c U) = 50 frames -> ~2e-3 of the 0.3 U harmonic
        check(f"travelling wave, s={s}: equals the analytic U->sU field to interpolation error",
              e < 5e-3, f"max rel err {e:.2e}, rms rel err {e_rms:.2e}; amplitude-only scale "
              f"would be {e_amp:.2e} rms")

    # ------------------------------------------------------------ 5. contiguity, exactness
    lin = (0.3 + 0.01 * torch.arange(Tn, dtype=torch.float64)).view(Tn, 1, 1, 1).expand(Tn, 2, 2, 2)
    for s in (1.25, 0.8, 1.0137):
        got = resim_frames(lin.float(), 33, s).double()
        ana = s * (0.3 + 0.01 * (33 + s * j)).view(N, 1, 1, 1)
        err = (got - ana).abs().max().item()
        steps = got[1:, 0, 0, 0] - got[:-1, 0, 0, 0]
        cross = (steps[T_IN - 1] - steps.mean()).abs().item()
        check(f"linear-in-time field, s={s}: exact to fp32 and contiguous across the in/out boundary",
              err < 2e-6 and cross < 2e-6,
              f"max abs err {err:.1e}; step at frame 19->20 {steps[T_IN - 1].item():.6f} vs mean "
              f"{steps.mean().item():.6f} (s^2*0.01 = {s * s * 0.01:.6f})")

    # ------------------------------------------------------------ 6. the physical zero
    static, sporadic = [], []
    for d in trajs:
        z = (d[..., 0] == 0) & (d[..., 1] == 0)
        za = z.all(0)
        zany = z.any(0)
        static.append(za.mean())
        sporadic.append((zany & ~za).mean())
        # per-channel zeros that are not joint zeros
    one_ch = max(float((((d == 0).any(-1)) & ~((d == 0).all(-1))).mean()) for d in trajs)
    static, sporadic = np.array(static), np.array(sporadic)
    n_spor = int((sporadic > 0).sum())
    print(f"     zero set: static fraction per trajectory {static.min():.4f}..{static.max():.4f} "
          f"(mean {static.mean():.4f}); pixels zero in SOME but not all frames: "
          f"{n_spor}/{len(trajs)} trajectories, max {sporadic.max():.2e} of pixels; "
          f"elements zero in exactly one channel: max {one_ch:.2e} of elements", flush=True)
    worst = [names_all[k] for k in np.argsort(-sporadic)[:3] if sporadic[k] > 0]
    if worst:
        print(f"     trajectories with the most time-varying zeros: {worst}", flush=True)
    check("zero set of the real data characterised (informative; the brief expected it static)", True,
          "static in all trajectories" if n_spor == 0 else
          f"NOT static: {n_spor}/{len(trajs)} trajectories carry pixels that are zero in some frames only")
    zo, za = [], []
    for i in range(0, len(tt), 3):
        d = tt[i]
        for t0 in range(0, len(d) - 50, 40):
            zo.append((d[t0:t0 + N] == 0).float().mean().item())
            for s in (1.25, 0.8):
                za.append((resim_frames(d, t0, s) == 0).float().mean().item())
    print(f"     element zero fraction (u, v): stored windows {np.mean(zo):.4f}, augmented windows "
          f"(s = 1.25 and 0.8, union rule) {np.mean(za):.4f}", flush=True)
    zc = (-MEAN / STD).astype(np.float32)
    worst_ok = True
    for i in range(0, len(tt), 5):
        d = tt[i]
        for t0 in (0, 211, len(d) - 60):
            for s in (1.25, 0.8, 1.17):
                if t0 + s * (N - 1) > len(d) - 1:
                    continue
                got = resim_frames(d, t0, s)
                src_t = t0 + s * j
                k0 = src_t.floor().long()
                k1 = (k0 + 1).clamp(max=len(d) - 1)
                w = (src_t - k0.double()).view(N, 1, 1, 1)
                zmask = (d[k0] == 0) | ((d[k1] == 0) & (w > 0))
                if not bool((got[zmask] == 0).all()):
                    worst_ok = False
                nrm = (got - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
                zz = nrm[zmask.all(-1)]
                if zz.numel() and not torch.equal(zz, torch.from_numpy(zc).expand_as(zz)):
                    worst_ok = False
    check("masked elements stay exactly 0 physical, and normalize to exactly -MEAN/STD", worst_ok)
    d = torch.ones(N, 1, 1, 2)
    d[3, 0, 0, 0] = 0.0                                   # zero in frame 3 only, channel u
    got = resim_frames(d, 0, 0.5)                         # frames at t = 0, .5, 1, ... 19.5
    t_src = 0.5 * j
    hit = (t_src.floor() == 3) | ((t_src.floor() == 2) & (t_src > 2))
    exp_u = torch.where(hit, 0.0, 0.5).float()
    check("an element 0 in one weighted neighbour comes out 0; w=0 ignores the other neighbour",
          torch.equal(got[:, 0, 0, 0], exp_u) and bool((got[:, 0, 0, 1] == 0.5).all()),
          f"u zeroed at source t = {t_src[hit].tolist()}")

    # ------------------------------------------------------------ 7. past the end
    Tlen = 200
    toy = [torch.ones(Tlen, 2, 2, 2)]
    t_edge = [Tlen - N - k for k in range(0, 12)]         # the last 12 starts
    sel = torch.tensor([[0, t] for t in t_edge])
    g = torch.Generator().manual_seed(11)
    rows, fr, sv, ns = aug_resim(toy, sel, 0.25, 1.0, g)
    fits = all(t_edge[r] + float(sv[r]) * (N - 1) <= Tlen - 1 for r in rows)
    rest_one = all(float(sv[r]) == 1.0 for r in range(len(t_edge)) if r not in rows)
    check("samples that would run past the end are left un-augmented (s = 1) and counted",
          fits and rest_one and len(rows) + ns == len(t_edge) and ns > 0,
          f"{len(rows)} replaced, {ns} too close to the end, of {len(t_edge)} edge windows (p = 1)")
    d_edge = torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1).expand(100, 1, 1, 2) + 1
    s_b = (99 - 40) / 39.0                                # t0 + s*39 == T-1 exactly
    got = resim_frames(d_edge, 40, s_b)
    check("a window ending exactly on the last frame is allowed and uses it",
          abs(got[-1, 0, 0, 0].item() - s_b * 100.0) < 1e-3)

    def frac_short(tr, stride, s=1.25):
        idx = window_index(tr, stride)
        lens = np.array([len(tr[i]) for i in idx[:, 0]])
        return float(np.mean(idx[:, 1] + s * (N - 1) > lens - 1)), len(idx)

    f_all, n_all = frac_short(trajs, 5)
    ex_re = None
    p_re = "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ckpt/RE/exclude.txt"
    if os.path.exists(p_re):
        ex_re = open(p_re).read().split()
    tr_re = [d for d, n in zip(trajs, names_all) if ex_re is None or n not in ex_re]
    f_re, n_re = frac_short(tr_re, 5)
    f_all08, _ = frac_short(trajs, 5, 0.8)
    print(f"     stride 5: {f_all:.4%} of {n_all} windows (all 82 trajectories) and {f_re:.4%} of "
          f"{n_re} (RE training list, {len(tr_re)} trajectories) cannot be augmented at s = 1.25; "
          f"at s = 0.8: {f_all08:.4%}", flush=True)
    check("fraction of stride-5 windows that cannot take s = 1.25 is small (< 3 %)", f_re < 0.03)

    # ------------------------------------------------------------ 8. variance / TKE targets
    loop = main_src[main_src.index("for ep in range(start_ep"):]
    order = [loop.find(k) for k in ("= aug_resim(trajs_t", "x, y = self_norm(x, s", "def compute_loss",
                                     "s_true = t_sig if t_sig is not None else std_t(y)",
                                     "l_tke = rel_l2(ke_pred, tke(y, mul))")]
    check("aug block precedes self_norm, compute_loss, std_t(y) and tke(y, mul) in the loop",
          all(o >= 0 for o in order) and order == sorted(order), f"offsets {order}")
    d = tt[0]
    base = resim_frames(d, 100, 1.2) / 1.2
    aug = resim_frames(d, 100, 1.2)
    r = (std_t(aug[None, T_IN:]) - 1.2 * std_t(base[None, T_IN:])).abs().max().item()
    check("std_t of the augmented target is s x that of the resampled one (TKE scales by s^2)",
          r < 1e-6, f"max abs diff {r:.1e}")

    # ------------------------------------------------------------ 9. refusals
    py = sys.executable
    for extra, needle in ((["--hist"], "--hist"), (["--ctx", "40"], "--ctx"),
                          (["--teacher", "/nonexistent"], "--teacher"),
                          (["--subpix", "0.3", "--arch", "fno", "--loss", "mse"], "--subpix"),
                          (["--mixup", "0.2"], "--mixup"), (["--norm", "self"], "--norm self"),
                          (["--aug-scale", "0.2"], "--aug-scale"),
                          (["--aug-resim-p", "0"], "0 < --aug-resim-p"),
                          (["--aug-resim-p", "1.5"], "0 < --aug-resim-p")):
        cmd = [py, os.path.join(unet, "train2.py"), "--out", "/dev/null", "--aug-resim", "0.25"] + extra
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=unet)
        check(f"refuses {' '.join(extra)}", pr.returncode != 0 and needle in pr.stderr
              and "load" not in pr.stdout.lower(), pr.stderr.strip().splitlines()[-1][:110]
              if pr.stderr.strip() else "no stderr")
    cmd = [py, os.path.join(unet, "train2.py"), "--out", "/dev/null", "--aug-resim", "-0.1"]
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=unet)
    check("refuses a negative --aug-resim", pr.returncode != 0 and "S > 0" in pr.stderr)

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--cpu-train" in sys.argv:
        k = sys.argv.index("--cpu-train")
        tree = sys.argv[k + 1]
        rest = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        cpu_train(tree, rest)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--tree", default=DEF_TREE)
        sys.exit(checks(ap.parse_args().tree))
