"""Gate the PER-PIXEL sim->real domain map (round 17 Q3g) before it trains anything.

    python Track_1/verify/simmap_modes.py

`simreal.load_sim_mapped(mode=...)` gained two variants that copy more of the real twin's
STATIC statistics onto sim: "std" takes the twin's per-pixel fluctuation amplitude instead of
one scalar, "meanstd" also takes its time-mean field. Three things have to be true before a
GPU-hour is spent on them, and none of them is visible in a training log:

  A  `mode="scalar"` is BIT-IDENTICAL to the pre-round-17 map. Jobs are running against this
     file, and a checkpoint trained on a silently different map is not comparable to the
     controls it will be scored against. The reference here is a verbatim transcription of the
     old three-step map (`_orig_map` below), not a call into the new code, so this is an
     independent check rather than a tautology.
  B  the CACHE KEY of scalar mode is unchanged -- `simmap_p1_f0.npy` and
     `simmap_p1_f0_x<hash>.npy` -- so a running job keeps hitting the file it already built,
     and each new mode gets its own name instead of colliding with it.
  C  a condition with no usable twin (held out via `real_exclude`, or absent from real) falls
     back to the SCALAR map in every mode, bit for bit. That is round 14's leak guard: the map
     reads the twin's mean speed, amplitude and mask, and a held-out twin must contribute
     nothing. Losing it would put the held-out set's statistics back into training exactly
     where the gain is claimed.

Then the point of the exercise: what the modes actually do to the domain gap `sim_map.py`
measured (time-mean field rel-L2 0.400, per-pixel std field 0.603 after amplitude). "std"
should drive the std gap toward 0 and leave the mean at 0.40; "meanstd" should drive both.
Anything left is the clamp, whose fraction is printed per trajectory.

Light by construction: it symlinks THREE sim conditions and their real twins into a temporary
directory and runs the real `load_sim_mapped` over that, so it is a handful of HDF5 reads on a
login node rather than the 10 GB / 10 min map the training job builds.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import re
import sys
import tempfile

import numpy as np

SIM = "/users/PAS1187/qwu/RealPDE/train_sim/train_sim"
REAL = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
# Two conditions at one AoA (so a held-out twin still has a nearest-Re mask to fall back to)
# plus one at another angle. 13950_10 is the trajectory CLAUDE.md's TKE anchors use.
NAMES = ["13950_10", "16500_10", "21600_20"]
UNET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unet")
sys.path.insert(0, UNET)

SUB = 2


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def load32(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        d = np.stack([f["u"][:].astype(np.float32), f["v"][:].astype(np.float32)], -1)
    return d[:, ::SUB, ::SUB]


def link(root: str) -> tuple[str, str]:
    """A sim dir and a real dir holding only NAMES, as symlinks."""
    s, r = os.path.join(root, "sim"), os.path.join(root, "real")
    os.makedirs(s, exist_ok=True)
    os.makedirs(r, exist_ok=True)
    for nm in NAMES:
        for src, dst in ((SIM, s), (REAL, r)):
            p = os.path.join(dst, nm + ".h5")
            if not os.path.exists(p):
                os.symlink(os.path.join(src, nm + ".h5"), p)
    return s, r


def _orig_map(sim_dir: str, real_dir: str,
              real_exclude: list[str] | None = None) -> dict[str, np.ndarray]:
    """The pre-round-17 map, transcribed VERBATIM from simreal.py at commit ab0df2e.

    Kept as a literal copy on purpose: it is the definition of "unchanged" that check A is
    against, and it must not follow simreal.py if simreal.py moves.
    """
    import h5py
    C_AOA = {0: 1.1381e-05, 5: 1.1487e-05, 10: 1.0976e-05, 15: 1.0018e-05, 20: 0.9910e-05}
    TI_REF = 5.7649
    rex = sorted(real_exclude or [])
    twin = {}
    for p in sorted(glob.glob(os.path.join(real_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        if nm in rex:
            continue
        with h5py.File(p, "r") as f:
            d = np.stack([f["u"][:].astype(np.float32), f["v"][:].astype(np.float32)], -1)
        d = d[:, ::SUB, ::SUB]
        m = ~np.all(d == 0, axis=(0, 3))
        twin[nm] = (float(d[..., 0].mean()), float(np.linalg.norm(d.std(0))), m)
    by_aoa: dict[int, list] = {}
    for nm in twin:
        re_, aoa = (int(v) for v in nm.split("_"))
        by_aoa.setdefault(aoa, []).append((re_, nm))
    out = {}
    for p in sorted(glob.glob(os.path.join(sim_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        re_, aoa = (int(v) for v in nm.split("_"))
        with h5py.File(p, "r") as f:
            u = f["u"][:].astype(np.float32)
            v = f["v"][:].astype(np.float32)
        if nm in twin:
            mu_r, sd_r, mask = twin[nm]
        else:
            mu_r = C_AOA[aoa] * re_
            sd_r = TI_REF * mu_r
            near = min(by_aoa[aoa], key=lambda t: abs(t[0] - re_))[1]
            mask = twin[near][2]
        d = np.stack([u[:, ::SUB, ::SUB], v[:, ::SUB, ::SUB]], -1)
        d = d * mask[None, :, :, None]
        d = d * (mu_r / d[..., 0].mean())
        m = d.mean(0, keepdims=True)
        sd_s = np.linalg.norm(d.std(0))
        d = m + (d - m) * (sd_r / max(sd_s, 1e-12))
        d = d * mask[None, :, :, None]
        out[nm] = np.ascontiguousarray(d, dtype=np.float32)
    return out


def run(sim_dir, real_dir, mode, real_exclude=None, cache=None):
    """load_sim_mapped over the three linked conditions -> (dict name->array, log text)."""
    from simreal import load_sim_mapped
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        arrs = load_sim_mapped(sim_dir, real_dir, mode=mode, real_exclude=real_exclude,
                               cache=cache, verbose=True)
    from simreal import sim_names
    return dict(zip(sim_names(sim_dir), arrs)), buf.getvalue()


def main() -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    ok = True
    with tempfile.TemporaryDirectory(prefix="simmap_modes_") as root:
        sim_dir, real_dir = link(root)
        twins = {nm: load32(os.path.join(REAL, nm + ".h5")) for nm in NAMES}
        tw = {nm: (d.mean(0), d.std(0)) for nm, d in twins.items()}

        print("== the map's effect on the domain gap (3 conditions, twins present)")
        print(f"{'mode':8s} {'condition':10s} {'mean relL2':>11s} {'std relL2':>10s} "
              f"{'||std|| ratio':>13s} {'clamped':>9s}")
        maps = {}
        for mode in ("scalar", "std", "meanstd"):
            maps[mode], log = run(sim_dir, real_dir, mode)
            frac = dict(re.findall(r"simmap: (\S+) mode \S+ clamped ([0-9.]+)%", log))
            for nm in NAMES:
                d = maps[mode][nm]
                mt, sg = d.mean(0), d.std(0)
                print(f"{mode:8s} {nm:10s} {rel(mt, tw[nm][0]):11.4g} "
                      f"{rel(sg, tw[nm][1]):10.4g} "
                      f"{np.linalg.norm(sg) / np.linalg.norm(tw[nm][1]):13.4f} "
                      f"{(frac.get(nm, '0') + '%'):>9s}")
            for l in log.splitlines():          # the clamp lines, per trajectory then pooled
                if l.startswith("simmap:"):
                    print("   " + l)

        print("\n== A  scalar mode is bit-identical to the pre-round-17 map")
        orig = _orig_map(sim_dir, real_dir)
        for nm in NAMES:
            same = np.array_equal(maps["scalar"][nm], orig[nm])
            ok &= same
            print(f"   {nm:10s} array_equal {same}  "
                  f"maxabsdiff {np.abs(maps['scalar'][nm] - orig[nm]).max():.3e}")

        print("\n== C  a held-out twin falls back to the scalar map, bit for bit "
              f"(real_exclude=['{NAMES[0]}'])")
        ex = [NAMES[0]]
        sc_x, _ = run(sim_dir, real_dir, "scalar", real_exclude=ex)
        ms_x, logx = run(sim_dir, real_dir, "meanstd", real_exclude=ex)
        og_x = _orig_map(sim_dir, real_dir, real_exclude=ex)
        same = np.array_equal(sc_x[NAMES[0]], og_x[NAMES[0]])
        ok &= same
        print(f"   scalar+exclude == original+exclude on {NAMES[0]}: {same}")
        same = np.array_equal(ms_x[NAMES[0]], sc_x[NAMES[0]])
        ok &= same
        print(f"   meanstd+exclude == scalar+exclude on the EXCLUDED {NAMES[0]}: {same}")
        for nm in NAMES[1:]:
            diff = not np.array_equal(ms_x[nm], sc_x[nm])
            ok &= diff
            print(f"   meanstd+exclude differs from scalar on the kept {nm}: {diff}")
        print("   " + [l for l in logx.splitlines() if l.startswith("sim: map")][0])

        print("\n== B  cache keys")
        cache = os.path.join(root, "cache")
        for mode in ("scalar", "std", "meanstd"):
            run(sim_dir, real_dir, mode, cache=cache)
        run(sim_dir, real_dir, "scalar", real_exclude=ex, cache=cache)
        got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cache, "*.npy"))
                     if not p.endswith("_names.npy"))
        for g in got:
            print(f"   {g}")
        want = "simmap_p1_f0.npy"
        same = want in got
        ok &= same
        print(f"   scalar key is still exactly {want}: {same}")
        ok &= sum(g.startswith("simmap_p1_f0_x") for g in got) == 1
        ok &= "simmap_p1_f0_mstd.npy" in got and "simmap_p1_f0_mmeanstd.npy" in got

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
