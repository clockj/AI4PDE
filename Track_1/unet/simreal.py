"""Map train_sim into train_real's measurement space, so one model can train on both.

WHY A MAP IS NEEDED AT ALL. Sim is a strict superset of real's condition grid -- every real
(Re, AoA) has an exact CFD twin, and sim adds Re 15225, Re 27975 and the holes in real's grid.
That pairing lets the domain gap be measured per condition rather than per distribution, which
`Track_1/sim_map.py` does. What it finds (12 matched pairs, 32x64, speed-matched):

  * time-mean field      rel-L2 0.400   (our own model predicts real at 0.11)
  * per-pixel std field  rel-L2 1.457, and only 0.603 after amplitude is divided out
  * fluctuation energy   2.9x real at k~1, 5.6x at k~10, 0.55x at k~35
  * transverse TI        0.263 sim vs 0.115 real (2.28x)

So sim is WRONG about the mean field and WRONG about the amplitude, by margins far larger than
the model's own error. Round 5 measured sim -> real pretraining at +0.00 and this is why. The
energy ratio is also non-monotone in k, so it is not a resolution/low-pass effect that a filter
could undo -- it is 2D-CFD over-prediction of large-scale unsteadiness.

WHAT IS LEFT. Sim's std SHAPE is informative (0.603 against 0.796 for a flat field), and its
dynamics -- given a window, where does the structure go next -- are untested by any statistic.
So the map's job is to put sim INPUTS in real's distribution (otherwise the trunk sees tensors
it will never meet at test time), and the training loss's job is to be blind to the two
quantities sim is wrong about. `train2.py --sim ... --sim-loss shape` does the second half.

THE MAP, per trajectory, in this order:
  1. speed match   -- multiply so the trajectory's mean u equals real's at that (Re, AoA).
                      Real obeys mean_u = c(AoA) * Re with c = [1.138, 1.149, 1.098, 1.002,
                      0.991]e-5 for AoA 0/5/10/15/20 (scatter 3-8%). Where a real twin exists
                      its MEASURED mean is used instead of the law.
  2. fluctuation   -- rescale the fluctuation alone so ||std field|| matches real's twin. This
                      is the 2.28x TI correction; it changes the input distribution only, since
                      the shape loss already ignores output amplitude.
  3. support       -- paint the twin's FOV/airfoil zero set onto sim. Real has 1.2-5.5% of
                      pixels identically zero and sim has none; without this the model is
                      taught to predict flow where the PIV never measures. 221 of 2048 pixels
                      differ between trajectories, so the mask is taken per condition, not
                      pooled.

PER-PIXEL VARIANTS (round 17 Q3g). Steps 1 and 2 above are SCALARS: sim keeps its own time-mean
FIELD (rel-L2 0.400 from real's) and its own per-pixel std SHAPE (0.603 after the amplitude is
divided out). `mode=` replaces those shapes with the twin's, so that the sim term teaches only
what sim is uniquely informative about -- the dynamics, where the structure goes next -- and
less of the two static fields sim is measured to be wrong about:

  mode="scalar"   (default, unchanged)  d = m_sim + (d - m_sim) * g_s
  mode="std"      per-PIXEL amplitude   d = m_sim + (d - m_sim) * g(x)
  mode="meanstd"  + the twin's mean     d = m_twin + (d - m_sim) * g(x)

with g(x) = sigma_twin(x) / max(sigma_sim(x), eps) CLAMPED to [g_s/3, 3*g_s], where g_s is the
scalar ratio ||sigma_twin|| / ||sigma_sim|| the default map already uses and
eps = 1e-3 * rms(sigma_sim) is a divide-by-zero guard at one part in a thousand of the
trajectory's own fluctuation. THE CLAMP IS NOT OPTIONAL: `--sim-loss shape` was rejected in
round 14 (-0.7 wfinal, MVPE +46%) for exactly the failure an unclamped ratio would reproduce --
dividing by a per-pixel temporal std that is near zero in the freestream amplifies quiescent
regions into noise. A factor of 3 either side of the scalar the map already trusts bounds that
by construction; the clamped fraction is printed per trajectory. Pixels inside the mask are
zero in both fields and are re-zeroed at the end regardless.

PRESSURE (round 17 Q3j). `with_p=True` carries sim's `p` as a THIRD channel of the mapped
array, `[u, v, p]`. Real has no pressure -- it is zero-filled in every real file and never
scored -- so p can only ever be a sim-side auxiliary TARGET, and there is no twin to match it
to. It therefore gets exactly two of the map's three steps: the FOV/airfoil zero mask (so its
support matches u,v) and the speed match, applied as the SQUARE of the velocity factor, since
p scales as U^2 in an incompressible flow. The fluctuation-amplitude step is NOT applied -- it
is fitted against a twin quantity real does not have. The u,v channels are bit-identical to the
2-channel map, and the cache carries its own `_p` tag so the existing caches are untouched.

Both variants are TWIN-ONLY. A condition whose twin is held out (`real_exclude`) or absent from
real has no sigma_twin(x) and no m_twin(x) to copy, and falls back to the scalar map with the
freestream law exactly as before -- round 14's leak guard is unchanged and still binding. The
twin's per-pixel fields are read at phase (0,0), the same subsampling the twin table has always
used, and are applied to every sim phase, as the FOV mask already is.
"""
from __future__ import annotations

import glob
import hashlib
import os

import numpy as np

# mean_u = C_AOA[aoa] * Re, fitted on all 82 real trajectories at 32x64 (R^2 0.944 pooled).
C_AOA = {0: 1.1381e-05, 5: 1.1487e-05, 10: 1.0976e-05, 15: 1.0018e-05, 20: 0.9910e-05}
# ||std field|| / mean_u on real, pooled: used only where no twin exists.
TI_REF = 5.7649


def _stats(d: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """(mean u, ||std field||, live mask, time-mean field, per-pixel std field) of a raw
    (T, H, W, 2) trajectory. The last two are used only by the per-pixel `mode`s; the first
    three are what the scalar map has always used and are computed identically."""
    m = ~np.all(d == 0, axis=(0, 3))
    sd = d.std(0)
    return float(d[..., 0].mean()), float(np.linalg.norm(sd)), m, d.mean(0), sd


CACHE = "/fs/scratch/PAS1187/qwu/tmp/realpde/work"


def load_sim_mapped(sim_dir: str, real_dir: str, exclude: list[str] | None = None,
                    phases: tuple[tuple[int, int], ...] = ((0, 0),),
                    frames: int = 0, verbose: bool = True,
                    cache: str | None = CACHE,
                    real_exclude: list[str] | None = None,
                    mode: str = "scalar", with_p: bool = False) -> list[np.ndarray]:
    """Every sim trajectory, mapped into real's measurement space. Raw physical units.

    `real_exclude` IS NOT OPTIONAL WHEN THERE IS A HELD-OUT SPLIT. The map reads each sim
    trajectory's real TWIN for three things -- measured mean speed, fluctuation amplitude and
    FOV mask. If that twin is a trajectory the model is being evaluated on, all three are
    leaked into training, and the mean speed is precisely the quantity the model gets worst at
    an unseen Reynolds number: the leak would land exactly where the gain is claimed. Names
    passed here are dropped from the twin table, and the conditions that lose their twin fall
    back to the freestream law and the nearest surviving mask, which is what sim's 18
    twin-less conditions already use. An ALL-DATA run has no held-out set and passes nothing.

    `mode` is "scalar" (the shipped three-step map, the default and bit-identical to every
    round before 17), "std" (the twin's per-pixel fluctuation amplitude instead of one scalar)
    or "meanstd" (that, plus the twin's time-mean field instead of sim's). See the module
    docstring for the gain's clamp rule and for why a twin-less condition still gets "scalar".

    `with_p` adds sim's pressure as a third channel (see the module docstring); the u,v
    channels are unchanged and the cache file is a different one.

    Building the map reads ~10 GB of HDF5 (82 real trajectories for the twins' statistics,
    then 100 sim ones) and takes ~10 min on shared storage -- longer than a training run.
    The result is deterministic given (sim_dir, real_dir, phases, frames), so it is cached
    to one .npy and memory-mapped thereafter. `exclude` is applied AFTER the cache is read,
    so every split shares one file.
    """
    import h5py

    from model import SUB

    if mode not in ("scalar", "std", "meanstd"):
        raise ValueError(f"unknown sim map mode {mode!r}")
    rex = sorted(real_exclude or [])
    key = None
    if cache:
        tag = "" if not rex else "_x" + hashlib.sha1(" ".join(rex).encode()).hexdigest()[:10]
        # The scalar tag is EMPTY, so every pre-round-17 cache file keeps its exact name and a
        # job already running against the old code still hits it.
        mtag = "" if mode == "scalar" else "_m" + mode
        ptag = "_p" if with_p else ""
        key = os.path.join(cache, f"simmap_p{len(phases)}_f{frames}{mtag}{ptag}{tag}.npy")
        nk = key[:-4] + "_names.npy"
        if os.path.exists(key) and os.path.exists(nk):
            arr = np.load(key, mmap_mode="r")
            nms = list(np.load(nk))
            keep = [i for i, n in enumerate(nms) if n not in set(exclude or [])]
            if verbose:
                print(f"sim: {len(keep)}/{len(nms)} trajectory-phases from cache {key}",
                      flush=True)
            return [np.ascontiguousarray(arr[i]) for i in keep]

    excl = set(exclude or [])
    twin: dict[str, tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in sorted(glob.glob(os.path.join(real_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        if nm in rex:               # held out: its statistics must not reach training
            continue
        with h5py.File(p, "r") as f:
            d = np.stack([f["u"][:].astype(np.float32), f["v"][:].astype(np.float32)], -1)
        twin[nm] = _stats(d[:, ::SUB, ::SUB])

    # Fallback mask for the conditions real does not have: the nearest Re at the same AoA.
    by_aoa: dict[int, list[tuple[int, str]]] = {}
    for nm in twin:
        re_, aoa = (int(v) for v in nm.split("_"))
        by_aoa.setdefault(aoa, []).append((re_, nm))

    out, names, n_law, clamped = [], [], 0, []
    skipped: dict[int, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(sim_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        if key is None and nm in excl:      # when caching, build every condition once
            continue
        re_, aoa = (int(v) for v in nm.split("_"))
        with h5py.File(p, "r") as f:
            u = f["u"][: frames or None].astype(np.float32)
            v = f["v"][: frames or None].astype(np.float32)
            q = f["p"][: frames or None].astype(np.float32) if with_p else None
        has_twin = nm in twin
        if has_twin:
            mu_r, sd_r, mask, mt_r, sg_r = twin[nm]
        else:
            # No twin: the freestream comes from the law, the amplitude from the pooled ratio,
            # and the mask from the nearest Reynolds number at this angle of attack. There is
            # no per-pixel field to copy either, so this condition keeps the SCALAR map in
            # every mode -- round 14's leak guard, unchanged.
            if aoa not in by_aoa:
                # NO REAL TWIN AT THIS ANGLE AT ALL. The freestream still comes from the law, but
                # the MASK does not: it is copied from the nearest Reynolds number at the same
                # AoA, and there is no such trajectory left. This happens exactly once -- in the
                # AoA-holdout screen, where every real trajectory at the held-out angle is in
                # `real_exclude` (job 54376781 died here with KeyError: 20). Skipping is not a
                # workaround, it is the right answer: a sim trajectory at the held-out angle can
                # only be mapped through that angle's real geometry, which is precisely what the
                # holdout forbids. The cache key carries a hash of `real_exclude`, so the
                # shortened table lands in its own file and can never reach a run that has those
                # twins.
                skipped.setdefault(aoa, []).append(nm)
                continue
            n_law += 1
            mu_r = C_AOA[aoa] * re_
            sd_r = TI_REF * mu_r
            near = min(by_aoa[aoa], key=lambda t: abs(t[0] - re_))[1]
            mask = twin[near][2]
            mt_r = sg_r = None

        for py, px in phases:
            d = np.stack([u[:, py::SUB, px::SUB], v[:, py::SUB, px::SUB]], -1)
            # Mask FIRST: both scale factors are then measured on exactly the pixels the
            # twin measures, so the mapped mean and amplitude match the twin's rather than
            # landing a few percent low because the zeroed pixels dilute the average.
            d = d * mask[None, :, :, None]               # 3. support
            sc = mu_r / d[..., 0].mean()                 # 1. speed match
            d = d * sc
            m = d.mean(0, keepdims=True)
            sg = d.std(0)
            sd_s = np.linalg.norm(sg)
            g_s = sd_r / max(sd_s, 1e-12)                # the scalar gain
            if mode == "scalar" or sg_r is None:
                d = m + (d - m) * g_s                    # 2. fluctuation amplitude
            else:
                # Per-pixel amplitude, clamped to a factor of 3 either side of the scalar the
                # map already trusts: an unclamped sigma_twin/sigma_sim blows up wherever the
                # freestream is quiescent, which is the `--sim-loss shape` failure of round 14.
                eps = 1e-3 * sd_s / np.sqrt(sg.size)
                g = sg_r / np.maximum(sg, eps)
                lo, hi = g_s / 3.0, 3.0 * g_s
                live = mask[:, :, None]
                n_live = max(int(live.sum()) * sg.shape[-1], 1)
                n_lo = int(((g < lo) & live).sum())
                n_hi = int(((g > hi) & live).sum())
                clamped.append((nm, (n_lo + n_hi) / n_live, n_lo / n_live, n_hi / n_live))
                g = np.clip(g, lo, hi).astype(np.float32)
                d = (m if mode == "std" else mt_r[None]) + (d - m) * g[None]
            d = d * mask[None, :, :, None]               # re-zero: step 2 lifts the mask off
            if q is not None:
                # p gets the mask and the speed match SQUARED (p ~ U^2), and nothing else:
                # there is no real twin pressure to match an amplitude to.
                qd = q[:, py::SUB, px::SUB] * mask[None, :, :] * (sc * sc)
                d = np.concatenate([d, qd[..., None]], -1)
            out.append(np.ascontiguousarray(d, dtype=np.float32))
            names.append(nm)

    if verbose and clamped:
        for nm, fr, flo, fhi in clamped:
            print(f"simmap: {nm} mode {mode} clamped {fr:.4%} of live elements "
                  f"(lo {flo:.4%} hi {fhi:.4%})", flush=True)
        fr = [c[1] for c in clamped]
        print(f"simmap: mode {mode} on {len(fr)} twin condition(s), clamped fraction "
              f"mean {np.mean(fr):.4%} max {np.max(fr):.4%}", flush=True)
    if verbose:
        print(f"sim: map mode '{mode}'{' +p' if with_p else ''}; "
              f"{len(out)} trajectory-phases mapped into real's space "
              f"({n_law} condition(s) placed by the freestream law, the rest by their twin"
              + (f"; {len(rex)} real trajectories withheld from the twin table)" if rex
                 else ")"), flush=True)
    for aoa_, nms_ in sorted(skipped.items()):
        print(f"sim: SKIPPED {len(nms_)} trajectories at AoA {aoa_} -- every real trajectory at "
              f"that angle is in real_exclude, so there is no twin to take a mask from. This is "
              f"the AoA-holdout case; the map is built without them.", flush=True)

    if key is not None:
        os.makedirs(cache, exist_ok=True)
        np.save(key, np.stack(out))
        np.save(key[:-4] + "_names.npy", np.array(names))
        print(f"sim: cached {key} ({np.stack(out).nbytes/1e9:.2f} GB)", flush=True)
        out = [d for d, n in zip(out, names) if n not in excl]
    return out


def sim_names(sim_dir: str, exclude: list[str] | None = None,
              phases: tuple[tuple[int, int], ...] = ((0, 0),)) -> list[str]:
    excl = set(exclude or [])
    return [os.path.basename(p)[:-3]
            for p in sorted(glob.glob(os.path.join(sim_dir, "*.h5")))
            if os.path.basename(p)[:-3] not in excl
            for _ in phases]
