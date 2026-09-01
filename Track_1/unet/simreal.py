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
"""
from __future__ import annotations

import glob
import os

import numpy as np

# mean_u = C_AOA[aoa] * Re, fitted on all 82 real trajectories at 32x64 (R^2 0.944 pooled).
C_AOA = {0: 1.1381e-05, 5: 1.1487e-05, 10: 1.0976e-05, 15: 1.0018e-05, 20: 0.9910e-05}
# ||std field|| / mean_u on real, pooled: used only where no twin exists.
TI_REF = 5.7649


def _stats(d: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(mean u, ||per-pixel std field||, live mask) of a raw (T, H, W, 2) trajectory."""
    m = ~np.all(d == 0, axis=(0, 3))
    return float(d[..., 0].mean()), float(np.linalg.norm(d.std(0))), m


CACHE = "/fs/scratch/PAS1187/qwu/tmp/realpde/work"


def load_sim_mapped(sim_dir: str, real_dir: str, exclude: list[str] | None = None,
                    phases: tuple[tuple[int, int], ...] = ((0, 0),),
                    frames: int = 0, verbose: bool = True,
                    cache: str | None = CACHE) -> list[np.ndarray]:
    """Every sim trajectory, mapped into real's measurement space. Raw physical units.

    Building the map reads ~10 GB of HDF5 (82 real trajectories for the twins' statistics,
    then 100 sim ones) and takes ~10 min on shared storage -- longer than a training run.
    The result is deterministic given (sim_dir, real_dir, phases, frames), so it is cached
    to one .npy and memory-mapped thereafter. `exclude` is applied AFTER the cache is read,
    so every split shares one file.
    """
    import h5py

    from model import SUB

    key = None
    if cache:
        key = os.path.join(cache, f"simmap_p{len(phases)}_f{frames}.npy")
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
    twin: dict[str, tuple[float, float, np.ndarray]] = {}
    for p in sorted(glob.glob(os.path.join(real_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        with h5py.File(p, "r") as f:
            d = np.stack([f["u"][:].astype(np.float32), f["v"][:].astype(np.float32)], -1)
        twin[nm] = _stats(d[:, ::SUB, ::SUB])

    # Fallback mask for the conditions real does not have: the nearest Re at the same AoA.
    by_aoa: dict[int, list[tuple[int, str]]] = {}
    for nm in twin:
        re_, aoa = (int(v) for v in nm.split("_"))
        by_aoa.setdefault(aoa, []).append((re_, nm))

    out, names, n_law = [], [], 0
    for p in sorted(glob.glob(os.path.join(sim_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        if key is None and nm in excl:      # when caching, build every condition once
            continue
        re_, aoa = (int(v) for v in nm.split("_"))
        with h5py.File(p, "r") as f:
            u = f["u"][: frames or None].astype(np.float32)
            v = f["v"][: frames or None].astype(np.float32)
        if nm in twin:
            mu_r, sd_r, mask = twin[nm]
        else:
            # No twin: the freestream comes from the law, the amplitude from the pooled ratio,
            # and the mask from the nearest Reynolds number at this angle of attack.
            n_law += 1
            mu_r = C_AOA[aoa] * re_
            sd_r = TI_REF * mu_r
            near = min(by_aoa[aoa], key=lambda t: abs(t[0] - re_))[1]
            mask = twin[near][2]

        for py, px in phases:
            d = np.stack([u[:, py::SUB, px::SUB], v[:, py::SUB, px::SUB]], -1)
            # Mask FIRST: both scale factors are then measured on exactly the pixels the
            # twin measures, so the mapped mean and amplitude match the twin's rather than
            # landing a few percent low because the zeroed pixels dilute the average.
            d = d * mask[None, :, :, None]               # 3. support
            d = d * (mu_r / d[..., 0].mean())            # 1. speed match
            m = d.mean(0, keepdims=True)
            sd_s = np.linalg.norm(d.std(0))
            d = m + (d - m) * (sd_r / max(sd_s, 1e-12))  # 2. fluctuation amplitude
            d = d * mask[None, :, :, None]               # re-zero: step 2 lifts the mask off
            out.append(np.ascontiguousarray(d, dtype=np.float32))
            names.append(nm)

    if verbose:
        print(f"sim: {len(out)} trajectory-phases mapped into real's space "
              f"({n_law} condition(s) placed by the freestream law, the rest by their twin)",
              flush=True)
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
