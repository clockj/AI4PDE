"""Map the CFD trajectories (train_sim) into the measurement space of the PIV data (train_real),
so that one network can train on both.

Every CFD trajectory `<Re>_<AoA>` is mapped by three steps, in this order, using the statistics
of the real recording of the same condition (its "twin") where one exists:

  1. support     -- the twin's zero set (field of view, airfoil body) is painted onto the CFD
                    field, so the network is never taught to predict flow where the PIV does
                    not measure;
  2. speed match -- the trajectory is scaled so its mean u equals the twin's measured mean u;
  3. fluctuation -- the fluctuation about the time mean is rescaled so that the norm of the
                    per-pixel std field equals the twin's.

A condition without a twin takes its mean speed from the freestream law `mean_u = C_AOA * Re`
fitted on the real recordings, its fluctuation from the pooled turbulence-intensity ratio
`TI_REF`, and its zero set from the real recording nearest in Re at the same angle of attack.

`real_exclude` names real trajectories whose statistics must not be used (a twin's held-out
trajectories): their conditions then fall back to the law. `exclude` drops CFD trajectories
from the result. The map is deterministic, costs about ten minutes of HDF5 reading, and is
cached as one .npy under `$RPDE_SIMCACHE` (one file per `real_exclude` set; `exclude` is
applied after the cache is read).
"""
from __future__ import annotations

import glob
import hashlib
import os

import numpy as np

from data import SUB, real_files

# mean_u = C_AOA[aoa] * Re, fitted on the 82 real trajectories at 32 x 64.
C_AOA = {0: 1.1381e-05, 5: 1.1487e-05, 10: 1.0976e-05, 15: 1.0018e-05, 20: 0.9910e-05}
# ||std field|| / mean_u on the real recordings, pooled: used only where no twin exists.
TI_REF = 5.7649

CACHE = os.environ.get("RPDE_SIMCACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "simcache"))


def _stats(d: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(mean u, ||std field||, live mask) of a raw (T, H, W, 2) trajectory."""
    m = ~np.all(d == 0, axis=(0, 3))
    sd = d.std(0)
    return float(d[..., 0].mean()), float(np.linalg.norm(sd)), m


def load_sim_mapped(sim_dir: str, real_dir: str, exclude: list[str] | None = None,
                    real_exclude: list[str] | None = None, cache: str | None = CACHE,
                    verbose: bool = True) -> list[np.ndarray]:
    """Every CFD trajectory of `sim_dir` mapped into the real measurement space, as (T, 32, 64,
    2) float32 arrays in physical units, in sorted file order."""
    import h5py

    rex = sorted(real_exclude or [])
    key = None
    if cache:
        tag = "" if not rex else "_x" + hashlib.sha1(" ".join(rex).encode()).hexdigest()[:10]
        key = os.path.join(cache, f"simmap_p1_f0{tag}.npy")
        nk = key[:-4] + "_names.npy"
        if os.path.exists(key) and os.path.exists(nk):
            arr = np.load(key, mmap_mode="r")
            nms = list(np.load(nk))
            keep = [i for i, n in enumerate(nms) if n not in set(exclude or [])]
            if verbose:
                print(f"sim: {len(keep)}/{len(nms)} trajectories from cache {key}", flush=True)
            return [np.ascontiguousarray(arr[i]) for i in keep]

    excl = set(exclude or [])
    twin: dict[str, tuple[float, float, np.ndarray]] = {}
    for p in real_files(real_dir):
        nm = os.path.basename(p)[:-3]
        if nm in rex:
            continue
        with h5py.File(p, "r") as f:
            d = np.stack([f["u"][:].astype(np.float32), f["v"][:].astype(np.float32)], -1)
        twin[nm] = _stats(d[:, ::SUB, ::SUB])

    by_aoa: dict[int, list[tuple[int, str]]] = {}
    for nm in twin:
        re_, aoa = (int(v) for v in nm.split("_"))
        by_aoa.setdefault(aoa, []).append((re_, nm))

    out, names, n_law = [], [], 0
    for p in sorted(glob.glob(os.path.join(sim_dir, "*.h5"))):
        nm = os.path.basename(p)[:-3]
        if key is None and nm in excl:
            continue
        re_, aoa = (int(v) for v in nm.split("_"))
        with h5py.File(p, "r") as f:
            u = f["u"][:].astype(np.float32)
            v = f["v"][:].astype(np.float32)
        if nm in twin:
            mu_r, sd_r, mask = twin[nm]
        else:
            if aoa not in by_aoa:
                raise SystemExit(f"sim {nm}: no real recording at AoA {aoa} to take a mask from")
            n_law += 1
            mu_r = C_AOA[aoa] * re_
            sd_r = TI_REF * mu_r
            near = min(by_aoa[aoa], key=lambda t: abs(t[0] - re_))[1]
            mask = twin[near][2]

        d = np.stack([u[:, ::SUB, ::SUB], v[:, ::SUB, ::SUB]], -1)
        d = d * mask[None, :, :, None]               # 1. support
        sc = mu_r / d[..., 0].mean()                 # 2. speed match
        d = d * sc
        m = d.mean(0, keepdims=True)
        sg = d.std(0)
        sd_s = np.linalg.norm(sg)
        g_s = sd_r / max(sd_s, 1e-12)
        d = m + (d - m) * g_s                        # 3. fluctuation amplitude
        d = d * mask[None, :, :, None]               # re-zero: step 3 lifts the mask off
        out.append(np.ascontiguousarray(d, dtype=np.float32))
        names.append(nm)

    if verbose:
        print(f"sim: {len(out)} trajectories mapped into the real measurement space "
              f"({n_law} condition(s) placed by the freestream law, the rest by their twin"
              + (f"; {len(rex)} real trajectories withheld from the twin table)" if rex
                 else ")"), flush=True)
    if key is not None:
        os.makedirs(cache, exist_ok=True)
        np.save(key, np.stack(out))
        np.save(key[:-4] + "_names.npy", np.array(names))
        print(f"sim: cached {key} ({np.stack(out).nbytes/1e9:.2f} GB)", flush=True)
        out = [d for d, n in zip(out, names) if n not in excl]
    return out
