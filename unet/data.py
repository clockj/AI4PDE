"""The real-data pipeline and the loss helpers the training script shares with the model code.

Geometry is the competition's: a 32 x 64 grid (the 64 x 128 recording sub-sampled by 2), 20 input
frames to 20 output frames, channels [u, v]. The network maps a whole input window to a whole
output window in one pass, with time folded into the channel axis (CH = 20 * 2 = 40 channels).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

T_IN = T_OUT = 20
H, W, C = 32, 64, 2
SUB = 2                      # sub-sampling of the 64 x 128 recording
CH = T_IN * C                # channels once time is folded in

# Normalisation statistics of train_real at 32 x 64, channels [u, v]: the competition's own
# "input" statistics (example_data/mean_std_real.pt), which the shipped _fno.py carries too.
MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

# The eight trajectories every twin holds out (band/make_val.py builds the validation windows
# from them). A member trains on all trajectories; a twin on all but these.
VAL_TRAJ = ["3750_0", "6300_10", "8850_20", "11400_20",
            "13950_10", "16500_5", "21600_15", "26700_0"]

# Never read: `train_real/7575_0.h5` is a copy of `6300_0.h5` that the data page asks
# participants to leave out, and the organizers' training mount does not contain it.
DROP = ["7575_0"]


def release_dirs(data_dir: str) -> tuple[str, str]:
    """(real, sim): the directories holding the `*.h5` files of `train_real` and `train_sim`,
    whether the release was unpacked flat (`<data_dir>/train_real/*.h5`, the organizers'
    mount) or as the archives unpack (`<data_dir>/train_real/train_real/*.h5`)."""
    out = []
    for part in ("train_real", "train_sim"):
        for d in (os.path.join(data_dir, part), os.path.join(data_dir, part, part)):
            if glob.glob(os.path.join(d, "*.h5")):
                out.append(d)
                break
        else:
            raise SystemExit(f"no {part}/*.h5 under {data_dir}")
    return out[0], out[1]


def real_files(data_dir: str) -> list[str]:
    """The `*.h5` files of a `train_real` directory, sorted, without `DROP`."""
    return [p for p in sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            if os.path.basename(p)[:-3] not in DROP]


def load_trajectories(data_dir: str, exclude: list[str]) -> list[np.ndarray]:
    """One (T, 32, 64, 2) float32 array per trajectory of `data_dir`, in sorted file order,
    in physical units (m/s), skipping the stems in `exclude`."""
    import h5py

    out = []
    for path in real_files(data_dir):
        if os.path.basename(path)[:-3] in exclude:
            continue
        with h5py.File(path, "r") as f:
            u = f["u"][:].astype(np.float32)
            v = f["v"][:].astype(np.float32)
        out.append(np.stack([u[:, ::SUB, ::SUB], v[:, ::SUB, ::SUB]], -1))
    return out


def window_index(trajs: list[np.ndarray], stride: int) -> np.ndarray:
    """(trajectory, start) pairs of every 40-frame window at this stride; windows are sliced
    from the trajectories per batch rather than materialised."""
    return np.asarray([(i, t) for i, d in enumerate(trajs)
                       for t in range(0, len(d) - T_IN - T_OUT + 1, stride)], dtype=np.int64)


def to_ch(x: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, T*C, H, W)."""
    return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], CH, H, W)


def from_ch(x: torch.Tensor) -> torch.Tensor:
    """(B, T*C, H, W) -> (B, T, H, W, C)."""
    return x.reshape(x.shape[0], T_OUT, C, H, W).permute(0, 1, 3, 4, 2)


def std_t(y: torch.Tensor) -> torch.Tensor:
    """Per-pixel temporal standard deviation of (B, T, H, W, C), keepdim on T. Biased (1/T), as
    the scorer's kinetic energy is."""
    return y.std(1, keepdim=True, unbiased=False)


def tke(y: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, H, W): 0.5 * sum_c std_t(y_c)^2, the scorer's turbulent kinetic
    energy. `scale` (the per-channel std the data was normalised by) puts it in physical units."""
    s = std_t(y)
    if scale is not None:
        s = s * scale
    return 0.5 * (s ** 2).squeeze(1).sum(-1)


def rel_l2(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 over everything but the batch axis."""
    p = p.reshape(p.shape[0], -1)
    t = t.reshape(t.shape[0], -1)
    return (p - t).norm(dim=1) / t.norm(dim=1).clamp(min=1e-8)
