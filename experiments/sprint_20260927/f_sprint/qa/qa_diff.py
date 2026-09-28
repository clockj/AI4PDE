import numpy as np, sys, glob, os
d = sys.argv[1]; ref = sys.argv[2]
R = np.load(ref)
rp, rh = R["pred"][..., :2], (R["upper"] - R["lower"])[..., :2] / 2
for f in sorted(glob.glob(d + "/*_A.npz")):
    if f == ref: continue
    Z = np.load(f); p = Z["pred"][..., :2]; h = (Z["upper"] - Z["lower"])[..., :2] / 2
    dp, dh = np.abs(p - rp), np.abs(h - rh)
    rel = dh / np.maximum(rh, 1e-6)
    q = lambda a: " ".join(f"{np.quantile(a, x):.4g}" for x in (0.5, 0.99, 0.9999))
    i = np.unravel_index(dp.argmax(), dp.shape)
    print(f"{os.path.basename(f)[:-6]:45s} pred |d| mean {dp.mean():.4g} q50/99/99.99 {q(dp)} max {dp.max():.4g} at win{i[0]} t{i[1]} y{i[2]} x{i[3]} c{i[4]} (ref |p| {np.abs(rp).mean():.3g})")
    print(f"{'':45s} half |d| mean {dh.mean():.4g} q {q(dh)} max {dh.max():.4g}; rel half-ratio q {q(rel)} max {rel.max():.4g}; half mean {h.mean():.4f} vs ref {rh.mean():.4f}")
    # per-window max shift in pred relative to window rms
    w = dp.reshape(len(dp), -1).max(1); print(f"{'':45s} per-window max|dpred| top3 {np.sort(w)[-3:]} windows {np.argsort(w)[-3:]}")
