"""Dump the held-out twins' fields on val_unseen (per member, alpha 0) and their sigma heads: the input of make_k0_band.py.

CPU or GPU. Output npz: F (M,N,20,32,64,2) member fields, physical; S (M,N,1,32,64,2) sigma; x, y (inputs / targets, u and v);
ids; names.

usage: python dump_twins.py OUT.npz [--unet DIR] [--val Track_1/val_unseen.npz] [--twins NAME ...]
"""
import argparse, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); T1 = os.path.dirname(HERE)
sys.path.insert(0, T1); sys.path.insert(0, os.path.join(T1, "submit_fno"))
import _fno
TW = ["p2_ffno_l8_sim_s1", "p2_ffno_l8_sim_s2", "p2_ffno_l8_sim_s3", "arch_vit_d384", "arch_vit_cmb3sim"]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--unet", default=os.path.join(T1, "unet"), help="directory holding the twins' checkpoints")
ap.add_argument("--val", default=os.path.join(T1, "val_unseen.npz"), help="make_val.py's held-out windows")
ap.add_argument("--twins", nargs="+", default=TW, help="checkpoint stems, or paths to .pth files")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
d = np.load(a.val)
X, Y, ids = d["x"], d["y"], d["ids"]
F, S = [], []
for n in a.twins:
    t0 = time.time()
    m = _fno.load_any(n if n.endswith(".pth") else os.path.join(a.unet, n + ".pth"), dev)
    s = np.empty((len(X), 1, 32, 64, 2), np.float32)
    f = _fno.predict_fno([m], X, dev, alpha=0.0, sigma_out=s, batch=16, amp=None)
    F.append(f[..., :2].astype(np.float32)); S.append(s.copy())
    print(n, f.shape, "%.1fs" % (time.time() - t0), flush=True)
np.savez(a.out, F=np.stack(F), S=np.stack(S), x=X[..., :2], y=Y[..., :2], ids=ids,
         names=np.array([os.path.basename(n).replace(".pth", "") for n in a.twins]))
print("wrote", a.out)
