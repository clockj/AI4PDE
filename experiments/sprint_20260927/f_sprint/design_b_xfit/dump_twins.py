"""Design B probe 0: dump held-out twin fields on val_unseen (per member, alpha 0) + sigma heads.
CPU or GPU. Output: $S/ckpt/b/dump_unseen.npz  (F: (M,N,20,32,64,2) member fields physical, S: (M,N,1,32,64,2) sigma,
x (inputs u,v), y (targets u,v), ids).  Usage: python dump_twins.py [out.npz]"""
import os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); T1 = os.path.dirname(HERE)
sys.path.insert(0, T1); sys.path.insert(0, os.path.join(T1, "submit_fno"))
import _fno
U = "/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/"
TW = ["p2_ffno_l8_sim_s1", "p2_ffno_l8_sim_s2", "p2_ffno_l8_sim_s3", "arch_vit_d384", "arch_vit_cmb3sim"]
out = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/b/dump_unseen.npz"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
d = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
X, Y, ids = d["x"], d["y"], d["ids"]
F, S = [], []
for n in TW:
    t0 = time.time()
    m = _fno.load_any(U + n + ".pth", dev)
    s = np.empty((len(X), 1, 32, 64, 2), np.float32)
    f = _fno.predict_fno([m], X, dev, alpha=0.0, sigma_out=s, batch=16, amp=None)
    F.append(f[..., :2].astype(np.float32)); S.append(s.copy())
    print(n, f.shape, "%.1fs" % (time.time() - t0), flush=True)
np.savez(out, F=np.stack(F), S=np.stack(S), x=X[..., :2], y=Y[..., :2], ids=ids, names=np.array(TW))
print("wrote", out)
