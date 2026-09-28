"""Design C: dump held-out twin outputs on a stream (model space), once, for CPU iteration.
out npz: y (M,N,20,32,64,2) f32 raw field per member (model-normalised, _fno.MEAN/STD), sig (M,N,1,32,64,2),
X, Y (N,20,32,64,2) model-normalised input / target, names (N,), tid (N,), first (N,), members (M,)."""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "submit"))
import t2_common as t2
import _fno
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_fno._SPEC = "bmm"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--split", default="unseen")
ap.add_argument("--out", required=True); ap.add_argument("--fp16", action="store_true")
a = ap.parse_args()
names = ([n for n in t2.all_names() if n.endswith(a.split[3:])] if a.split.startswith("aoa") else {"unseen": t2.UNSEEN, "seen": t2.SEEN, "all": t2.all_names()}[a.split])
st = t2.build_stream(names)
X = np.stack([t2.normalize_in(s[2]) for s in st])[..., :2]
Y = np.stack([t2.normalize_in(s[3]) for s in st])[..., :2]
N = len(st); dev = "cuda"
ys, ss = [], []
for p in a.ckpt:
    m = _fno.load_any(p, dev); m.eval()
    yo, so = [], []
    with torch.no_grad():
        for i in range(0, N, 16):
            x = torch.from_numpy(X[i:i + 16]).to(dev)
            pv = x.permute(0, 1, 4, 2, 3).reshape(len(x), 40, 32, 64).contiguous()
            o = m(pv)
            raw = o[0].float().reshape(len(x), 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
            sg = o[1].float().clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
            yo.append(raw.cpu().numpy()); so.append(sg.cpu().numpy())
    ys.append(np.concatenate(yo)); ss.append(np.concatenate(so))
    print(os.path.basename(p), ys[-1].shape, ss[-1].shape, "rel", float(np.mean(np.linalg.norm((ys[-1]-Y).reshape(N,-1),axis=1)/np.linalg.norm(Y.reshape(N,-1),axis=1))), flush=True)
    del m
dt = np.float16 if a.fp16 else np.float32
np.savez(a.out, y=np.stack(ys).astype(dt), sig=np.stack(ss).astype(np.float32), X=X.astype(dt), Y=Y.astype(np.float32),
         names=np.array([s[0] for s in st]), tid=np.array([s[1] for s in st]), first=np.array([s[4] for s in st]),
         members=np.array([os.path.basename(p) for p in a.ckpt]))
print("saved", a.out, flush=True)
