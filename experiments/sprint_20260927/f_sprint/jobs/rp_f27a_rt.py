# EA: packed round trip (int8 tensor + fp16) and fp16 autocast forward of an ffvit checkpoint vs fp32.
import sys, os, time, numpy as np, torch
W = sys.argv[1]; CK = sys.argv[2]
sys.path.insert(0, os.path.join(W, "Track_1", "submit_fno"))
import _fno
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(os.path.join(W, "Track_1", "val_unseen.npz"))
idx = np.linspace(0, len(d["x"]) - 1, 96).astype(int)
x = d["x"][idx].astype(np.float32); y = d["y"][idx][..., :2].astype(np.float32)
m32 = _fno.load_any(CK, dev)
blob = torch.load(CK, map_location="cpu")
blob["model_state_dict"] = _fno.pack_state_dict(blob["model_state_dict"], fp16=True, int8=True, linear="tensor")
p = f"/tmp/rp_f27a_rt_{os.getpid()}.pth"; torch.save(blob, p)
print("packed size MB", os.path.getsize(p) / 1e6, " fp32 MB", os.path.getsize(CK) / 1e6)
mq = _fno.load_any(p, dev)
def run(m, amp=None):
    return _fno.predict_fno(m, x, dev, alpha=0.80, amp=amp)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
a = run(m32); b = run(mq); c = run(mq, amp=torch.float16); e = run(m32, amp=torch.float16)
print("model rel-L2 vs target fp32 %.5f  packed %.5f  packed+fp16 %.5f" % (rel(a, y), rel(b, y), rel(c, y)))
print("rel diff packed vs fp32 %.2e | fp16-autocast(fp32 w) vs fp32 %.2e | packed+fp16 vs fp32 %.2e" % (rel(b, a), rel(e, a), rel(c, a)))
print("finite:", np.isfinite(c).all())
# batch-1 path (T2-like): per-window forward at N=1 equals batched
one = np.concatenate([_fno.predict_fno(mq, x[i:i + 1], dev, alpha=0.80) for i in range(8)])
print("batch-1 vs batched rel diff %.2e" % rel(one, b[:8]))
os.remove(p)
print("RT DONE")
