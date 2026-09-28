"""Profile the fused F-FNO / ViT groups at batch 56 (fp16 autocast) and try the no-retraining knobs."""
import sys, os, time, json
import numpy as np
pkg = sys.argv[1]
os.chdir(pkg); sys.path.insert(0, pkg)
import torch, submission
_fno = submission._fno
models, dev = submission._state[0], submission._state[1]
x = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"][:112].astype(np.float32)
def prep(n):
    xb = torch.from_numpy(np.ascontiguousarray(x[:n])).to(dev)[..., :2].float()
    return ((xb - torch.from_numpy(_fno.MEAN).to(dev)) / torch.from_numpy(_fno.STD).to(dev)).permute(0, 1, 4, 2, 3).reshape(n, 40, 32, 64).contiguous()
def tm(m, prev, reps=12):
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16): m(prev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            with torch.autocast("cuda", dtype=torch.float16): o = m(prev)
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / reps / len(prev), o
ff = [m for m in models if type(m).__name__ == "StackedFFNO"][0]
vit = [m for m in models if type(m).__name__ == "StackedViT"][0]
for n in (14, 56, 99):
    prev = prep(n)
    t_ff, o0 = tm(ff, prev); t_v, _ = tm(vit, prev)
    print(f"N {n}: F-FNO stack {t_ff:.3f} ms/win   ViT stack {t_v:.3f} ms/win", flush=True)
prev = prep(56)
t0_, ref = tm(ff, prev)
for tile in (0, 4, 8, 16, 28, 32):
    _fno.FFNO_TILE = tile
    t, o = tm(ff, prev)
    print(f"FFNO_TILE {tile:2d}: {t:.3f} ms/win  maxabs field diff vs shipped {float((o[0].float()-ref[0].float()).abs().max()):.2e}", flush=True)
_fno.FFNO_TILE = 16
ff.set_spectral_path("fft"); t, o = tm(ff, prev)
print(f"fft path (fp32 spectral): {t:.3f} ms/win  maxabs diff {float((o[0].float()-ref[0].float()).abs().max()):.2e}", flush=True)
ff.set_spectral_path("dense")
# channels / TF32 / cudnn knobs
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(12): o = ff(prev)
    torch.cuda.synchronize()
print(f"fp32+TF32 no autocast: {1e3*(time.perf_counter()-t0)/12/56:.3f} ms/win", flush=True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from torch.profiler import profile, ProfilerActivity
for name, m in (("F-FNO", ff), ("ViT", vit)):
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.float16): m(prev)
        torch.cuda.synchronize()
    print(f"==== {name} stack, batch 56, 5 forwards")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=40)[:6000], flush=True)
