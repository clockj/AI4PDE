"""LayerNorm kernel replacement in the fused stacks (no retraining): timing + output difference, stack- and package-level."""
import sys, os, time
import numpy as np
pkg = sys.argv[1]
os.chdir(pkg); sys.path.insert(0, pkg)
import torch, submission
_fno = submission._fno
_F = torch.nn.functional
models, dev = submission._state[0], submission._state[1]
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
x, ytrue = z["x"].astype(np.float32), z["y"].astype(np.float32)
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
LN = _fno._StkLN
orig = LN.forward
def ln_a(self, t):
    x32 = t.float()
    xc = x32 - x32.mean(-1, keepdim=True)
    var = (xc * xc).mean(-1, keepdim=True)
    return torch.addcmul(self.b, xc * torch.rsqrt(var + self.eps), self.w)
def ln_c(self, t):
    x32 = t.float()
    var, mu = torch.var_mean(x32, dim=-1, keepdim=True, unbiased=False)
    return torch.addcmul(self.b, (x32 - mu) * torch.rsqrt(var + self.eps), self.w)
def ln_d(self, t):   # moments in fp32, one fused affine: y = x*a + c with a = w*rsqrt, c = b - mu*a
    x32 = t.float()
    var, mu = torch.var_mean(x32, dim=-1, keepdim=True, unbiased=False)
    a = torch.rsqrt(var + self.eps)
    return torch.addcmul(self.b, (x32 - mu).mul_(a), self.w)
ff = [m for m in models if type(m).__name__ == "StackedFFNO"][0]
vit = [m for m in models if type(m).__name__ == "StackedViT"][0]
prev = prep(56)
def run_pkg(n=56):
    ts = []
    for r in range(8):
        if hasattr(submission, "_POOL_USED"): submission._POOL_USED = False
        t0 = time.perf_counter(); o = submission.predict(np.ascontiguousarray(x[:n])); ts.append(1e3 * (time.perf_counter() - t0) / n)
    return float(np.median(ts[2:])), o
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
def score(o, n):
    p = o["prediction"][..., :2]; t = ytrue[:n][..., :2]
    e = np.sqrt(((p - t) ** 2).reshape(n, -1).sum(1)) / np.sqrt((t ** 2).reshape(n, -1).sum(1))
    inside = ((t >= o["lower"][..., :2]) & (t <= o["upper"][..., :2]) & (t != 0))
    c = np.where(inside, np.exp(-(o["upper"][..., :2] - o["lower"][..., :2]) / 0.0563870259), 0.0)
    return float(e.mean()), float(c.mean())
t_ff0, r_ff = tm(ff, prev); t_v0, r_v = tm(vit, prev)
p0, o0 = run_pkg(56); p0b, o0b = run_pkg(99)
s0 = score({k: np.array(v) for k, v in o0b.items()}, 99)
o0 = {k: np.array(v) for k, v in o0.items()}; o0b = {k: np.array(v) for k, v in o0b.items()}
print(f"shipped LN : F-FNO {t_ff0:.3f}  ViT {t_v0:.3f} ms/win | package N56 {p0:.3f} N99 {p0b:.3f} | all-data(contaminated) N99 rel {s0[0]:.5f} C {s0[1]:.5f}", flush=True)
for name, fn in (("ln_a", ln_a), ("ln_c", ln_c), ("ln_d", ln_d)):
    LN.forward = fn
    t_ff, o_ff = tm(ff, prev); t_v, o_v = tm(vit, prev)
    p1, o1 = run_pkg(56); p1b, o1b = run_pkg(99)
    s1 = score(o1b, 99)
    print(f"{name}: F-FNO {t_ff:.3f} ({100*(t_ff/t_ff0-1):+.1f}%)  ViT {t_v:.3f} ({100*(t_v/t_v0-1):+.1f}%) | package N56 {p1:.3f} ({p1-p0:+.3f}) N99 {p1b:.3f} ({p1b-p0b:+.3f})"
          f" | stack field maxabs {float((o_ff[0].float()-r_ff[0].float()).abs().max()):.2e} | pkg pred rel-diff {rel(o1['prediction'], o0['prediction']):.2e} maxabs {float(np.abs(o1['prediction']-o0['prediction']).max()):.2e}"
          f" lower maxabs {float(np.abs(o1['lower']-o0['lower']).max()):.2e} | N99 rel {s1[0]:.5f} C {s1[1]:.5f}", flush=True)
LN.forward = orig
