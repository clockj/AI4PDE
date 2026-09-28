# EA: verify StackedFFViT against the member loop (fp32 fft + dense, fp16 autocast), K=2, arm + control.
import sys, os, copy, numpy as np, torch
W = sys.argv[1]; CK = sys.argv[2]; SRC = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, os.path.join(W, "Track_1", "submit_fno"))
import _fno
if SRC:
    exec(open(SRC).read(), _fno.__dict__)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(os.path.join(W, "Track_1", "val_unseen.npz"))
x = torch.from_numpy(((d["x"][:40, ..., :2] - _fno.MEAN) / _fno.STD).astype(np.float32))
x = x.permute(0, 4, 1, 2, 3).reshape(40, -1, 32, 64).to(dev) if False else None
# build the model input the way predict_fno does: (B, 20*C, H, W)
xr = d["x"][:40, ..., :2].astype(np.float32)
xn = (xr - _fno.MEAN) / _fno.STD
xin = torch.from_numpy(xn).permute(0, 1, 4, 2, 3).reshape(40, -1, 32, 64).to(dev)
m1 = _fno.load_any(CK, dev)
m2 = copy.deepcopy(m1)
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for p in m2.parameters():
        p.add_(0.02 * p.abs().mean() * torch.randn(p.shape, generator=g).to(p.device))
    m2.mix.fill_(0.4)
def mk0(m):
    m0 = _fno.FFViT2d(links=False).to(dev)
    sd = {k: v for k, v in m.state_dict().items() if not (k.startswith("f2v") or k.startswith("v2f"))}
    m0.load_state_dict(sd); return m0.eval()
for name, mem in (("arm", [m1, m2]), ("ctl", [mk0(m1), mk0(m2)])):
    for path in ("fft", "dense"):
        for mm in mem: mm.set_spectral_path(path)
        st = _fno.fuse(mem)
        assert len(st) == 1 and type(st[0]).__name__ == "StackedFFViT", [type(s).__name__ for s in st]
        st = st[0]; st.set_spectral_path(path)
        with torch.no_grad():
            outs = [mm(xin) for mm in mem]
            fl = torch.stack([o[0] for o in outs]).mean(0)
            sl = torch.stack([o[1].clamp(-8, 2).exp() for o in outs]).mean(0).log()
            ff, sg = st(xin)
            print(f"{name} {path} fp32: field max|d| {(ff - fl).abs().max().item():.2e} (scale {fl.abs().max().item():.2f})"
                  f"  logstd max|d| {(sg - sl).abs().max().item():.2e}")
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    fa, sa = st(xin)
                print(f"{name} {path} fp16-autocast fused vs fp32 loop: field rel {((fa.float()-fl).norm()/fl.norm()).item():.2e}")
        # predict_fno end to end: fused vs unfused list
        pa = _fno.predict_fno(mem, xr, dev, alpha=0.80)
        pb = _fno.predict_fno([st], xr, dev, alpha=0.80)
        print(f"{name} {path} predict_fno fused vs loop rel {np.linalg.norm(pa-pb)/np.linalg.norm(pa):.2e}")
print("FUSETEST DONE")
