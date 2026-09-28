# EA: micro-benchmark fused vs unfused ffvit forward (fp16 autocast), B=112 dense and B=1 fft.
import sys, os, copy, time, torch
W = sys.argv[1]; CK = sys.argv[2]
sys.path.insert(0, os.path.join(W, "Track_1", "submit_fno"))
import _fno
dev = "cuda"
m1 = _fno.load_any(CK, dev); m2 = copy.deepcopy(m1)
def tm(fn, x, n=50):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5): fn(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3
for B, path in ((112, "dense"), (1, "fft"), (1, "dense")):
    x = torch.randn(B, 40, 32, 64, device=dev)
    for m in (m1, m2): m.set_spectral_path(path)
    st = _fno.StackedFFViT([m1, m2]).eval(); st.set_spectral_path(path)
    sf = _fno.StackedFFNO([m1.f, m2.f]).eval(); sf.set_spectral_path(path)
    sv = _fno.StackedViT([m1.v, m2.v]).to(dev).eval()
    r = {}
    r["unfused 2 hybrids"] = tm(lambda x: (m1(x), m2(x)), x)
    r["fused StackedFFViT"] = tm(st, x)
    r["StackedFFNO(f)+StackedViT(v) no links"] = tm(lambda x: (sf(x), sv(x)), x)
    r["1 hybrid"] = tm(m1, x)
    r["1 f"] = tm(m1.f, x); r["1 v"] = tm(m1.v, x)
    print(f"B={B} {path}: " + " | ".join(f"{k} {v:.3f} ms" for k, v in r.items()))
