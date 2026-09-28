"""Design B2 unit test (CPU): the PATCHED submission.py scaler (R1_TSM / R1_BLK_N0) against the lab's kfields on a cached stream.
Also: knobs off -> the patched class returns exactly what the as-copied class returns.   python b2unit.py [n0_blk] [tag]"""
import importlib.util, os, sys
import numpy as np, torch
import b2lab as L
HERE = os.path.dirname(os.path.abspath(__file__))
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "submit")); sp.loader.exec_module(m); return m
new = load(os.path.join(os.path.dirname(HERE), "submit", "submission.py"), "sub_new")
old = load(L.S + "/ckpt/b2/submission_c_asof.py", "sub_old")
n0b = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0; tag = sys.argv[2] if len(sys.argv) > 2 else "in_d30_f129"
Z = np.load(f"{L.S}/ckpt/b2/{tag}.npz"); nW = 30
P, C, H, Y = (torch.from_numpy(Z[k][:nW]) for k in ("P", "C", "h", "Y"))
for m in (new, old):
    m.R1_ASYM = True; m.R1_FP32 = True
def scaler(m, tsm, nb):
    if hasattr(m, "R1_TSM"):
        m.R1_TSM = tsm; m.R1_BLK_N0 = nb; m.R1_BLK = 4
    return m._CausalBandScalerT(m.R1_KS, 2.0, [1.0, 1.0], m._SIGMA_G, "cpu", (32, 64, 2))
a_off, a_old, a_on = scaler(new, 1, 0.0), scaler(old, 1, 0.0), None
off_same = True
for i in range(nW):
    for r in (a_off, a_old):
        r.update_gated_asym(P[i][None], C[i][None], H[i][None], Y[i][None], (Y[i] != 0)[None], torch.ones((), dtype=torch.float64))
    k1, k2 = a_off.kfield_asym(), a_old.kfield_asym()
    off_same &= bool(torch.equal(k1[0], k2[0]) and torch.equal(k1[1], k2[1]) and k1[0].shape == (1, 1, 32, 64, 2))
print("knobs OFF: patched class == as-copied class, bitwise, over", nW, "windows:", off_same)
a_on = scaler(new, 3, n0b)
v = L.V(tsm=3, n0e=n0b, blk=4); ks = torch.tensor([2.0 ** (j / 4.0) for j in range(-4, 5)], dtype=torch.float64); K = 9
st = dict(n=0, cum2=torch.zeros((K, K, 32, 64, 2)), cumE=torch.zeros((K, K, 20, 32, 64, 2)), cumT=None)
worst = 0.0
for i in range(nW):
    klo, khi = a_on.kfield_asym() if i > 0 else (torch.ones(1, 20, 32, 64, 2), torch.ones(1, 20, 32, 64, 2))
    l_lo, l_hi = L.kfields(st, v, ks, K)
    l_lo = l_lo.expand(20, 32, 64, 2) if l_lo.shape[0] == 1 else l_lo; l_hi = l_hi.expand(20, 32, 64, 2) if l_hi.shape[0] == 1 else l_hi
    d = max(float((klo[0].double() - l_lo).abs().max()), float((khi[0].double() - l_hi).abs().max()))
    frac = float(((klo[0].double() - l_lo).abs() > 1e-4).double().mean())
    worst = max(worst, d)
    if i in (1, 2, 5, 29):
        print(f"window {i}: max |k_patch - k_lab| {d:.2e}  (elements differing by > 1e-4: {frac:.2%})  k range [{float(klo.min()):.3f}, {float(khi.max()):.3f}] shape {tuple(klo.shape)}")
    assert torch.isfinite(klo).all() and torch.isfinite(khi).all() and float(klo.min()) >= 0.5 - 1e-6 and float(khi.max()) <= 2.0 + 1e-6
    a_on.update_gated_asym(P[i][None], C[i][None], H[i][None], Y[i][None], (Y[i] != 0)[None], torch.ones((), dtype=torch.float64))
    A, B = L.tables(P[i], C[i], H[i], Y[i], ks.float())
    st["cum2"] += torch.einsum("athwc,bthwc->abhwc", A, B); st["cumE"] += A[:, None] * B[None]; st["n"] += 1
print("knobs ON (TSM 3, BLK_N0 %g): worst max |k_patch - k_lab| = %.2e" % (n0b, worst))
