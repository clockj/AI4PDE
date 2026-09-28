"""T1 real-code-path twin read: baseline T1 x27 code + the 5 twins (t2twin_asnp.zip members, shipped band), knob off vs on."""
import os, sys, shutil, subprocess, zipfile
import numpy as np
W = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2xr"
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927"
sys.path.insert(0, W + "/Track_1/verify"); sys.path.insert(0, W + "/Track_2"); sys.path.insert(0, W + "/Track_1")
sys.argv = [sys.argv[0]]
import joint_gain as j
T = sys.argv_T = "/tmp/rp_k272_t1twin_" + os.environ.get("SLURM_JOB_ID", "x")
os.makedirs(T, exist_ok=True)
# twin T1 zip = x27 T1 code + twin members/band
tz = os.path.join(T, "t1twin.zip")
with zipfile.ZipFile(tz, "w") as zo:
    zb = zipfile.ZipFile(S + "/sub_candidates/track1_x27_a080.zip")
    for n in ("submission.py", "_fno.py"):
        zo.writestr(n, zb.read(n))
    zt = zipfile.ZipFile(S + "/pkg/x_work/t2twin_asnp.zip")
    for n in zt.namelist():
        if n.endswith(".pth") or n == "band.npz":
            zo.writestr(n, zt.read(n))
V = {"off": {"sub": {"JOINT_GAIN": "False"}, "fno": {"JOINT_GAIN": "False"}},
     "on": {"sub": {"JOINT_GAIN": "True"}, "fno": {"JOINT_GAIN": "True"}}}
dirs = j.stage(tz, T, 1, V)
R = {}
for n in ("off", "on"):
    o = os.path.join(T, f"{n}.npz")
    subprocess.run([sys.executable, "-u", W + "/Track_1/verify/joint_gain.py", "--run-t1", dirs[n], o], check=True)
    R[n] = np.load(o)
import stream_eval as se
d = np.load(W + "/Track_1/val_unseen.npz")
Y = d["y"][..., :2].astype(np.float32); names = np.array([s.split(":")[0] for s in d["ids"]])
aoa = np.array([int(n.split("_")[1]) for n in names])
WT = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
fin = lambda q: sum(WT[k] * q[k] for k in WT)
pad = lambda a: np.concatenate([a, np.zeros(a.shape[:-1] + (1,), a.dtype)], -1)
for sel_name, sel in (("all8", np.ones(len(Y), bool)), ("aoa15-20", np.isin(aoa, [15, 20]))):
    ds = {k: se.score(pad(z["pred"][sel]), pad(z["lower"][sel]), pad(z["upper"][sel]), pad(Y[sel]), 1e-3) for k, z in R.items()}
    a, b = ds["off"], ds["on"]
    for lab, q in (("OFF", a), ("ON ", b)):
        print(f"T1 {sel_name:9s} {lab} rel {q['rel_err']:.5f} tke {q['tke_err']:.5f} mvpe {q['mvpe_err']:.5f} | scores rel {q['rel']:.4f} "
              f"tke {q['tke']:.4f} mvpe {q['mvpe']:.4f} sps {q['sps']:.4f} final4 {fin(q):.4f}")
    print(f"T1 {sel_name:9s} d   rel_err {100*(b['rel_err']/a['rel_err']-1):+.3f}% tke_err {100*(b['tke_err']/a['tke_err']-1):+.4f}% "
          f"mvpe_err {100*(b['mvpe_err']/a['mvpe_err']-1):+.4f}% | d rel {b['rel']-a['rel']:+.4f} tke {b['tke']-a['tke']:+.4f} "
          f"mvpe {b['mvpe']-a['mvpe']:+.4f} sps {b['sps']-a['sps']:+.4f} final4 {fin(b)-fin(a):+.4f}")
    tr = list(dict.fromkeys(names[sel])); nrel = nfin = 0
    for n in tr:
        m = names == n
        p = {k: se.score(pad(z["pred"][m]), pad(z["lower"][m]), pad(z["upper"][m]), pad(Y[m]), 1e-3) for k, z in R.items()}
        nrel += p["on"]["rel_err"] < p["off"]["rel_err"]; nfin += fin(p["on"]) > fin(p["off"])
    print(f"T1 {sel_name:9s} per-trajectory: rel err down {nrel}/{len(tr)}, final4 up {nfin}/{len(tr)}")
shutil.rmtree(T, ignore_errors=True)
