"""Exchange rate: what is an X% across-the-board accuracy gain worth in each subscore?

Held-out 6-member ensemble on the AoA 15-20 proxy, errors synthetically scaled by k
(P' = Y + k*(P-Y)), board tail inflation 1.39x applied, band refit at every k.
"""
import os, sys, numpy as np
HERE="/home/clockj/RealPDE/AI4PDE/Track_1"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE,"submit_fno"))
import torch, _fno, sps9
from sps9 import sc, prep, optimal_band, sps_score
from eval_v9 import infer

F=1.39
CK=["unet/fno_grad.pth","unet/fno_grad_s2.pth","unet/fno_grad_s3.pth",
    "unet/fno_w48.pth","unet/fno_m1220.pth","unet/fno_w64.pth"]
dev="cuda" if torch.cuda.is_available() else "cpu"
m=[_fno.load(os.path.join(HERE,c),dev) for c in CK]
ds=np.load(os.path.join(HERE,"val_seen.npz")); du=np.load(os.path.join(HERE,"val_unseen.npz"))
hs=np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]],[15,20])
hu=np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]],[15,20])
Ps,Ys=infer(m,ds["x"],dev,0.75)[hs],ds["y"][hs]
Pu,Yu=infer(m,du["x"],dev,0.75)[hu],du["y"][hu]
W=dict(rel=0.2657,tke=0.2435,mvpe=0.2406,sps=0.1808)   # mid of admissible family

print(f"{'k':>5} {'relL2':>7} {'TKE':>7} {'MVPE':>7} | {'rel_s':>6} {'tke_s':>6} {'mvp_s':>6} {'sps':>6} | {'dfinal':>7}")
base=None
for k in [1.00,0.95,0.90,0.85,0.80,0.70,0.50]:
    Pk_s=Ys+k*(Ps-Ys); Pk=Yu+k*(Pu-Yu)
    As,as_,ss,_=prep(Pk_s,Ys); Au,_,su,nu=prep(Pk,Yu)
    band=optimal_band(np.float32(F)*As,as_,ss)
    r=float(np.mean(sc.rel_l2_per_sample(Pk,Yu,2))); t=float(np.mean(sc.tke_rel_l2_per_sample(Pk,Yu,2)))
    mv=sc.mvpe_rel_l2(Pk,Yu)
    dm,tk,mvv=sps9.sample_metrics(Pk,Yu)
    a=sps9.weights_from_metrics(F*dm,F*tk,F*mvv)
    s=sps_score(np.float32(F)*Au,a,su,nu,band)
    sub=dict(rel=sc.score_error(r*F),tke=sc.score_error(t*F),mvpe=sc.score_error(mv*F),sps=s)
    fin=sum(W[x]*sub[x] for x in W)
    if base is None: base=fin
    print(f"{k:5.2f} {r:7.4f} {t:7.4f} {mv:7.4f} | {sub['rel']:6.2f} {sub['tke']:6.2f} "
          f"{sub['mvpe']:6.2f} {s:6.2f} | {fin-base:+7.3f}")
