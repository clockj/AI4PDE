# T1 stacking / tree-gating probe on held-out twin predictions (LOTO by trajectory). numpy only.
import numpy as np, time
t0=time.time()
z=np.load("../ckpt/c/twins_unseen.npz", allow_pickle=True)
y=z["y"].astype(np.float32); Y=z["Y"].astype(np.float32); sig=z["sig"].astype(np.float32); tid=z["tid"]; mem=list(z["members"])
print("members", mem)
MEAN=np.array([0.154962569,-0.000517793698],np.float32); STD=np.array([0.0968104079,0.0159636438],np.float32)
def rel(p):   # per-window rel-L2 in raw units over u,v
    P=p*STD+MEAN; T=Y*STD+MEAN
    n=np.sqrt(((P-T)**2).reshape(len(P),-1).sum(1)); d=np.sqrt((T**2).reshape(len(T),-1).sum(1))
    return n/d
fam_v=[i for i,m in enumerate(mem) if "vit" in m]; fam_f=[i for i in range(len(mem)) if i not in fam_v]
yA=y[fam_f].mean(0); yB=y[fam_v].mean(0); y0=y.mean(0)
base=rel(y0); trajs=np.unique(tid)
def report(name,p):
    r=rel(p); d=(r-base)
    per=[d[tid==t].mean()/base[tid==t].mean() for t in trajs]
    print(f"{name:42s} rel {r.mean():.5f}  d {100*(r.mean()-base.mean())/base.mean():+.2f}%  traj better {sum(x<0 for x in per)}/{len(trajs)}",flush=True)
print(f"baseline uniform mean rel {base.mean():.5f}")
report("F-FNO family only",yA); report("ViT family only",yB)
# (1) oracle per-window scalar family weight (hindsight ceiling for any window-level gate)
d=yB-yA; e=Y-yA
num=(d*e*STD**2).reshape(len(d),-1).sum(1); den=(d*d*STD**2).reshape(len(d),-1).sum(1)
w=np.clip(num/den,0,1); report("ORACLE per-window family weight", yA+w[:,None,None,None,None]*d)
# (2) LOTO global member weights (least squares, 5 weights + none constrained)
def loto(fn):
    out=np.empty_like(y0)
    for t in trajs:
        tr=tid!=t; te=tid==t; out[te]=fn(tr,te)
    return out
def glob(tr,te):
    A=np.stack([(y[i][tr]*STD).reshape(-1) for i in range(5)],1)[::37]; b=((Y[tr])*STD).reshape(-1)[::37]
    wts=np.linalg.lstsq(A.astype(np.float64),b.astype(np.float64),rcond=None)[0].astype(np.float32)
    return np.tensordot(wts,y[:,te],1)
report("LOTO global 5-member LS weights",loto(glob))
# (3) LOTO per-pixel-per-channel family weight map (static), box-smoothed
def box(a,k=3):
    p=np.pad(a,((0,0),)*(a.ndim-3)+((k//2,k//2),(k//2,k//2),(0,0)),mode="edge"); o=np.zeros_like(a)
    for i in range(k):
        for j in range(k): o+=p[...,i:i+a.shape[-3],j:j+a.shape[-2],:]
    return o/(k*k)
def pix(tr,te):
    n=(d[tr]*e[tr]).sum((0,1)); q=(d[tr]*d[tr]).sum((0,1))
    wm=np.clip(box(n)/np.maximum(box(q),1e-12),0,1); wm=0.4+0.5*(wm-0.4)
    return yA[te]+wm*d[te]
report("LOTO per-pixel family weight map (shrunk)",loto(pix))
# (4) LOTO tree-like gating: piecewise-constant weight on (lead block, spread quantile, sigma quantile, |d| quantile)
spread=y.std(0); sg=np.broadcast_to(sig.mean(0),y0.shape); ad=np.abs(d)
lead=np.broadcast_to((np.arange(20)//4)[None,:,None,None,None],y0.shape)
ch=np.broadcast_to(np.arange(2)[None,None,None,None,:],y0.shape)
def qbin(a,tr,nb):
    out=np.empty(a.shape,np.int16)
    for c in range(2):
        edges=np.quantile(a[tr][...,c].reshape(-1)[::53],np.linspace(0,1,nb+1)[1:-1]); out[...,c]=np.searchsorted(edges,a[...,c])
    return out
def tree(tr,te,nb=6):
    b=(((lead*nb+qbin(spread,tr,nb))*nb+qbin(sg,tr,nb))*nb+qbin(ad,tr,nb))*2+ch
    nbin=5*nb*nb*nb*2
    n=np.bincount(b[tr].reshape(-1),weights=(d[tr]*e[tr]).reshape(-1),minlength=nbin); q=np.bincount(b[tr].reshape(-1),weights=(d[tr]*d[tr]).reshape(-1),minlength=nbin)
    wb=np.clip(n/np.maximum(q,1e-12),0,1).astype(np.float32)
    return yA[te]+wb[b[te]]*d[te]
report("LOTO tree-gate 5x6x6x6 bins (family weight)",loto(tree))
# (5) LOTO tree-like residual correction: piecewise-constant shrink of the fluctuation about the input-window mean
print("elapsed %.0fs"%(time.time()-t0))
