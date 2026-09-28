def B(**k):
    d = dict(mode="ebg", n0=30.0, lead="rank1", n0l=5.0); d.update(k); return d
ARMS2 = {
    "K0(D1c10+VC30)": dict(bias=dict(mode="d1c", n0=10.0), vc_n0=30.0),
    "S1:EBG30+r1+VC30": dict(bias=B(), vc_n0=30.0),
    "C2:EBG30(ebn1)+r1+VC30": dict(bias=B(eb_n0=1.0), vc_n0=30.0),
    "C2v50:EBG30(ebn1)+r1+VC50": dict(bias=B(eb_n0=1.0), vc_n0=50.0),
    "CL:D1c30+r1+VC50": dict(bias=dict(mode="d1c", n0=30.0, lead="rank1", n0l=5.0), vc_n0=50.0),
}
