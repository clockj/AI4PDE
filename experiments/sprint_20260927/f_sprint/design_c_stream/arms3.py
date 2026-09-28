def B(**k):
    d = dict(mode="ebg", n0=30.0, lead="rank1", n0l=5.0); d.update(k); return d
ARMS2 = {
    "K0(D1c10+VC30)": dict(bias=dict(mode="d1c", n0=10.0), vc_n0=30.0),
    "S1:EBG30+r1+VC30": dict(bias=B(), vc_n0=30.0),
    "F20:EBG20+r1+VC30": dict(bias=B(n0=20.0), vc_n0=30.0),
    "F15:EBG15+r1+VC30": dict(bias=B(n0=15.0), vc_n0=30.0),
    "F10:EBG10+r1+VC30": dict(bias=B(n0=10.0), vc_n0=30.0),
    "S1_kse05": dict(bias=B(kse=0.5), vc_n0=30.0),
    "S1_kse2": dict(bias=B(kse=2.0), vc_n0=30.0),
    "S1_ebn1": dict(bias=B(eb_n0=1.0), vc_n0=30.0),
    "S1_VC20": dict(bias=B(), vc_n0=20.0),
    "S1_VC50": dict(bias=B(), vc_n0=50.0),
    "S1_nolead": dict(bias=B(lead="none"), vc_n0=30.0),
    "D1c10+r1+VC30": dict(bias=dict(mode="d1c", n0=10.0, lead="rank1", n0l=5.0), vc_n0=30.0),
    "D1c30+r1+VC50": dict(bias=dict(mode="d1c", n0=30.0, lead="rank1", n0l=5.0), vc_n0=50.0),
    "S1+FW8": dict(bias=B(), vc_n0=30.0, fw_n0=8.0),
}
