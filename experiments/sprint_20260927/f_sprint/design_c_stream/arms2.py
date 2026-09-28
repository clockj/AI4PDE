C = [0.31, 0.12]
def B(**k):
    d = dict(mode="ebg", n0=30.0); d.update(k); return d
ARMS2 = {
    "K0(D1c10+VC30)": dict(bias=dict(mode="d1c", n0=10.0), vc_n0=30.0),
    "EBGc": dict(bias=B(c=C, n0s=3.0)),
    "EBGc+r1": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=5.0)),
    "EBGc10+r1": dict(bias=B(c=C, n0s=10.0, lead="rank1", n0l=5.0)),
    "S1:EBG+r1+VC30": dict(bias=B(lead="rank1", n0l=5.0), vc_n0=30.0),
    "S2:EBGc+r1+VC30": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=5.0), vc_n0=30.0),
    "S3:EBGc+r1(n0l2)+VC30": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=2.0), vc_n0=30.0),
    "S4:EBGc+r1(n0l12)+VC30": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=12.0), vc_n0=30.0),
    "S5:EBGc+r1+VC30g5": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=5.0), vc_n0=30.0, vc_g_n0=5.0),
    "S6:EBGc+r1+VC50": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=5.0), vc_n0=50.0),
    "S7:EBGc+r1+VC30+FW8": dict(bias=B(c=C, n0s=3.0, lead="rank1", n0l=5.0), vc_n0=30.0, fw_n0=8.0),
    "S8:EBGc(kse2)+r1+VC30": dict(bias=B(c=C, n0s=3.0, kse=2.0, lead="rank1", n0l=5.0), vc_n0=30.0),
    "S9:EBGc+VC30": dict(bias=B(c=C, n0s=3.0), vc_n0=30.0),
}
