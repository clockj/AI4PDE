"""Load a twin export written by export.py. Pure numpy.

    r = load(path);  r["y"], r["x"], r["e_a0"], r["e_a1"]  float32, (N,20,32,64,2)
    pred(r, a) = r["y"] + (1-a)*r["e_a0"] + a*r["e_a1"]   (compose is linear in ALPHA)
"""
import numpy as np


def load(path):
    d = np.load(path)
    if "y" in d.files:                    # export.py: x, y stored exactly in f32
        y, x = d["y"].astype(np.float32), d["x"].astype(np.float32)
    else:                                 # assemble.py format: time-mean f32 + fluctuation f16
        y = (d["y_m"] + d["y_f"].astype(np.float32)).astype(np.float32)
        zero = np.unpackbits(d["y_zero"], count=y.size).reshape(y.shape).astype(bool)
        y[zero] = 0.0
        x = (d["x_m"] + d["x_f"].astype(np.float32)).astype(np.float32)
    out = {k: d[k] for k in ("ids", "sig_twin", "sig_full", "full", "twin")}
    out["sig_full_is_twin"] = int(d["sig_full_is_twin"]) if "sig_full_is_twin" in d.files else 0
    out.update(x=x, y=y,
               e_a0=d["e_a0"].astype(np.float32), e_a1=d["e_a1"].astype(np.float32))
    return out


def pred(r, a):
    return r["y"] + (1.0 - a) * r["e_a0"] + a * r["e_a1"]


def load_members(path):
    """members_unseen.npz -> e_mem (K,N,20,32,64,2) and sig_mem (K,N,1,32,64,2), float32."""
    d = np.load(path)
    return {"ids": d["ids"], "twin": d["twin"], "e_mem": d["e_mem"].astype(np.float32),
            "sig_mem": d["sig_mem"]}
