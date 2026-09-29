"""Arm I6: a band refitted on pooled HELD-OUT-condition errors, leave one regime out (post hoc fit, real-path read).

    python ood_i6.py fit --src SRC --target-band X_BAND.npz --out DIR  DUMP[:GROUP] [DUMP[:GROUP] ...]
    python ood_i6.py compare CONTROL_RUN I6_RUN [--groups g ...]

fit: pool the dumped held-out windows of the OTHER regimes (ood_i6dump.py, each from its own ensemble; ':GROUP' keeps one
  group of a dump, e.g. re_mirror/i6/heldout.npz:relo), each window's sigma ratio taken against ITS OWN ensemble's
  sigma_ref, and fit with the shipping construction:
    static  SRC/Track_1/submit_fno/fit_band.py (+ patches/fit_band_from_dump.patch) --from-dump --sigma-ref-from X_BAND
            --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored (inflate 1.39, smoothing 3: defaults)
    SIGBLEND SRC/Track_1/sigblend/make_k0_band.py on the pooled dump (lambda 0.5, inflate 1.29, 16 bins), unchanged
  -> DIR/k0/band_k0_sigblend_lam0.5.{npz,json}, carrying X's own sigma_ref. The package of X's ensemble is then built with
  that band (ood_run.sh with OOD_BAND / OOD_SIGBLEND) and streamed on the real code path.
compare: containment C (x100), sps, mean interval width, composite of the I6 run against the control run, per group, for
  frozen and control (and base), full streams and the first window of each stream, with per-trajectory counts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = os.environ.get("PY", "/users/PAS1187/qwu/.conda/envs/realpde/bin/python")
CLIP_LO, CLIP_HI = 0.3, 3.0


def fit(a):
    import ood_lib
    parts = {k: [] for k in ("F", "S", "y", "P", "R", "ids")}
    desc = []
    for spec in a.dumps:
        path, _, group = spec.partition(":")
        d = np.load(path)
        ids = d["ids"]
        keep = np.ones(len(ids), bool)
        if group:
            trajs = ood_lib.regime(str(d["regime"]))[1][group]
            keep = np.isin(np.array([str(i).split(":")[0] for i in ids]), trajs)
        S = d["S"][0][keep]
        parts["F"].append(d["F"][0][keep]); parts["S"].append(S); parts["y"].append(d["y"][keep])
        parts["P"].append(d["P"][keep]); parts["ids"].append(ids[keep])
        parts["R"].append(np.clip(S / d["sref"][..., :2], CLIP_LO, CLIP_HI).astype(np.float32))
        desc.append(f"{path}{':' + group if group else ''} ({int(keep.sum())} windows, ensemble {str(d['regime'])})")
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    os.makedirs(a.out, exist_ok=True)
    fd = os.path.join(a.out, "pooled_fit.npz")
    kd = os.path.join(a.out, "pooled_k0.npz")
    np.savez(fd, P=cat["P"], C=cat["F"], Y=cat["y"], R=cat["R"])
    np.savez(kd, F=cat["F"][None], S=cat["S"][None], x=np.zeros((len(cat["y"]), 1), np.float32), y=cat["y"],
             ids=cat["ids"], names=np.array(["pooled"]))
    del cat, parts
    print("[ood_i6] pooled " + "; ".join(desc), flush=True)
    static = os.path.join(a.out, "band_static.npz")
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    subprocess.run([PY, os.path.join(a.src, "Track_1/submit_fno/fit_band.py"), "--from-dump", fd, "--sigma-ref-from",
                    a.target_band, "--alpha", "0.80", "--band-p", "0.45", "--centre-alpha", "0", "--fill-unscored",
                    "--out", static], check=True, env=env)
    subprocess.run([PY, "make_k0_band.py", os.path.join(a.out, "k0")], check=True, cwd=os.path.join(a.src, "Track_1/sigblend"),
                   env=dict(env, B_DUMP=kd, K0_BAND=static, K0_INFL="1.29", K0_NB="16",
                            PYTHONPATH=os.path.join(a.src, "Track_1")))
    for f in (fd, kd):
        os.remove(f)
    json.dump({"pooled": desc, "target_band": a.target_band}, open(os.path.join(a.out, "i6_fit.json"), "w"), indent=1)
    print(f"[ood_i6] wrote {a.out}/k0/band_k0_sigblend_lam0.5.npz (+ .json)", flush=True)


def _rec(run, v):
    p = os.path.join(run, "json", v + ".json")
    if not os.path.exists(p):
        return None
    w = json.load(open(p))["windows"]
    return {k: np.asarray(w[k]) for k in w if k not in ("hP", "hL", "hU")}


def compare(a):
    import ood_score
    groups = json.load(open(os.path.join(a.control_run, "json", "control.json")))["groups"]
    out = {}
    L = [f"| group | variant | subset | ΔC×100 (n+) | Δsps (n+) | Δcomposite (n+) | width ratio (I6 / control band) |",
         "|---|---|---|---|---|---|---|"]
    for g, trajs in groups.items():
        if a.groups and g not in a.groups:
            continue
        for v in ("frozen", "base", "control"):
            A, B = _rec(a.control_run, v), _rec(a.i6_run, v)
            if A is None or B is None:
                continue
            assert np.array_equal(A["traj"], B["traj"])
            for sub in ("full", "first window"):
                def agg(R, ts):
                    s = np.isin(R["traj"], ts)
                    if sub != "full":
                        s &= R["k"] == 0
                    return ood_score.aggregate(R, s)
                x, y = agg(A, trajs), agg(B, trajs)
                per = [(agg(B, [t]), agg(A, [t])) for t in trajs]
                cnt = lambda key: sum(p[0][key] > p[1][key] for p in per)  # noqa: E731
                wr = (y["width"] / x["width"]) if ("width" in x and "width" in y) else float("nan")
                e = dict(dC=y["C"] - x["C"], dsps=y["sps"] - x["sps"], dcomp=y["composite"] - x["composite"],
                         nC=cnt("C"), nsps=cnt("sps"), ncomp=cnt("composite"), n=len(trajs), width_ratio=wr)
                out.setdefault(g, {}).setdefault(v, {})[sub] = e
                L.append(f"| {g} | {v} | {sub} | {e['dC']:+.3f} ({e['nC']}/{e['n']}) | {e['dsps']:+.3f} ({e['nsps']}/{e['n']}) | "
                         f"{e['dcomp']:+.4f} ({e['ncomp']}/{e['n']}) | {wr:.3f} |")
    print("\n".join(L))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--src", required=True, help="a source tree with patches/fit_band_from_dump.patch applied")
    f.add_argument("--target-band", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("dumps", nargs="+")
    c = sp.add_parser("compare")
    c.add_argument("control_run")
    c.add_argument("i6_run")
    c.add_argument("--groups", nargs="*", default=None)
    c.add_argument("--out", default=None)
    a = ap.parse_args()
    fit(a) if a.cmd == "fit" else compare(a)


if __name__ == "__main__":
    main()
