"""Arm I7 screen (zero propagation), post hoc on dumped predictions (ood_stream.py --dump).

    python ood_i7.py RUN              # reads RUN/dump/{frozen,control}[__m1|__m2|__m3]/, writes RUN/i7.md, RUN/i7.json

Stage A (where the squared error sits):
  classes of an element (window, pixel, channel) by its state in the window's own INPUT: S20 = the exact physical zero in
  all 20 input frames; LAST = zero in the last input frame but not in all 20; SOME = zero in some input frame but not the
  last; NEVER = never zero in the input. Item 1: for elements whose TARGET is zero, per class, the share of elements, the
  share of the squared error sum (P - Y)^2 over u and v (the numerator of rel), mean |P|. Item 2: S20 elements whose target
  is not zero, by lead time. Item 3: P(target zero at lead t | zero in the last input frame), P(... | S20). Item 4: item 1's
  target-zero share for control by revealed windows n (0, 1-3, 4-13, 14+).
Stage B (the rule, applied to P and the interval exactly as the package's containment would: where P is set to 0,
  lower = min(lower, 0), upper = max(upper, 0)), per channel:
  Z1 P = 0 at every output frame where the element is S20; Z2_k the same for the first k output frames only;
  Z3_m P = 0 where the element is zero in the last m input frames (all output frames).
  Rescored with the kit's per-window decomposition (ood_score); deltas to control and to frozen, per subscore, counts,
  first 3 / first 14 windows, and on the re-blanked streams the share of the control's blanking loss recovered.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_score  # noqa: E402

CLASSES = ("S20", "LAST", "SOME", "NEVER")
NB = [("0", 0, 0), ("1-3", 1, 3), ("4-13", 4, 13), ("14+", 14, 10 ** 6)]
RULES = [("Z1", "s20", 20), ("Z2_k5", "s20", 5), ("Z2_k10", "s20", 10), ("Z3_m1", 1, 20), ("Z3_m5", 5, 20),
         ("Z3_m10", 10, 20)]
SUBS = [("full", None), ("first3", 3), ("first14", 14)]
PROBE_Y = [16 + 2 * j for j in range(-4, 5)]
PROBE_X = [13, 21, 29, 37]


def load(d):
    g = lambda n: np.load(os.path.join(d, n + ".npy"), mmap_mode=None)  # noqa: E731
    traj = g("traj")
    k = np.zeros(len(traj), int)
    for i in range(1, len(traj)):
        k[i] = k[i - 1] + 1 if traj[i] == traj[i - 1] else 0
    return dict(P=g("P"), L=g("L"), U=g("U"), Y=g("Y"), zt=g("ztrail"), za=g("zany"), traj=traj, k=k)


def classes(D):
    zt, za = D["zt"], D["za"]
    return {"S20": zt == 20, "LAST": (zt >= 1) & (zt < 20), "SOME": (zt == 0) & za, "NEVER": ~za}


def stage_a(D, sel):
    P, Y = D["P"][sel], D["Y"][sel]
    C = {c: m[sel] for c, m in classes(D).items()}
    se = (P.astype(np.float64) - Y) ** 2
    tot = se.sum()
    yz = Y == 0.0
    n_all = Y.size
    out = {"target_zero_elem": float(yz.mean()), "target_zero_se": float(se[yz].sum() / tot)}
    for c in CLASSES:
        m = yz & C[c][:, None]                                    # class is constant over the 20 output frames
        out[c] = dict(elem=float(m.sum() / n_all), se=float(se[m].sum() / tot),
                      mean_abs_p=float(np.abs(P[m]).mean()) if m.any() else float("nan"))
    s20 = np.broadcast_to(C["S20"][:, None], Y.shape)
    last = np.broadcast_to((D["zt"][sel] >= 1)[:, None], Y.shape)
    ns20 = s20.sum(axis=(0, 2, 3, 4))
    out["s20_target_nonzero_by_t"] = ((s20 & ~yz).sum(axis=(0, 2, 3, 4)) / np.maximum(ns20, 1)).tolist()
    out["p_zero_given_s20_by_t"] = ((s20 & yz).sum(axis=(0, 2, 3, 4)) / np.maximum(ns20, 1)).tolist()
    out["p_zero_given_last_by_t"] = ((last & yz).sum(axis=(0, 2, 3, 4)) / np.maximum(last.sum(axis=(0, 2, 3, 4)), 1)).tolist()
    out["s20_elem_frac"] = float(C["S20"].mean())
    return out


def rule_mask(D, rule):
    _, cond, kf = rule
    base = D["zt"] == 20 if cond == "s20" else D["zt"] >= cond          # (N, H, W, 2)
    m = np.zeros(D["P"].shape, bool)
    m[:, :kf] = base[:, None]
    return m


def apply(D, m):
    P, L, U = D["P"].copy(), D["L"].copy(), D["U"].copy()
    P[m] = 0.0
    L[m] = np.minimum(L[m], 0.0)
    U[m] = np.maximum(U[m], 0.0)
    return P, L, U


def records(P, L, U, Y):
    return ood_score.window_records(P, L, U, Y)


def agg(rec, traj, k, trajs, kmax=None):
    s = np.isin(traj, trajs)
    if kmax is not None:
        s &= k < kmax
    return ood_score.aggregate(rec, s)


def block(rec, ref, D, trajs):
    """deltas of rec against ref over the group: pooled full / first3 / first14, per subscore, counts."""
    out = {}
    for sname, km in SUBS:
        a, b = agg(rec, D["traj"], D["k"], trajs, km), agg(ref, D["traj"], D["k"], trajs, km)
        out[sname] = {x: a[x] - b[x] for x in ("composite", "rel", "tke", "mvpe", "sps", "C")}
        out[sname]["level"] = a["composite"]
    pos = [agg(rec, D["traj"], D["k"], [t])["composite"] > agg(ref, D["traj"], D["k"], [t])["composite"] for t in trajs]
    neg = [agg(rec, D["traj"], D["k"], [t])["composite"] < agg(ref, D["traj"], D["k"], [t])["composite"] for t in trajs]
    out["npos"], out["nneg"], out["n"] = int(sum(pos)), int(sum(neg)), len(trajs)
    return out


def main():
    run = sys.argv[1]
    groups = json.load(open(os.path.join(run, "json", "control.json")))["groups"]
    dd = os.path.join(run, "dump")
    sets = sorted({d.partition("__")[2] for d in os.listdir(dd)})
    res = {"groups": groups, "A": {}, "B": {}, "probe": {}, "check": {}}
    ctrl_plain = {}
    for m in sets:
        suf = "__" + m if m else ""
        Dc = load(os.path.join(dd, "control" + suf))
        Df = load(os.path.join(dd, "frozen" + suf))
        assert np.array_equal(Dc["traj"], Df["traj"]) and np.array_equal(Dc["Y"], Df["Y"]), "frozen / control dumps differ in data"
        rc = records(Dc["P"], Dc["L"], Dc["U"], Dc["Y"])
        rf = records(Df["P"], Df["L"], Df["U"], Df["Y"])
        # check: the dump reproduces the stream's own json
        for v, r in (("control", rc), ("frozen", rf)):
            jp = os.path.join(run, "json", f"{v}{suf}.json")
            if os.path.exists(jp) and "rel" in json.load(open(jp))["windows"]:
                w = json.load(open(jp))["windows"]
                res["check"][f"{v}{suf}"] = float(max(np.max(np.abs(np.asarray(w[x]) - r[x])) for x in ("rel", "tke", "mvpe", "con")))
        tag = m or "plain"
        res["A"][tag], res["B"][tag], res["probe"][tag] = {}, {}, {}
        for g, trajs in groups.items():
            sel = np.isin(Dc["traj"], trajs)
            res["A"][tag][g] = {"control": stage_a(Dc, sel), "frozen": stage_a(Df, sel)}
            if not m:
                res["A"][tag][g]["control_by_n"] = {}
                for bn, lo, hi in NB:
                    s2 = sel & (Dc["k"] >= lo) & (Dc["k"] <= hi)
                    if s2.any():
                        a = stage_a(Dc, s2)
                        res["A"][tag][g]["control_by_n"][bn] = dict(target_zero_se=a["target_zero_se"],
                                                                     S20_se=a["S20"]["se"], windows=int(s2.sum()))
                ctrl_plain[g] = agg(rc, Dc["traj"], Dc["k"], trajs)["composite"]
            res["B"][tag][g] = {"control": agg(rc, Dc["traj"], Dc["k"], trajs), "frozen": agg(rf, Df["traj"], Df["k"], trajs)}
        for rule in RULES:
            mk = rule_mask(Dc, rule)
            pr = np.zeros(mk.shape[:-1] + (2,), bool)
            res["probe"][tag][rule[0]] = int(mk[:, :, PROBE_Y][:, :, :, PROBE_X].sum())
            rcz = records(*apply(Dc, mk), Dc["Y"])
            rfz = records(*apply(Df, mk), Df["Y"])
            for g, trajs in groups.items():
                e = res["B"][tag][g]
                e[rule[0]] = {"vs_control": block(rcz, rc, Dc, trajs), "frozen_vs_frozen": block(rfz, rf, Df, trajs),
                              "touched_frac": float(mk[np.isin(Dc["traj"], trajs)].mean()),
                              "touched_target_nonzero_frac": float((mk & (Dc["Y"] != 0))[np.isin(Dc["traj"], trajs)].mean())}
                if m and g in ctrl_plain:
                    lvl = agg(rcz, Dc["traj"], Dc["k"], trajs)["composite"]
                    loss = ctrl_plain[g] - e["control"]["composite"]
                    e[rule[0]]["recovered_share"] = float((lvl - e["control"]["composite"]) / loss) if loss else None
            del rcz, rfz
        print(f"[ood_i7] {run}: set {tag} done", flush=True)
    json.dump(res, open(os.path.join(run, "i7.json"), "w"), indent=1)
    write_md(run, res)


def write_md(run, res):
    L = [f"# Arm I7 screen: `{os.path.basename(run.rstrip('/'))}`", ""]
    L += ["Dump vs stream json, max |diff| of per-window records: " +
          ", ".join(f"{k} {v:.1e}" for k, v in res["check"].items()), ""]
    for tag, G in res["A"].items():
        for g, a in G.items():
            L += [f"## Stage A, {tag}, {g}", "",
                  "| variant | target-zero elements | their share of SE | " +
                  " | ".join(f"{c}: elem / SE / mean abs P" for c in CLASSES) + " |", "|---|---|---|" + "---|" * 4]
            for v in ("frozen", "control"):
                x = a[v]
                L.append(f"| {v} | {x['target_zero_elem']:.4f} | {x['target_zero_se']:.4f} | " + " | ".join(
                    f"{x[c]['elem']:.4f} / {x[c]['se']:.4f} / {x[c]['mean_abs_p']:.4f}" for c in CLASSES) + " |")
            x = a["control"]
            L += ["", "| lead t | " + " | ".join(str(t) for t in range(20)) + " |", "|---|" + "---|" * 20,
                  "| S20: target NOT zero | " + " | ".join(f"{v:.3f}" for v in x["s20_target_nonzero_by_t"]) + " |",
                  "| P(target 0 given S20) | " + " | ".join(f"{v:.3f}" for v in x["p_zero_given_s20_by_t"]) + " |",
                  "| P(target 0 given zero in last input frame) | " + " | ".join(f"{v:.3f}" for v in x["p_zero_given_last_by_t"]) + " |", ""]
            if "control_by_n" in a:
                L += ["| n (control) | windows | target-zero share of SE | S20 part |", "|---|---|---|---|"]
                for bn, b in a["control_by_n"].items():
                    L.append(f"| {bn} | {b['windows']} | {b['target_zero_se']:.4f} | {b['S20_se']:.4f} |")
                L.append("")
    for tag, G in res["B"].items():
        for g, B in G.items():
            L += [f"## Stage B, {tag}, {g} (control {B['control']['composite']:.4f}, frozen {B['frozen']['composite']:.4f})", "",
                  "| rule | touched % (target ≠ 0 %) | Δ composite vs control (n+ / n−) | rel | tke | mvpe | sps | ΔC | first 3 / 14 | "
                  "Track 1: Δ on frozen (n+) | recovered share of blanking loss |", "|---|" + "---|" * 10]
            for r, _, _ in RULES:
                e = B[r]; c = e["vs_control"]; f = e["frozen_vs_frozen"]
                rs = e.get("recovered_share")
                L.append(f"| {r} | {100 * e['touched_frac']:.2f} ({100 * e['touched_target_nonzero_frac']:.2f}) | "
                         f"{c['full']['composite']:+.4f} ({c['npos']}/{c['n']} / {c['nneg']}) | {c['full']['rel']:+.3f} | "
                         f"{c['full']['tke']:+.3f} | {c['full']['mvpe']:+.3f} | {c['full']['sps']:+.3f} | {c['full']['C']:+.3f} | "
                         f"{c['first3']['composite']:+.4f} / {c['first14']['composite']:+.4f} | "
                         f"{f['full']['composite']:+.4f} ({f['npos']}/{f['n']}) | {'' if rs is None else f'{rs:.2f}'} |")
            L.append("")
    L += ["mvpe probe elements touched (window x probe pixel x channel x lead): " +
          "; ".join(f"{t}: " + ", ".join(f"{r} {n}" for r, n in P.items()) for t, P in res["probe"].items()), ""]
    open(os.path.join(run, "i7.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
