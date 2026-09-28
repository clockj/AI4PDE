"""Tables for one ensemble run: pooled per group, per trajectory, deltas to control and to base, first 3 / first 14 windows.

    python ood_report.py RUN            # reads RUN/json/*.json, writes RUN/report.md and RUN/summary.json

json/<variant>.json is the plain stream; json/<variant>__<mask>.json the same package on a re-blanked stream
(ood_reblank.py). Each mask set is reported on its own, with deltas to the `control` of the SAME set.

Composite = final_v9.W0 . (rel, tke, mvpe, sps), no time term. A delta is the pooled difference; (n/N) counts trajectories
whose own difference is > 0. Identity: on the first window of every trajectory each variant's pred / lower / upper must be
bit-identical to `frozen` (every correction is shrunk by n/(n+N0), n = 0).

"By revealed windows": composite of base and control over the windows that had n = 0, 1, 2, 3, 4-7, 8-13, 14-27, 28+
revealed targets before them (n = the window's index in its stream), and the arm-I4 hindsight bound: the gain of the whole
stream if every window with n <= 2 had the control - base gain of the 8-13 bin, to first order
sum_b (N_b / N) (g_8-13 - g_b), b = 0, 1, 2 -- on full streams and on the first 14 windows of each stream.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_score  # noqa: E402

ORDER = ["frozen", "base", "control", "prev_input", "statsmask", "famw", "record"]
SUBSETS = [("full", None), ("first3", 3), ("first14", 14)]
NBINS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), ("8-13", 8, 13), ("14-27", 14, 27),
         ("28+", 28, 10 ** 6)]


def _key(v):
    return (ORDER.index(v) if v in ORDER else 50, v)


def load(run):
    d = os.path.join(run, "json")
    sets = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            v, _, m = f[:-5].partition("__")
            sets.setdefault(m, {})[v] = json.load(open(os.path.join(d, f)))
    return {m: dict(sorted(V.items(), key=lambda kv: _key(kv[0]))) for m, V in sorted(sets.items())}


def rec(j):
    w = j["windows"]
    return ({k: np.asarray(w[k]) for k in ("rel", "tke", "mvpe", "a", "con", "ns", "ins", "k", "wsum") if k in w},
            np.asarray(w["traj"]))


def sel(r, names, trajs, kmin=0, kmax=None):
    m = np.isin(names, trajs) & (r["k"] >= kmin)
    if kmax is not None:
        m &= r["k"] < kmax
    return m


def variant_table(V):
    L = ["| variant | PREV_SPACE | STATS_MASK | R1 | R1_N0 | TSM | BLK_N0 | D1C_N0 | EB_N0 | LEAD_N0 | VC_N0 | FW_N0 | "
         "active r1/d1c/vc/fw | groups | graph | ms/step (median) | causal | selfcheck max |",
         "|---|" + "---|" * 17]
    for v, j in V.items():
        kn, ac = j["knobs"], j["active"]
        ca = "-" if not j["causal"] else f"OK ({j['causal']['steps']})"
        L.append(f"| {v} | {kn['PREV_SPACE']} | {kn['STATS_MASK']} | {kn['R1']} | {kn['R1_N0']} | {kn['R1_TSM']} | "
                 f"{kn['R1_BLK_N0']} | {kn['D1C_N0']} | {kn['D1C_EB_N0']} | {kn['D1C_LEAD_N0']} | {kn['VC_N0']} | "
                 f"{kn['FW_N0']} | {int(ac['r1'])}/{int(ac['d1c'])}/{int(ac['vc'])}/{int(ac['fw'])} | {ac['groups']} | "
                 f"{ac['graph']} | {j['step_ms_mean']:.3f} ({j['step_ms_median']:.3f}) | {ca} | "
                 f"{max(j['selfcheck'].values()):.1e} |")
    return L


def identity(V):
    if "frozen" not in V:
        return []
    first = np.asarray(V["frozen"]["windows"]["k"]) == 0
    L = []
    for v, j in V.items():
        if v == "frozen":
            continue
        ok = {x: int(sum(a == b for a, b, f in zip(j["windows"][x], V["frozen"]["windows"][x], first) if f))
              for x in ("hP", "hL", "hU")}
        L.append(f"* `{v}` vs `frozen`, first window of each of {int(first.sum())} trajectories, bit-identical: "
                 f"pred {ok['hP']}, lower {ok['hL']}, upper {ok['hU']}")
    return L


def group_block(R, trajs):
    """-> (markdown lines, summary dict) for one group of trajectories."""
    S = {}
    for v, (r, nm) in R.items():
        S[v] = {s: ood_score.aggregate(r, sel(r, nm, trajs, 0, km)) for s, km in SUBSETS}
        S[v]["traj"] = {t: {s: ood_score.aggregate(r, sel(r, nm, [t], 0, km)) for s, km in SUBSETS} for t in trajs}
    L = ["| variant | rel | tke | mvpe | sps | C | cov % | composite | first 3 | first 14 | Δ control (n+) | "
         "Δ sps vs control (n+) | Δ base (n+) | Δ control first 3 / 14 (n+ 14) | Δ base first 3 / 14 |",
         "|---|" + "---|" * 14]
    summ = {}
    for v in S:
        f = S[v]["full"]
        row = [v, f"{f['rel']:.3f}", f"{f['tke']:.3f}", f"{f['mvpe']:.3f}", f"{f['sps']:.3f}", f"{f['C']:.3f}",
               f"{f['cov']:.2f}", f"{f['composite']:.4f}", f"{S[v]['first3']['composite']:.4f}",
               f"{S[v]['first14']['composite']:.4f}"]
        dd = {}
        for ref in ("control", "base"):
            if ref not in S:
                continue
            d = {s: S[v][s]["composite"] - S[ref][s]["composite"] for s, _ in SUBSETS}
            pos = lambda s, key="composite": int(sum(S[v]["traj"][t][s][key] > S[ref]["traj"][t][s][key]  # noqa: E731
                                                  for t in trajs))
            dd[ref] = dict(d, npos=pos("full"), npos3=pos("first3"), npos14=pos("first14"), n=len(trajs),
                           mean_traj=float(np.mean([S[v]["traj"][t]["full"]["composite"]
                                                    - S[ref]["traj"][t]["full"]["composite"] for t in trajs])),
                           **{f"d_{k}": S[v]["full"][k] - S[ref]["full"][k] for k in ("rel", "tke", "mvpe", "sps", "C")},
                           npos_sps=pos("full", "sps"))
        n = len(trajs)
        c, b = dd.get("control"), dd.get("base")
        row.append(f"{c['full']:+.4f} ({c['npos']}/{n})" if c else "n/a")
        row.append(f"{c['d_sps']:+.4f} ({c['npos_sps']}/{n})" if c else "n/a")
        row.append(f"{b['full']:+.4f} ({b['npos']}/{n})" if b else "n/a")
        row.append(f"{c['first3']:+.4f} / {c['first14']:+.4f} ({c['npos14']}/{n})" if c else "n/a")
        row.append(f"{b['first3']:+.4f} / {b['first14']:+.4f}" if b else "n/a")
        L.append("| " + " | ".join(row) + " |")
        summ[v] = {s: S[v][s] for s, _ in SUBSETS}
        summ[v]["traj"] = {t: {s: S[v]["traj"][t][s]["composite"] for s, _ in SUBSETS} for t in trajs}
        summ[v]["delta"] = dd
    if "control" in summ and "base" in summ.get("control", {}).get("delta", {}):
        x = summ["control"]["delta"]["base"]
        L += ["", f"`control` − `base` by subscore: rel {x['d_rel']:+.3f} tke {x['d_tke']:+.3f} mvpe {x['d_mvpe']:+.3f} "
                  f"sps {x['d_sps']:+.3f} ΔC×100 {x['d_C']:+.3f}"]
    # per trajectory, main variants only (the c_* set is summarised by its counts above)
    vs = [v for v in S if v != "record" and not v.startswith("c_")]
    dv = [v for v in vs if v not in ("control", "base", "frozen")]
    L += ["", "| trajectory | " + " | ".join(vs) + (" | control − base" if "base" in S and "control" in S else "") +
          "".join(f" | {v} − control" for v in dv if "control" in S) + " |",
          "|---|" + "---|" * (len(vs) + ("base" in S and "control" in S) + len(dv) * ("control" in S))]
    for t in trajs:
        row = [t] + [f"{S[v]['traj'][t]['full']['composite']:.3f}" for v in vs]
        if "base" in S and "control" in S:
            row.append(f"{S['control']['traj'][t]['full']['composite'] - S['base']['traj'][t]['full']['composite']:+.4f}")
        if "control" in S:
            row += [f"{S[v]['traj'][t]['full']['composite'] - S['control']['traj'][t]['full']['composite']:+.4f}" for v in dv]
        L.append("| " + " | ".join(row) + " |")
    return L, summ


def nbins_block(R, trajs):
    if "base" not in R or "control" not in R:
        return [], {}
    L = ["| n revealed | windows | base | control | control − base | ΔC×100 | Δ sps |", "|---|---|---|---|---|---|---|"]
    out = {"bins": {}}
    for tag, lo, hi in NBINS:
        a = {}
        for v in ("base", "control"):
            r, nm = R[v]
            m = sel(r, nm, trajs, lo, hi + 1)
            a[v] = ood_score.aggregate(r, m) if m.any() else None
        if a["base"] is None:
            continue
        g = a["control"]["composite"] - a["base"]["composite"]
        out["bins"][tag] = dict(n=a["base"]["n"], base=a["base"]["composite"], control=a["control"]["composite"], gain=g,
                                dC=a["control"]["C"] - a["base"]["C"], dsps=a["control"]["sps"] - a["base"]["sps"])
        L.append(f"| {tag} | {a['base']['n']} | {a['base']['composite']:.4f} | {a['control']['composite']:.4f} | {g:+.4f} | "
                 f"{out['bins'][tag]['dC']:+.3f} | {out['bins'][tag]['dsps']:+.3f} |")
    B = out["bins"]
    if "8-13" in B:
        g8 = B["8-13"]["gain"]
        r, nm = R["base"]
        for lab, kmax in (("full streams", None), ("first 14 windows", 14)):
            N = int(sel(r, nm, trajs, 0, kmax).sum())
            bound = sum(B[b]["n"] / N * (g8 - B[b]["gain"]) for b in ("0", "1", "2") if b in B)
            out[f"bound_{'full' if kmax is None else 'first14'}"] = bound
            L.append(f"\nI4 hindsight bound, {lab}: windows with n ≤ 2 given the 8-13 gain ({g8:+.4f}) → "
                     f"**{bound:+.4f}** composite (first order; {N} windows)")
    return L, out


def main():
    run = sys.argv[1]
    SETS = load(run)
    if not SETS:
        raise SystemExit(f"no json under {run}/json")
    any_j = next(iter(next(iter(SETS.values())).values()))
    groups = any_j["groups"]
    summary = dict(run=run, regime=any_j["regime"], sets={})
    L = [f"# OOD stream instrument: `{os.path.basename(run.rstrip('/'))}` (regime {any_j['regime']})", "",
         "Composite = `final_v9.W0` . (rel, tke, mvpe, sps), no time term; harness feeds the revealed target normalised "
         f"with **{any_j['prev_space_harness'].upper()}** statistics. Real code path (extracted package).", ""]
    for m, V in SETS.items():
        R = {v: rec(j) for v, j in V.items()}
        names0 = next(iter(R.values()))[1]
        for v, (_, nm) in R.items():
            if not np.array_equal(nm, names0):
                raise SystemExit(f"{v}{'__' + m if m else ''}: a different window list")
        title = "plain streams" if not m else f"re-blanked with {m}"
        L += [f"# Set: {title}", ""]
        if m:
            st = next(iter(V.values())).get("reblank_stats") or {}
            if st:
                mp = [s["mask_px"] for s in st.values()]; nw = [s["mask_new_px"] for s in st.values()]
                L += [f"Mask {m}: {min(mp)}-{max(mp)} px per trajectory, of which {min(nw)}-{max(nw)} newly blanked "
                      f"(not in the recording's static zero set), of 2048.", ""]
        L += ["## Variants", ""] + variant_table(V) + [""]
        idl = identity(V)
        if idl:
            L += ["## Identity checks", ""] + idl + [""]
        summary["sets"][m or "plain"] = {}
        for g, trajs in groups.items():
            gl, gs = group_block(R, trajs)
            L += [f"## {g} ({len(trajs)} trajectories){'' if not m else ', ' + m}", ""] + gl + [""]
            if not m:
                nl, ns = nbins_block(R, trajs)
                if nl:
                    L += [f"### {g}: by revealed windows n", ""] + nl + [""]
                    gs["_nbins"] = ns
            summary["sets"][m or "plain"][g] = gs
    open(os.path.join(run, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump(summary, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
