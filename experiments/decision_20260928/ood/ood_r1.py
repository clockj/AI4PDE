"""Chunk 6: gate R1 (reproducibility), R1b (the band), and the second candidate's twin read. No GPU.

    python ood_r1.py [RUNS_DIR]

All reads: twins on the 8 VAL_TRAJ trajectories, real package code, stream (`control`) and frozen.
  (a) id_repro_ship   the from-scratch run's five twins, band refitted the shipping way
  (b) id_t4all_ship   the campaign's five augmented twins, the same construction
  (c) id_twins_ship   the five twins of record, the same construction
  second candidate    id_cand2_ship: three from-scratch F-FNO twins + the two ViT twins of record, the same construction;
                      against (c) and against the campaign's id_t4_ship
R1b: (a) scored with four bands: its own shipping-way refit, the from-scratch run's band (cand/repro), the band of the
campaign's twins (id_t4all_ship), the band of record (k27).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
DEC = os.path.dirname(os.path.dirname(RUNS))
KEYS = ("rel", "tke", "mvpe", "sps", "C", "composite")


def S(run):
    p = os.path.join(RUNS, run, "summary.json")
    return json.load(open(p))["sets"]["plain"] if os.path.exists(p) else None


def cmp(a, b, v, g):
    A, B = S(a), S(b)
    if not A or not B:
        return None
    x, y = A[g][v], B[g][v]
    per = {t: y["traj"][t]["full"] - x["traj"][t]["full"] for t in x["traj"]}
    return dict(d={k: y["full"][k] - x["full"][k] for k in KEYS}, per=per, lev=(x["full"], y["full"]))


def table(title, a, b, L):
    L += [f"### {title}", "", "| variant | slice | Δ rel | Δ tke | Δ mvpe | Δ sps | ΔC | **Δ composite** | per trajectory (composite) |",
          "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for v in ("control", "frozen"):
        for g in ("all8", "a15_20"):
            c = cmp(a, b, v, g)
            if c is None:
                L.append(f"| {v} | {g} | pending |||||||"); continue
            d = c["d"]; out[(v, g)] = c
            per = ", ".join(f"{t} {x:+.4f}" for t, x in sorted(c["per"].items(), key=lambda kv: kv[0]))
            L.append(f"| {v} | {g} | {d['rel']:+.4f} | {d['tke']:+.4f} | {d['mvpe']:+.4f} | {d['sps']:+.4f} | {d['C']:+.4f} | "
                     f"**{d['composite']:+.4f}** | {per if g == 'all8' else ''} |")
    L.append("")
    return out


def band_stats(npz, js):
    z = np.load(npz); j = json.load(open(js))
    b = z["base"][..., :2]
    return dict(med_u=float(np.median(b[..., 0])), med_v=float(np.median(b[..., 1])), q=j["q"],
                logk_u=float(np.mean(np.asarray(j["logkappa"])[:, 0])), logk_v=float(np.mean(np.asarray(j["logkappa"])[:, 1])),
                p=float(z["p"]), sref_med=float(np.median(z["sigma_ref"][..., :2])))


def main():
    L = ["## Gate R1: from-scratch twins (a) against the campaign's twins (b); acceptance |Δ composite| ≤ 0.005", ""]
    r1 = table("(a) − (b): from-scratch run − campaign, five augmented twins each", "id_t4all_ship", "id_repro_ship", L)
    table("(a) − (c): from-scratch run − the five twins of record (the second, independent read of the candidate's in-distribution gain)",
          "id_twins_ship", "id_repro_ship", L)
    if ("control", "all8") in r1:
        d = r1[("control", "all8")]["d"]["composite"]
        L += [f"**R1: {'PASS' if abs(d) <= 0.005 else 'FAIL'}** — package composite (stream, all 8) of the from-scratch run "
              f"{r1[('control', 'all8')]['lev'][1]['composite']:.4f} against the campaign's {r1[('control', 'all8')]['lev'][0]['composite']:.4f}: "
              f"Δ {d:+.4f} (bar 0.005)", ""]
    # R1b
    bands = [("(a)'s own shipping-way refit", "id_repro_ship",
              os.path.join(RUNS, "id_repro_ship/band/k0/band_k0_sigblend_lam0.5.npz"), os.path.join(RUNS, "id_repro_ship/band/k0/band_k0_sigblend_lam0.5.json")),
             ("from-scratch run's band (reproduce.sh dp_band: members + twins)", "id_repro_bandrepro",
              os.path.join(DEC, "cand/repro/log/decision/band/k0/band_k0_sigblend_lam0.5.npz"), os.path.join(DEC, "cand/repro/log/decision/band/k0/band_k0_sigblend_lam0.5.json")),
             ("campaign twins' band (id_t4all_ship)", "id_repro_bandt4all",
              os.path.join(RUNS, "id_t4all_ship/band/k0/band_k0_sigblend_lam0.5.npz"), os.path.join(RUNS, "id_t4all_ship/band/k0/band_k0_sigblend_lam0.5.json")),
             ("band of record (k27)", "id_repro_bandrec",
              os.path.join(DEC, "ood/wt_ood/Track_1/bands/k27_sigblend_lam0.5.npz"), os.path.join(DEC, "ood/wt_ood/Track_1/sigblend/band_k0_sigblend_lam0.5.json"))]
    L += ["## R1b: the band", "",
          "| band | median base′ u / v | σ_ref median | q (u, v) | mean log κ u / v | (a) stream composite all 8 / AoA 15-20 | sps | C | mean width | (a) frozen composite all 8 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, run, npz, js in bands:
        st = band_stats(npz, js) if os.path.exists(npz) else None
        s = S(run)
        if s is None or st is None:
            L.append(f"| {lab} | pending |||||||||"); continue
        c, h, f = s["all8"]["control"]["full"], s["a15_20"]["control"]["full"], s["all8"]["frozen"]["full"]
        L.append(f"| {lab} | {st['med_u']:.5f} / {st['med_v']:.5f} | {st['sref_med']:.5f} | {st['q'][0]:.4f}, {st['q'][1]:.4f} | "
                 f"{st['logk_u']:.3f} / {st['logk_v']:.3f} | {c['composite']:.4f} / {h['composite']:.4f} | {c['sps']:.3f} | {c['C']:.3f} | "
                 f"{c.get('width', float('nan')):.5f} | {f['composite']:.4f} |")
    L.append("")
    # second candidate
    L += ["## The second candidate's twins (three from-scratch F-FNO twins + the two ViT twins of record)", ""]
    table("second candidate − the five twins of record", "id_twins_ship", "id_cand2_ship", L)
    table("second candidate − the campaign's T4 F-FNO-only twins (id_t4_ship; its reproducibility read)", "id_t4_ship", "id_cand2_ship", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
