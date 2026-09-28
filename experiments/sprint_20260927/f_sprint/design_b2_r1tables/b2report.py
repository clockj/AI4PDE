"""Design B2 report: paired deltas of every band variant vs shipped asym R1 on the same prediction-side stream.
    python b2report.py <tag> [<tag> ...] [--set wave1] [--only name ...]
d = delta of the no-time composite = 0.2498 * delta sps (the prediction is identical across band variants).
cols: full | AoA 15-20 (in-distribution dumps) | first 14 windows of every stream | n-bins of the window index."""
import sys, json, argparse
import numpy as np
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927"
ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--set", default=None); ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--base", default="R1ship")
a_ = ap.parse_args()
for tag in a_.tags:
    R = json.load(open(f"{S}/ckpt/b2/res_{tag}{'' if a_.set is None else '_' + a_.set}.json"))
    names = np.array(R["names"]); a = np.array(R["a"]); nsc = np.array(R["nsc"]); idx = np.array(R["idx"])
    trajs = list(dict.fromkeys(names)); aoa = np.array([int(t.split("_")[1]) for t in names])
    def sps(con, m):
        return 100.0 * float((a[m] * con[m]).sum()) / float(nsc[m].sum())
    b = np.array(R["var"][a_.base]["con"])
    allm = np.ones(len(names), bool); hi = aoa >= 15; t14 = idx < 14
    bins = [("n=0", idx == 0), ("1-3", (idx >= 1) & (idx <= 3)), ("4-13", (idx >= 4) & (idx <= 13)), ("14+", idx >= 14)]
    print(f"== {tag}: {len(trajs)} trajectories, {len(names)} windows; base {a_.base} sps full {sps(b, allm):.3f} (probe.run ref {R['ref_sps']:.3f}) "
          f"a15-20 {sps(b, hi):.3f} first14 {sps(b, t14):.3f}; composite(no time) ref {R['ref_comp']:.3f}")
    for nm, v in R["var"].items():
        if a_.only and nm not in a_.only:
            continue
        c = np.array(v["con"])
        per = [0.2498 * (sps(c, names == t) - sps(b, names == t)) for t in trajs]
        per14 = [0.2498 * (sps(c, (names == t) & t14) - sps(b, (names == t) & t14)) for t in trajs]
        ph = [p for p, t in zip(per, trajs) if int(t.split("_")[1]) >= 15]
        w = np.array(v["wid"]); wb = np.array(R["var"][a_.base]["wid"])
        print(f"{nm:18s} full d {0.2498 * (sps(c, allm) - sps(b, allm)):+.4f} ({sum(p > 0 for p in per)}/{len(per)}) min {min(per):+.3f} | "
              f"a15-20 d {0.2498 * (sps(c, hi) - sps(b, hi)):+.4f} ({sum(p > 0 for p in ph)}/{len(ph)}) | "
              f"first14 d {0.2498 * (sps(c, t14) - sps(b, t14)):+.4f} ({sum(p > 0 for p in per14)}/{len(per14)}) | "
              + " ".join(f"{l} {0.2498 * (sps(c, m) - sps(b, m)):+.3f}" for l, m in bins) + f" | width x{w.sum() / wb.sum():.3f}")
