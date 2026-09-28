"""Finals-method k decision from Fen's qdre per-window tables (rule fixed 12:19Z 09-24, #Leaders:0cb36eee a9daedcd,
Ilse's version):

  1  CROSS-SELECT: pick k on Re 25425 over the full QD grid (kgrid_d), evaluate k vs kq=1.1399 on Re 26700; then
     the reverse. Both cross-evaluated widenings must be > 0 (mean of per-trajectory seed-averaged deltas).
  2  rehi gain := mean of the two cross-evaluated gains. The finals k (Emre's split of testing from setting) is then
     picked on the whole rehi group; rein and relo cost at THAT k (vs kq) must each be > -gain/2.
  3  otherwise both the board and the finals use kq.
  Also reported, never judged: weighted composites 6:1:1 (Corin) and 5:2:1 (Jonas) rehi:rein:relo.

Selection scores average the seeds (the sum of a*v over windows divided by n_scored, then the mean over seeds). All
deltas are in final units (W * d_sps). Per-trajectory values are seed means.
usage: python3 finals_k.py qdre_tables_1.npz qdre_tables_2.npz      |  python3 finals_k.py --selftest
"""
import sys
import numpy as np

HI_A, HI_B = "25425", "26700"


def load(paths):
    Z = [np.load(p, allow_pickle=True) for p in paths]
    kd, kq, W = Z[0]["kgrid_d"], float(Z[0]["kq"]), float(Z[0]["W"])
    for z in Z[1:]:
        assert np.array_equal(z["kgrid_d"], kd) and float(z["kq"]) == kq, "seeds disagree on the grid"
    G = {}
    for g in ("rein", "rehi", "relo"):
        tr = [str(t) for t in Z[0][f"{g}_traj"]]
        for z in Z[1:]:
            assert [str(t) for t in z[f"{g}_traj"]] == tr, f"{g}: window order differs between seeds"
        G[g] = dict(traj=np.array(tr), a=[z[f"{g}_a"] for z in Z], nsc=[z[f"{g}_nsc"] for z in Z],
                    tq=[z[f"{g}_tq"] for z in Z])
    return G, kd, kq, W


def S(G, g, ki, sel):
    """seed-mean pooled sps of QD at grid index ki over windows sel."""
    T = G[g]
    return float(np.mean([100.0 * (a[sel] @ tq[ki][sel]) / max(int(n[sel].sum()), 1)
                          for a, n, tq in zip(T["a"], T["nsc"], T["tq"])]))


def per_traj_delta(G, g, ki, kqi, W, only=None):
    T = G[g]
    names = sorted(set(T["traj"]) if only is None else [t for t in set(T["traj"]) if t.split("_")[0] == only])
    return {t: W * (S(G, g, ki, T["traj"] == t) - S(G, g, kqi, T["traj"] == t)) for t in names}


def decide(G, kd, kq, W, out=print):
    kqi = int(np.argmin(np.abs(np.log(kd / kq))))
    assert abs(kd[kqi] - kq) < 1e-4
    T = G["rehi"]
    re_of = np.array([t.split("_")[0] for t in T["traj"]])
    assert set(re_of) == {HI_A, HI_B}, f"rehi Re values {sorted(set(re_of))}"
    cross = {}
    for pick, ev in ((HI_A, HI_B), (HI_B, HI_A)):
        ki = max(range(len(kd)), key=lambda i: S(G, "rehi", i, re_of == pick))
        d = per_traj_delta(G, "rehi", ki, kqi, W, only=ev)
        m = float(np.mean(list(d.values())))
        cross[pick] = (float(kd[ki]), ki - kqi, m, d)
        out(f"(1) pick on {pick}: k* {kd[ki]:.4f} ({ki - kqi:+d} steps)  -> eval on {ev}: widening "
            f"{m:+.5f}  per-traj " + " ".join(f"{t}:{v:+.4f}" for t, v in d.items()))
    c1 = all(v[2] > 0 for v in cross.values())
    gain = float(np.mean([v[2] for v in cross.values()]))
    out(f"(1) both cross-evaluated widenings > 0: {c1}   rehi gain (mean of the two) {gain:+.5f}")
    kfi = max(range(len(kd)), key=lambda i: S(G, "rehi", i, np.ones(len(re_of), bool)))
    kf = float(kd[kfi])
    costs = {}
    for g in ("rein", "relo"):
        d = per_traj_delta(G, g, kfi, kqi, W)
        costs[g] = float(np.mean(list(d.values())))
    c2 = c1 and all(c > -gain / 2 for c in costs.values())
    out(f"(2) finals k candidate (whole rehi) {kf:.4f} ({kfi - kqi:+d} steps); cost at it vs {kq:.4f}: "
        + "  ".join(f"{g} {c:+.5f}" for g, c in costs.items()) + f"  | bar -gain/2 = {-gain / 2:+.5f}  -> {c2}")
    if kfi == kqi:
        out("    NOTE: whole-rehi k* equals kq; nothing to change")
    dhi = float(np.mean(list(per_traj_delta(G, "rehi", kfi, kqi, W).values())))
    for lab, w in (("6:1:1 (Corin)", (6, 1, 1)), ("5:2:1 (Jonas)", (5, 2, 1))):
        comp = (w[0] * dhi + w[1] * costs["rein"] + w[2] * costs["relo"]) / sum(w)
        out(f"[report only] composite {lab}: {comp:+.5f}  (rehi term is IN-SAMPLE at the whole-rehi k*)")
    kfinal = kf if (c2 and kfi != kqi) else kq
    out(f"==> FINALS k = {kfinal:.4f}   (board k stays {kq:.4f})")
    return kfinal, c1, c2


def _selftest():
    rng = np.random.default_rng(0)
    kd = np.exp(np.arange(np.log(0.4), np.log(3.0), np.log(1.0911))).astype(np.float32)
    kq = float(kd[np.argmin(np.abs(np.log(kd / 1.1399)))])
    groups = {"rein": ["12675_0", "12675_5", "19050_0", "19050_5"], "rehi": ["25425_0", "25425_5", "26700_0", "26700_5"],
              "relo": ["3750_0", "5025_0"]}

    def make(best, curv):        # sps(k) peaked at `best` per group, curvature per group; 3 windows per trajectory
        z = dict(kgrid=kd, kgrid_d=kd, kq=np.float32(kq), W=np.float64(0.25))
        for g, tr in groups.items():
            t = np.repeat(tr, 3)
            z[f"{g}_traj"], z[f"{g}_a"], z[f"{g}_nsc"] = t, np.ones(len(t)), np.full(len(t), 100)
            z[f"{g}_tq"] = np.stack([50 - curv[g] * (np.log(k / best[g])) ** 2 + rng.normal(0, .01, len(t)) for k in kd])
        return z
    import io
    def run(best, curv=None):
        curv = curv or {"rein": 30, "rehi": 30, "relo": 30}
        bufs = []
        for _ in range(2):
            b = io.BytesIO(); np.savez(b, **make(best, curv)); b.seek(0); bufs.append(b)
        G, KD, KQ, W = load(bufs)
        return decide(G, KD, KQ, W, out=lambda *_: None)
    k, c1, c2 = run({"rein": kq, "rehi": kq * 1.0911 ** 1, "relo": kq}, {"rein": 5, "rehi": 30, "relo": 5})
    assert c1 and c2 and k > kq and abs(np.log(k / kq) / np.log(1.0911) - 1) < .1, (k, c1, c2)
    k, c1, c2 = run({"rein": kq, "rehi": kq, "relo": kq})
    assert k == kq and not c1, (k, c1)
    k, c1, c2 = run({"rein": kq * 1.0911 ** -3, "rehi": kq * 1.0911, "relo": kq * 1.0911 ** -3})
    assert c1 and not c2 and k == kq, (k, c1, c2)          # widening helps rehi but costs rein/relo too much
    k, c1, c2 = run({"rein": kq, "rehi": kq * 1.0911 ** 1, "relo": kq})
    assert c1 and not c2 and k == kq, (k, c1, c2)          # symmetric curvature: cost == gain > gain/2 -> refuse
    print("selftest 4/4 ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        G, kd, kq, W = load(sys.argv[1:])
        print(f"seeds {len(sys.argv) - 1}; grid {len(kd)} pts; kq {kq:.4f}; W {W}")
        decide(G, kd, kq, W)
