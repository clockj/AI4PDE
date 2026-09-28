"""Is the blanked region a property of the rig, the angle or the recording?  (CPU; chunk 4, part 1)

    python ood_blank.py OUTDIR [--dump A20=RUN/dump/frozen A10=RUN/dump/frozen]

Per released real recording at 32 x 64 ([::2, ::2]) and per channel: the STATIC zero set (u, resp. v, exactly 0 in every
frame) and the MOSTLY zero set (0 in more than half of the frames).
  (a) sizes by AoA x Re; (b) union and intersection over the 82 recordings;
  (c) per regime (A20, A10, relo, rehi, rein): pixels static-zero in a held-out recording and static-zero in NO training
      recording of that regime (training = all 82 minus VAL_TRAJ minus the regime), per held-out recording, with a map;
      "strict" = not even MOSTLY zero in any training recording;
  (d) with --dump (ood_stream.py --dump of `frozen`): mean |P| on those novel pixels against mean |P| on the static-zero
      pixels the held-out recording shares with the training set, and the flow scale (mean |Y| on non-zero targets).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_lib  # noqa: E402

DATA = ood_lib.DATA


def zero_sets():
    import h5py
    out = {}
    for n in ood_lib.all_real():
        with h5py.File(os.path.join(DATA, n + ".h5"), "r") as f:
            st, mo = [], []
            for c in ("u", "v"):
                z = f[c][:, ::2, ::2] == 0
                st.append(z.all(0)); mo.append(z.mean(0) > 0.5)
        out[n] = dict(static=np.stack(st, -1), mostly=np.stack(mo, -1))      # (32, 64, 2)
    return out


def table(Z, key, ch):
    res = sorted({ood_lib.re_of(n) for n in Z}); aoas = sorted({ood_lib.aoa_of(n) for n in Z})
    L = ["| AoA \\ Re | " + " | ".join(str(r) for r in res) + " |", "|---|" + "---|" * len(res)]
    for a in aoas:
        row = []
        for r in res:
            n = f"{r}_{a}"
            row.append(str(int(Z[n][key][..., ch].sum())) if n in Z else "-")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    return L


def tmap(novel, train_union):
    """rows 0..31 top to bottom, columns 0..63; '#' novel (either channel), '+' static-zero in some training recording, '.'"""
    L = []
    for r in range(32):
        L.append("".join("#" if novel[r, c].any() else ("+" if train_union[r, c].any() else ".") for c in range(64)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dump", nargs="*", default=[], help="REGIME=DUMPDIR (frozen dumps of the stand-ins)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Z = zero_sets()
    names = sorted(Z)
    vt = set(ood_lib.val_traj())
    res = {"n_recordings": len(names)}
    L = ["# Blanked regions of the 82 released recordings (32 x 64)", ""]
    for key in ("static", "mostly"):
        for ch, cn in ((0, "u"), (1, "v")):
            L += [f"## (a) {key} zero set, channel {cn}: pixel count by AoA x Re", ""] + table(Z, key, ch) + [""]
        allz = np.stack([Z[n][key] for n in names])
        U, I = allz.any(0), allz.all(0)
        res[f"{key}_union"] = U.sum((0, 1)).tolist(); res[f"{key}_intersection"] = I.sum((0, 1)).tolist()
        L += [f"(b) {key}: union over 82 recordings u {int(U[..., 0].sum())} / v {int(U[..., 1].sum())} px; intersection "
              f"u {int(I[..., 0].sum())} / v {int(I[..., 1].sum())} px (of 2048). Per recording: median u "
              f"{int(np.median(allz[..., 0].sum((1, 2))))}, range {int(allz[..., 0].sum((1, 2)).min())}-"
              f"{int(allz[..., 0].sum((1, 2)).max())}.", ""]
    # how much of a recording's static set is explained by its AoA (union / intersection within each AoA)
    L += ["(b') static set (u or v) within each AoA: | AoA | recordings | intersection | union | mean per recording |",
          "|---|---|---|---|---|"]
    for aoa in sorted({ood_lib.aoa_of(n) for n in names}):
        g = np.stack([Z[n]["static"].any(-1) for n in names if ood_lib.aoa_of(n) == aoa])
        L.append(f"| {aoa} | {len(g)} | {int(g.all(0).sum())} | {int(g.any(0).sum())} | {g.sum((1, 2)).mean():.1f} |")
    L.append("")
    dumps = dict(s.split("=", 1) for s in a.dump)
    res["regimes"] = {}
    for reg, grp in (("A20", "A20"), ("A10", "A10"), ("RE", "relo"), ("RE", "rehi"), ("RE", "rein")):
        ev, groups, held = ood_lib.regime(reg)
        hs = groups[grp]
        train = [n for n in names if n not in set(held) and n not in vt]
        TU = np.stack([Z[n]["static"] for n in train]).any(0)              # (32, 64, 2)
        TM = np.stack([Z[n]["mostly"] for n in train]).any(0)
        per, novel_all = {}, np.zeros((32, 64, 2), bool)
        for h in hs:
            nov = Z[h]["static"] & ~TU
            strict = Z[h]["static"] & ~TM
            novel_all |= nov
            per[h] = dict(static=Z[h]["static"].sum((0, 1)).tolist(), novel=nov.sum((0, 1)).tolist(),
                          novel_strict=strict.sum((0, 1)).tolist())
        rr = dict(train=len(train), held=len(hs), per=per, novel_union=novel_all.sum((0, 1)).tolist())
        L += [f"## (c) {grp}: {len(hs)} held-out recordings, {len(train)} training recordings", "",
              "| held-out | static px u / v | novel u / v (static in no training recording) | strict u / v (not even mostly zero in training) |",
              "|---|---|---|---|"]
        for h in hs:
            p = per[h]
            L.append(f"| {h} | {p['static'][0]} / {p['static'][1]} | {p['novel'][0]} / {p['novel'][1]} | "
                     f"{p['novel_strict'][0]} / {p['novel_strict'][1]} |")
        L += ["", f"Map ('#' novel in some held-out recording, '+' static-zero in some training recording; row 0 = top):",
              "```"] + tmap(novel_all, TU) + ["```", ""]
        if grp in dumps:
            D = dumps[grp]
            P = np.load(os.path.join(D, "P.npy"), mmap_mode="r")
            Y = np.load(os.path.join(D, "Y.npy"), mmap_mode="r")
            traj = np.load(os.path.join(D, "traj.npy"))
            acc = {k: [0.0, 0] for k in ("novel", "strict", "shared", "flow")}
            rows = []
            for h in hs:
                idx = np.where(traj == h)[0]
                if not len(idx):
                    continue
                Ph = np.abs(np.asarray(P[idx]))                          # (n, 20, 32, 64, 2)
                Yh = np.asarray(Y[idx])
                masks = {"novel": Z[h]["static"] & ~TU, "strict": Z[h]["static"] & ~TM, "shared": Z[h]["static"] & TU}
                r = {}
                for k, m in masks.items():
                    mm = np.broadcast_to(m, Ph.shape)
                    s, n = float(Ph[mm].sum()), int(mm.sum())
                    acc[k][0] += s; acc[k][1] += n
                    r[k] = (s / n if n else float("nan"), n)
                fl = Yh != 0
                acc["flow"][0] += float(np.abs(Yh[fl]).sum()); acc["flow"][1] += int(fl.sum())
                rows.append((h, r))
            rr["pred"] = {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in acc.items()}
            L += [f"(d) {grp}, stand-in `frozen`: mean |P| (m/s) on novel static-zero pixels "
                  f"{rr['pred']['novel'][0]:.4f} ({rr['pred']['novel'][1]} element-frames), strict-novel "
                  f"{(rr['pred']['strict'][0] or float('nan')):.4f} ({rr['pred']['strict'][1]}), on static-zero pixels shared "
                  f"with training {rr['pred']['shared'][0]:.4f} ({rr['pred']['shared'][1]}); flow scale mean |Y| on non-zero "
                  f"targets {rr['pred']['flow'][0]:.4f}.", "",
                  "| held-out | novel: mean abs P (element-frames) | shared: mean abs P |", "|---|---|---|"]
            for h, r in rows:
                L.append(f"| {h} | {r['novel'][0]:.4f} ({r['novel'][1]}) | {r['shared'][0]:.4f} ({r['shared'][1]}) |")
            L.append("")
        res["regimes"][grp] = rr
    open(os.path.join(a.out, "blank.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(a.out, "blank.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
