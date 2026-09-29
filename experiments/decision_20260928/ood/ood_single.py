"""Single-member screens from the trainings' own re_split.py json files (T4v, T4b); no GPU.

    python ood_single.py t4v      # each augmented ViT against its seed-matched control ViT
    python ood_single.py t4b      # --aug-resim 0.5 against 0.25, seed-matched F-FNO members

Criteria (PLAN.md, rows of 11:03): T4v each of the three >= +0.20 wfinal on the mean of relo and rehi, rein >= -0.05;
T4b beats 0.25 by >= +0.05 on the mean of relo and rehi with 4 of 4 seeds, rein not worse by more than 0.02.
"""
from __future__ import annotations

import json
import os
import sys

D = "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928"


def w(path):
    if not os.path.exists(path):
        return None
    s = json.load(open(path))["sets"]
    return {k: s[k]["wfinal"] for k in ("relo", "rehi", "rein")}


def main():
    arm = sys.argv[1]
    if arm == "t4v":
        pairs = [(f"{D}/t4vc/ckpt/RE/re_vitc_resim_s3.json", f"{D}/ckpt/RE/re_vitc_s3.json", "ViT-C s3"),
                 (f"{D}/t4vc/ckpt/RE/re_vitc_resim_s1.json", f"{D}/ckpt/RE/re_vitc_s1.json", "ViT-C s1"),
                 (f"{D}/t4vp/ckpt/RE/re_vitp_resim_s1.json", f"{D}/ckpt/RE/re_vitp_s1.json", "ViT-P s1")]
        bar, rein_bar, need = 0.20, -0.05, 3
    else:
        pairs = [(f"{D}/t4b/ckpt/RE/re_ffnosim_resim50_s{k}.json", f"{D}/t4/ckpt/RE/re_ffnosim_resim_s{k}.json", f"s{k}")
                 for k in range(1, 5)]
        bar, rein_bar, need = 0.05, -0.02, 4
    print("| member | relo Δ | rehi Δ | mean(relo, rehi) Δ | rein Δ | passes |\n|---|---|---|---|---|---|")
    ok_all, n = True, 0
    for a, b, lab in pairs:
        A, B = w(a), w(b)
        if A is None or B is None:
            print(f"| {lab} | missing ({'arm' if A is None else 'control'} json) | | | | |"); ok_all = False; continue
        d = {k: A[k] - B[k] for k in A}; m = (d["relo"] + d["rehi"]) / 2
        ok = m >= bar and d["rein"] >= rein_bar
        n += ok; ok_all &= ok
        print(f"| {lab} | {d['relo']:+.3f} | {d['rehi']:+.3f} | {m:+.3f} | {d['rein']:+.3f} | {'yes' if ok else 'no'} |")
    print(f"\n{arm}: {'CLEARS' if ok_all and n >= need else 'does not clear'} the single-member screen ({n} of {len(pairs)})")


if __name__ == "__main__":
    main()
