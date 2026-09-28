# Result K3 (executor E3): warm-start polish arms — held-out reads, 2026-09-27 17:30 EDT

**Verdict: NO-SHIP for every arm. Nothing was built.** No arm reaches the pre-registered gate (+0.04 vs baseline AND +0.02 vs
ctl at AoA 15–20) or the lower +0.02 "gamble" build threshold. The best arm is `hib_fv` at +0.0125 d_fin_nt
(AoA 15–20, static, vs baseline), which is noise-sized.

Instrument: `rp_k273_eval_arm.sh` → `rp_k273_candref.sh`, i.e. a twin package with the band refitted per arm (shipped construction:
ALPHA 0.80, p 0.45, centre0, fill, inflate 1.39) and val_unseen stream reads. `d_fin_nt` = Δ(0.4778 rel + 0.1010 tke + 0.0921 mvpe
+ 0.2498 sps) in score units, with no time term. Full tables: `$K/notes/result_3_table.txt` (vs baseline), `result_3_table_vs_ctl_f.txt`,
`result_3_table_vs_ctl_fv.txt`. Logs: `$K/logs/rp_k273_cr_<tag>_<jobid>.log`.

## Jobs
* 39 of 40 training jobs finished with DONE. **`rp_k273_hio_m2` (7566766, node a0105) FAILED** at epoch ~135/150 with
  `CUDA error: an illegal memory access` in `loss.backward()`. This looks like a node or transient fault: the other jobs on a0105
  finished. Per orders it was not resubmitted (17:12, and the 17:20 cutoff was close). The failed member was an all-data member,
  so no evidence was lost. For the `hio` reads the band fit used the UNPOLISHED `ffno_all_w40l8_sim_s2` in its place (tags
  `hio_f_sub2`, `hio_fv_sub2`, via a new `SUBST_M` option in `rp_k273_eval_arm.sh`). All twins are polished, so the twin evidence is
  intact.
* Harness edits, all in `$K/jobs`: `rp_k273_table.py` gained `REF=<tag>` pairing and a coverage column;
  `rp_k273_eval_arm.sh` gained `SUBST_M`. The originals are in `/tmp/*.orig` on the login node. No other debugging was needed:
  polished checkpoints load (79/79 tensors, config identical to the parent except epochs/lr/init).
* Harness sanity: `ctl` reproduces baseline to ≤0.005 on every row, and the `hio`/`hio2` full-split reads move strongly. So the
  polished weights ARE being read; the null at AoA 15–20 is real and not a loading bug.

## Primary read: twin package, STATIC band, val_unseen (d_fin_nt; sign counts use the per-trajectory static d_fin_nt)
| tag | AoA15-20 vs base | AoA15-20 vs ctl | full vs base | full vs ctl | cov 15-20 / all | traj d_fin>0 (15-20, all) | traj rel>0 (15-20, all) | ens_v9 rel 15-20 (base 0.1009) |
|---|---|---|---|---|---|---|---|---|
| ctl_f  | −0.0025 | — | +0.0013 | — | 0.876 / 0.920 | 1/3, 5/8 | 0/3, 3/8 | 0.1009 |
| hib_f  | +0.0051 | +0.0076 | +0.0045 | +0.0032 | 0.875 / 0.919 | 2/3, 5/8 | 2/3, 4/8 | 0.1009 |
| hio_f_sub2 | −0.0007 | +0.0018 | **−0.2238** | −0.2251 | 0.875 / 0.916 | 2/3, 2/8 | 2/3, 2/8 | 0.1010 |
| hio2_f | −0.0103 | −0.0078 | **−0.3668** | −0.3681 | 0.873 / 0.914 | 0/3, 0/8 | 0/3, 0/8 | 0.1011 |
| ctl_fv | −0.0031 | — | −0.0004 | — | 0.876 / 0.920 | 0/3, 3/8 | 0/3, 0/8 | 0.1010 |
| **hib_fv** | **+0.0125** | **+0.0156** | +0.0122 | +0.0126 | 0.874 / 0.918 | 3/3, 7/8 | 2/3, 3/8 | 0.1009 |
| hio_fv_sub2 | +0.0058 | +0.0089 | **−0.4245** | −0.4241 | 0.872 / 0.909 | 3/3, 3/8 | 1/3, 1/8 | 0.1010 |
(`do0` was cancelled and not read. `hio2` has no ViTs, so it has no `fv` read.)

Subscore detail for the best arm, `hib_fv`, AoA 15–20 static vs baseline: rel +0.001, tke −0.017, mvpe −0.003, sps +0.056
(cov 0.874 vs 0.877). Nearly all of its gain is sps/containment noise; rel and tke do not move.

T2 rows (asym R1), AoA 15–20 vs baseline: hib_fv +0.0042 (+D1c +0.0034), hib_f +0.0011, ctl_f −0.0045, ctl_fv −0.0069,
hio_f −0.0044, hio_fv −0.0032, hio2_f −0.0165. Full split: every arm is ≤ +0.00, and hio/hio2 range from −0.17 to −0.34.

ens_v9 (twins, f 1.39) AoA 15–20 final Δ: ctl_f −0.003, hib_f +0.001, hio_f −0.008, hio2_f −0.019, ctl_fv −0.008,
hib_fv +0.000, hio_fv −0.012. Full: hio_f −0.329, hio2_f −0.520, hio_fv −0.625; the others are within ±0.006.

## Inflate (band trap)
Pre-registered trigger: AoA 15–20 static coverage more than 0.010 below the baseline's 0.877. Every arm is 0.872–0.876, so the
**trigger did not fire for any arm**. The coverage-matched inflate is therefore the default 1.39, the "matched-inflate" read
equals the default read above, and no i150/i160 re-read was run.

## Decision (pre-registered rule, goalposts unchanged)
* Rule 1 needs AoA15-20 ≥ +0.04 vs baseline and ≥ +0.02 vs ctl. **No arm passes.** The closest, `hib_fv`, reaches +0.0125 and
  +0.0156; it also fails the ens_v9 rel criterion (0.1009, equal to baseline, not lower).
* The build threshold for a "below-gate gamble" is ≥ +0.02 at AoA 15–20. **No arm reaches it, so NO build was made** (task D was
  not triggered).
* The specialist hypothesis is falsified on the held-out set. AoA-15/20-only polishing (`hio`, `hio2`) buys nothing at AoA 15–20
  (−0.001 to −0.010) and costs −0.22 to −0.42 on the full split, from forgetting low AoA (3750_0 −0.35…−0.59, 26700_0
  −0.64…−1.28). The higher-LR dose (`hio2`) is worse still. The mild over-sampling arm `hib` is harmless and slightly positive
  (7/8 trajectories positive in `fv`), but the effect is far below any shippable size.
* **K3 contributes nothing to today's package.** Best arm for the record: `hib_fv`, with a projected board Δ of 0.5 × 0.0125 ≈
  +0.006 [0, +0.013].
