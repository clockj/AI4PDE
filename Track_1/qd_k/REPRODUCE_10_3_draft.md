# Draft for REPRODUCE.md §10.3 (Bram, 09-24) — replaces the two "RPDE_QDK_* = <workspace file>" comment lines and the loop

**Scope (Devi, 09-24).** This section reproduces *k from the six twins*: make_val → fit_band → export → k* → H ratio.
It does NOT reproduce the twins themselves. Their training is §10.2's recipe (about 26 GPU-h), which is only checked when
the whole pipeline is retrained for the finals. The 09-24 check (job qdk_repro) ran this section on the 09-23 twins
(Devi's qd_r_s1..6, byte-identical); "k reproduced end to end" means this section only.

Everything below runs from a clean checkout plus the release data and checkpoints. No file comes from anyone's workspace.
Run from the repo root. `$CKD` is the directory holding the r34 members and twins (`ffno_all_w40l8_sim_s1..4`, the two
all-data ViTs, `p2_ffno_l8_sim_s1..3`, `arch_vit_d384`, `arch_vit_cmb3sim`). `qd_twin_s1..6.pth` come from §10.2.

```bash
# 0. validation windows (writes Track_1/val_{seen,unseen}.npz)
(cd Track_1 && $PY make_val.py --data "$REAL")

# 1. the control band, fitted in-repo with the r38 recipe (same members, ALPHA 0.80, centre decoupled)
U=Track_1/unet
$PY Track_1/submit_fno/fit_band.py \
    --ckpt $U/ffno_all_w40l8_sim_s{1,2,3,4}.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth \
    --err-ckpt $U/p2_ffno_l8_sim_s{1,2,3}.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth \
    --alpha 0.80 --band-p 0.45 --fill-unscored --centre-alpha 0 --out qdk_ctl_band.npz

# 2. the twin export the derivation reads
$PY Track_1/qd_k/export.py --ckpt-root $CKD --data "$REAL" --out qdk_export --tmp /tmp/qdk_exp_tmp

# 3. k on the six QD twins, nested leave-one-trajectory-out over the unseen AoA >= 15 trajectories
export RPDE_QDK_EXPORT=qdk_export
for s in 1 2 3 4 5 6; do
  $PY Track_1/qd_k/head_direct.py qd_twin_s$s.pth Track_1/unet --real-only --family direct --direct-kmax 20 \
      --pred-k 1.25 --band qdk_ctl_band.npz --csv qd_direct_band.csv
done
```

**Check:** every run prints `k* per fold [1.14, 1.14, 1.14]` (grid point 1.1399; neighbours 1.0446 / 1.2438; grid step
(2.5/0.4)^(1/21) = 1.0912). `k*` depends on the twins and the export only. The control band sets the reported gain,
not the choice: `--base recipe` (control refit from the export, no band file) gives the same k* (checked 09-24).

**Shipped k = K_USE from the like-for-like H check** (§10.1 members vs these twins, both in-sample on val_seen):
```bash
(cd Track_1 && $PY qd_k/h_ratio_check.py --members unet/ffno_all_w40l8_sim_qderr_s{1,2,3,4}.pth \
     --twins ../qd_twin_s{1,2,3,4,5,6}.pth --val val_seen.npz --k-base 1.1399)
```
Rule: 0.95 ≤ r ≤ 1.05 → k = 1.1399. r < 0.95 → 1.1399 × 1.0912 = 1.2438. r > 1.05 → 1.1399, noted in the README.
|r − 1| > 0.10 → exit 2, review. The 09-24 build gave r = 0.9884 → K_USE = 1.1399. Pass K_USE as `RPDE_ERR_K`.
