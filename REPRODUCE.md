# Reproducing the packages of record, from the released data

**Why this file exists.** Both Codabench pages say of the Decision phase (2026-09-28 → 10-25): *"The top 10
teams … are shortlisted. Organizers re-train each shortlisted team's method **from scratch on the
competition's GPU cluster** to ensure reproducibility, then evaluate on a private test set with unseen
parameter regimes; the resulting private-test scores decide the top 3."* A third party must therefore get from
the released data to the uploaded archive with nothing of ours but this repository. §6 lists every place that
is not yet true.

**ONE method of record on both tracks, as of 2026-09-20** — the same six members, the same band, the
same baked constants; the two archives differ only in the entry point (`predict` vs `ttt_step`) and in
`HOST_WRITE`, which Track 2 does not use:

| | package | archive |
|---|---|---|
| Track 1 | **R30 `r34`** — F-FNO w40 L8 **+ sim** ×4 (dense) + ViT ×2, fp16, band `r34_p2ffnoL8sim_fill.npz` | `../sub_candidates/track1_r34_p2ffnoL8sim_fill.zip` |
| Track 2 | **T2R17 `r34`** — the identical member set and band | `../sub_candidates/track2_r34_p2ffnoL8sim_fill.zip` |

Built and gated 2026-09-20 (job `rp_pkg_r34`, 7412354); staged, not yet scored. It is R28 / T2R13 with
**one** change, the sim co-training of the four F-FNO members and their three twins: the AoA-holdout
screen prices that at **+0.281 at a held-out AoA 10 and +1.12 at a held-out AoA 20** (two seeds), which
is the private set's regime, against **+0.02** on the dev board (`Track_1/plan/plan_20260917.md` §11.3,
§11.4). Superseded, and kept here because they are what the board actually scored:

| | superseded package | archive |
|---|---|---|
| Track 1 | **R25 `xfam6fp16`** — FNO w96 L8 `cmb3sim` ×4 + ViT ×2, band `r24_xfam6.npz` (81.344644) | `../sub_candidates/submitted/track1_r25_xfam6fp16_a060.zip` |
| Track 1 | **R28 `P2-L8`** — F-FNO w40 L8 ×4, **no sim**, band `t2r13_p2ffnoL8x4.npz` (81.401196, the Track-1 best) | `../sub_candidates/submitted/track1_r31_p2ffnoL8_a060.zip` |
| Track 2 | **T2R13 `P2-L8`** — the same no-sim members and band (81.314785) | `../sub_candidates/submitted/track2_r27_p2ffno4_a060.zip` |
| Track 2 | **T2R16** — T2R13 with the filled band `t2r16_p2ffnoL8_fill.npz` (81.323990, the Track-2 best by score) | `../sub_candidates/submitted/track2_r33_p2ffnoL8_fill.zip` |

`reproduce.sh` next to this file is a sequential driver for the whole thing; **it has never been run end to
end** — it is this audit written as a script, not a tested pipeline. Throughout:
`PY=…/.conda/envs/realpde/bin/python`, `REAL=$RPDE/train_real/train_real`, `SIM=$RPDE/train_sim/train_sim`,
`U=Track_1/unet`; training commands run from `Track_1/unet/`.

---

## 1. Stage A — the checkpoints

Every recipe below was read back from the shipped file with `torch.load(p, map_location="cpu")["config"]`, not
from any write-up. Two rules hold for all of them: **the all-data members pass `--all-data` and no
`--exclude`**, and **their held-out twins pass neither, so `--exclude` defaults to `model.VAL_TRAJ`** — 8
trajectories (`3750_0 6300_10 8850_20 11400_20 13950_10 16500_5 21600_15 26700_0`), 12 040 windows against 13
368 all-data. `re_split.excluded()` (26 trajectories) is the Re-extrapolation instrument's list and **no
shipped checkpoint uses it**.

### 1.1 R25 — the four FNO members (`fno_all_w96L8cmb3sim_s1..s4`)

**Superseded** — this is R25's primary family. The method of record's primary family is §1.4. Kept
because R25 scored 81.344644 and because the two ViTs of §1.2 are shared with it.

```bash
for s in 1 2 3 4; do
  $PY train2.py --data $REAL --arch fno --width 96 --n-layers 8 --modes-y 8 --modes-x 12 \
    --dropout 0.45 --var-head --var-mlp --lift-mlp \
    --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
    --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 \
    --stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5 --seed $s --all-data \
    --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
    --out fno_all_w96L8cmb3sim_s$s.pth
done
```

`--sim-exclude-held` is **absent** (`sim_exclude_held: false`): an all-data run has no held-out set, so there
is nothing to withhold. Jobs of record: `rp_r19e` (s1, s2), `rp_r19f` (s3, s4).

### 1.2 Track 1 — the two ViT members

```bash
# arch_vit_all_s1 -- plain MSE field loss, NO sim
$PY train2.py --data $REAL --arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 \
  --dropout 0.3 --var-head --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 \
  --alpha-train 0.0 --stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5 --seed 1 \
  --all-data --out arch_vit_all_s1.pth

# arch_vit_all_cmb3sim_s3 -- the cmb3sim losses and sim
$PY train2.py --data $REAL --arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 \
  --dropout 0.3 --var-head --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 \
  --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 \
  --stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5 --seed 3 --all-data \
  --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
  --out arch_vit_all_cmb3sim_s3.pth
```

`ViT2d` takes neither `--var-mlp` nor `--lift-mlp`; `--modes-y/-x` are recorded but unused by this family.
Jobs: `rp_r20d`, `rp_r24b`.

### 1.3 The two ViT twins (current), and R25's three FNO twins (superseded)

Identical recipes with `--all-data` dropped (so `--exclude` falls back to `model.VAL_TRAJ`) and, wherever
`--sim` is present, `--sim-exclude-held` added. `arch_vit_d384` and `arch_vit_cmb3sim` are in the current
band's error field; the three `fno_r18_cmb3sim*` are R25's and are superseded by §1.5.

```bash
# fno_r18_cmb3sim{,_s2,_s3}: EXACTLY the §1.1 command with `--all-data` removed and
# `--sim-exclude-held` added, seeds 1..3; s=1 writes fno_r18_cmb3sim.pth (no _s1 suffix).
# arch_vit_d384      : §1.2's first command minus `--all-data`.            (no sim, seed 1)
# arch_vit_cmb3sim   : §1.2's second minus `--all-data`, plus `--sim-exclude-held`.  (seed 1)
```

**`arch_vit_d384.pth` was written by an older `train2.py`**: its `config` has no `lam_cov`, `cov_c`, `mixup`,
`sam_rho`, `subpix`, `aug_mask` or `aux_*` keys. Today's default for each is the off value, so the *recipe* is
recoverable; the *code* that made the weights is pinned nowhere (§6.7).

### 1.4 Both tracks — the four F-FNO members (`ffno_all_w40l8_sim_s1..s4`)

```bash
for s in 1 2 3 4; do
  $PY train2.py --data $REAL --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 \
    --dropout 0.3 --var-head \
    --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
    --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 \
    --stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5 --seed $s --all-data \
    --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
    --out ffno_all_w40l8_sim_s$s.pth
done
```

No `--var-mlp`, no `--lift-mlp`, no `--drop-path`; `--sim-exclude-held` is **absent**, as for every
all-data run (§1.1). Jobs `pz_ffnosimAll1..4` on Pitzer, 2026-09-20; `config["git"]` records the
`train2.py` commit (`7ea0a86-dirty` for s1/s2, `d8926cb-dirty` for s3/s4 — §6.7).

**Superseded, R28 / T2R13:** the same command **without the `--sim …` line**, written as
`ffno_all_w40l8_s$s.pth` (`sim: ""`), jobs `rp_ffnoAll1..4`. Those four are still in `Track_1/unet/`
and are what the Track-1 best (81.401196) and the Track-2 method of record were built from; the sim
line is the single lever between them.

### 1.5 Both tracks — the three F-FNO held-out twins

Same recipe, `--all-data` dropped (so `--exclude` = `model.VAL_TRAJ`) and `--sim-exclude-held` added
wherever `--sim` is present, seeds 1–3, written as `p2_ffno_l8_sim_s{1,2,3}.pth`:

```bash
for s in 1 2 3; do
  $PY train2.py --data $REAL --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 \
    --dropout 0.3 --var-head \
    --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
    --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 \
    --stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5 --seed $s \
    --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
    --sim-exclude-held --out p2_ffno_l8_sim_s$s.pth
done
```

The band also uses Track 1's two ViT twins, so `M` for the build is

```
p2_ffno_l8_sim_s1 p2_ffno_l8_sim_s2 p2_ffno_l8_sim_s3 arch_vit_d384 arch_vit_cmb3sim
```

Jobs `pz_p2simS1..S3`. **Superseded, R28 / T2R13:** the same three without the `--sim …` line and
without `--sim-exclude-held`, as `p2_ffno_l8_s{1,2,3}.pth`. All six are now in `Track_1/unet/` (§6.2).

---

## 2. Stage B — the sim mapping (`Track_1/unet/simreal.py`)

`load_sim_mapped(sim_dir, real_dir, exclude=, real_exclude=, mode="scalar")` maps every sim trajectory into
real's measurement space in three steps, and **all three read real-data statistics**: (1) *speed match* —
multiply so the trajectory's mean `u` equals real's at that (Re, AoA), using the **measured** mean of the real
twin where one exists and the freestream law `mean_u = c(AoA)·Re` where it does not; (2) *fluctuation* —
rescale the fluctuation so `‖std field‖` matches the twin's; (3) *support* — paint the twin's FOV / airfoil
zero-set onto sim.

**Does it depend on the split? Yes, and it is handled.** `train2.py` passes `real_exclude=excl`
**unconditionally** (line 803), so a held-out trajectory's mean, amplitude and mask never reach the twin
table, and its condition falls back to the law + the pooled ratio + the nearest mask. This is the fix for
round 14's leak of 26 held-out trajectories. `--sim-exclude-held` additionally passes the same list as
simreal's `exclude=`, which drops those **sim** trajectories from the pool entirely. The all-data members pass
neither (`excl` is empty); the twins pass both.

The map is deterministic given `(sim_dir, real_dir, phases, frames, mode)` and cached to one `.npy`; `exclude`
is applied **after** the cache is read, so one cache serves every split. Building it reads ~10 GB of HDF5.
Both caches the shipped members need are present on scratch: `work/simmap_p1_f0.npy` (all-data) and
`work/simmap_p1_f0_x476193e300.npy` (the `model.VAL_TRAJ` twins; the `_x` tag is `sha1("
".join(sorted(real_exclude)))[:10]`). They sit on purgeable scratch, and rebuilding either costs ~10 min of
CPU.

---

## 3. Stage C — the band

```bash
$PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.60 --band-p 0.45 \
    --fill-unscored --out Track_2/submit/band8.npz
```

defaults `--fit-on seen --fit-aoa 15 20 --inflate 1.39 --band-smooth 3`. `$A` is the shipping (all-data) set,
`$M` the twins: `sigma_ref` comes from `$A` (it normalizes the σ the shipped models emit) and the error field
from `$M` (a band fitted on memorized errors scored `sps` 36.94, R6). Inputs are the derived caches
`Track_1/val_{seen,unseen}.npz`.

`--fill-unscored` is part of the method of record (`--fill-nok 8 --fill-smooth 5`): where the fitting split
never scores an element, `sps9.optimal_band`'s weight vector is all zeros, `argmax` returns index 0 and `base`
collapses to `min_i |E|` ≈ 0, so the containment indicator there can never fire on a recording blanked
differently from our three. The fill blends those elements toward a k=5 box mean of well-sampled neighbours
and only ever widens. Its **measured** dev-board price is −0.0066 `sps` = −0.0016 `final` (T2R16 against
T2R13); it is in the method as insurance for the private set, whose blanking we have not seen
(`Track_1/plan/lit_20260917.md` §4.9).

**A refit is not bit-reproducible** — it re-runs the models over the fitting split, and
`Track_1/bands/MANIFEST.md` is explicit that the band is the one part of a package that cannot be rebuilt from
the checkpoints. A reproduction must **reuse the committed band** (`bands/r34_p2ffnoL8sim_fill.npz` for
R30 / T2R17; `bands/r24_xfam6.npz` for R25; `bands/t2r13_p2ffnoL8x4.npz` for R28 / T2R13), or it is
reproducing a different package.

---

## 4. Stage D — the build

**Both tracks, R30 / T2R17** (job `rp_pkg_r34.sbatch`, 7412354) — one invocation of the script builds and
gates both archives from one member set, with `U=Track_1/unet`:

```bash
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth \
   $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth \
   $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
RPDE_A="$A" RPDE_M="$M" RPDE_BAND=Track_1/bands/r34_p2ffnoL8sim_fill.npz \
  RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
  RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_TAG="r34_p2ffnoL8sim_fill" \
  bash Track_1/submit_fno/build_round13.sh 0.60
```

`RPDE_BAND` reuses the committed band, which is what a reproduction of a scored package must do (§3).
Dropping `RPDE_BAND` and passing **`RPDE_BAND_FILL=1`** instead is how the band itself was fitted, and is
the right call only when the member set changes.

**Superseded — Track 1, R25** (job `r25b.sbatch`):

```bash
A="$U/fno_all_w96L8cmb3sim_s1.pth … _s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/fno_r18_cmb3sim.pth $U/fno_r18_cmb3sim_s2.pth $U/fno_r18_cmb3sim_s3.pth \
   $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" RPDE_BAND=Track_1/bands/r24_xfam6.npz \
  RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
  RPDE_HOST_WRITE=torch RPDE_TAG="r25_xfam6fp16_a060" \
  bash Track_1/submit_fno/build_round13.sh 0.60
```

**Superseded — R28 / T2R13** (jobs `rp_pkg_p2ffno.sbatch`, `rp_pkg_r31.sbatch`), the no-sim members with
`bands/t2r13_p2ffnoL8x4.npz`, which T2R13's own build fitted with `RPDE_BAND` unset:

```bash
A="$U/ffno_all_w40l8_s1.pth … _s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_s1.pth $U/p2_ffno_l8_s2.pth $U/p2_ffno_l8_s3.pth \
   $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" RPDE_INT8=1 RPDE_LINEAR=tensor \
  RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense \
  RPDE_TAG="r27_p2ffno4_a060" bash Track_1/submit_fno/build_round13.sh 0.60
```

Unset knobs fall back to older behaviour (`ALPHA` 0.90, fp32 packing, empty `SPEC_PATH`/`HOST_WRITE`, `AMP =
None`, `GRAPH = False`), so **every knob must be passed every time**. The script `sed`s `ALPHA`, `FUSE`,
`AMP`, `GRAPH`, `GRAPH_FP`, `SPEC_PATH` into both `submission.py` files and `HOST_WRITE`/`BLEND_BETA` into
Track 1's, **and leaves them there** (§6.6). It runs `validate_strict.py` and `validate_t2.py`.

---

## 5. GPU-hours on one A100-PCIe-40G

Wall times are measured, from the training logs' last-epoch stamp (`logs/r19/`, `logs/r20/`, `logs/r22/`,
`logs/r24/`, `logs/r18/`, `screen/all/`, `screen/p2/`) and `sacct`.

The method of record now needs **one** member set for both tracks, so there is one total, not two.

| stage | runs | h/run | h |
|---|---|---|---|
| F-FNO w40 L8 **+ sim**, all-data | 4 | 3.75 | **15.0** |
| F-FNO w40 L8 **+ sim**, twin | 3 | 3.44 | **10.3** |
| ViT 384 all-data, no sim (7 198 s) | 1 | 2.00 | **2.00** |
| ViT 384 all-data, `cmb3sim` + sim (13 703 s) | 1 | 3.81 | **3.81** |
| ViT 384 twin, no sim (*estimated* — log lost) | 1 | ~1.80 | **~1.80** |
| ViT 384 twin, `cmb3sim` + sim (12 362 s) | 1 | 3.43 | **3.43** |
| sim-map cache, one-off (CPU-bound, ~10 GB of HDF5) | 1 | ~0.2 | **~0.2** |
| band fit ×2 + both builds + both strict gates (job 7412354, 190 s) | 1 | 0.05 | **0.05** |

**Both tracks, 36.6 GPU-h** ≈ 8 h wall clock at the 4-job cap (longest single run 3.8 h). Add ~1 h if
`val_{seen,unseen}.npz` must be rebuilt; `train2.py` holds the training set in host memory, so the node
needs the full 12-core allocation.

**Where the two F-FNO + sim figures come from.** Those eight runs went to Pitzer (V100-32G), not to an
A100, so their wall times are measured there and converted: all-data **4:52 ± 0:01** (`pz_ffnosimAll1..4`,
`sacct -M pitzer`), twin **4:24** (`pz_p2simS1..S3`). The conversion is the **1.868× sim overhead measured
on the same cluster on a paired holdout run** (`pz_aoa20_ffno` 1:55:58 without sim → `pz_aoa20_ffnosim`
3:36:45 with it), applied to the A100 no-sim times below. It is an overhead ratio, not a V100→A100 ratio,
which is why it is applied to the A100 numbers rather than to the V100 ones.

Superseded stages, kept because they are what the scored packages cost:

| stage | runs | h/run | h |
|---|---|---|---|
| FNO w96 L8 `cmb3sim`, all-data (9 747 s) — R25 | 4 | 2.71 | **10.84** |
| FNO w96 L8 `cmb3sim`, twin (8 714 / 8 829 s) — R25 | 3 | 2.44 | **7.32** |
| F-FNO w40 L8, **no sim**, all-data (7 219 s) — R28 / T2R13 | 4 | 2.01 | **8.03** |
| F-FNO w40 L8, **no sim**, twin (6 606 s) — R28 / T2R13 | 3 | 1.84 | **5.51** |

---

## 6. What is NOT scripted, or lives outside the repository

| # | gap | what it would take |
|---|---|---|
| 6.1 | **Every shipped checkpoint exists only as an untracked `.pth` under `Track_1/unet/`** (`.gitignore` line 2) — 15 for the superseded packages, plus the 7 sim files of the current one. They were produced by one-off sbatch jobs on scratch, none of which is in the repo. | `reproduce.sh` is the end-to-end driver. It needs one real run to become evidence. |
| 6.2 | ~~**T2R13's three band twins `p2_ffno_l8_s{1,2,3}.pth` are only on `/fs/scratch/.../screen/p2/`** — not in the repo, not in any uploaded archive, and scratch is purgeable.~~ | **CLOSED.** All six F-FNO twins are now in `Track_1/unet/` alongside the members: the no-sim `p2_ffno_l8_s{1,2,3}.pth` and, since 2026-09-20, the sim twins `p2_ffno_l8_sim_s{1,2,3}.pth` the current band was fitted from (copied, not moved; the scratch originals stay under `members/ffno_sim/` and `screen/p2sim/`). They are still untracked (6.1), so the scratch copies remain the only redundancy. |
| 6.3 | **The band's bit-reproducibility, corrected 2026-09-18.** It *is* bit-reproducible for the band of record: refitting `t2r13_p2ffnoL8x4.npz` from its own inputs (the L8 twins + the two ViT twins, ALPHA 0.60, `--band-p 0.45`, `--fit-on seen --fit-aoa 15 20`) **reproduces it bitwise, md5 `b99d177d889126b0734bee0680bc69b0`** — verified while building the `--fill-unscored` variant. What is *not* reproducible is a refit with a **different twin set**, which moves whole populations of elements (R9→R10 was +1.410 `sps`). | Unchanged in practice: scored bands stay committed in `Track_1/bands/` and reproduction reuses them via `RPDE_BAND`; never refit a package of record with a different twin set. The bitwise result means an organizer re-running our recipe gets our band, which is what the reproducibility gate actually asks. |
| 6.4 | ~~**`FFNO_PATH` (dense / fft) is baked by `$SCRATCH/jobs/patch_ffno_path.py`**, which is not in the repo and is not called by `build_round13.sh`. T2R13 cannot be rebuilt without it.~~ | **CLOSED 09-18 (forge).** The helper is now `Track_1/submit_fno/bake_ffno_path.py` and `build_round13.sh` calls it behind **`RPDE_FFNO_PATH`** (unset = both `submission.py` untouched, so every non-F-FNO build is byte-for-byte what it was). Verified against the uploaded T2R13 archive: the block the repo now writes is **byte-identical** to the one it shipped. The scratch copy is deleted and the four `rp_pkg_*` job scripts point at the repo path. |
| 6.5 | ~~**T2R13's `ALPHA` was read from `$SCRATCH/pkg_ffno/P2_ALPHA` and `_fno.py` was pinned by `$SCRATCH/pkg_ffno/P2_FNO_MD5`.** Both are scratch files; neither value is recorded in the repo next to the package.~~ | **CLOSED 09-18 (forge).** `ALPHA = 0.60` and `_fno.py` md5 `73c83f572c477b8aaa17566f168f72b7` are recorded in `Track_1/bands/MANIFEST.md` under the `t2r13` band, together with the same two pins for the staged Track-1 twin `track1_r31_p2ffnoL8_a060.zip`. Both were re-read **out of the uploaded archive**, not copied from the scratch files. |
| 6.6 | **`build_round13.sh` `sed`s constants into `submission.py` and leaves them.** The committed `submission.py` reflects whichever build ran last, not any package of record; `git diff` on it is the only guard, and `rp_pkg_p2ffno.sbatch` checks it explicitly. | Print the full baked-constant set into the archive and assert it after the build; or have the script restore the tree itself, as the jobs currently do by hand. |
| 6.7 | ~~**No checkpoint records the `train2.py` commit that produced it.** `arch_vit_d384`'s config proves the schema has drifted.~~ | **CLOSED 09-18 (forge)** for everything trained from now on: `train2.py` reads the short hash ONCE at startup (`GIT_REV`, `-dirty` when the tree has uncommitted edits, `unknown` when git is unavailable) and writes `config["git"]` and `config["train2_argv"]` -- the exact command line, which pins every flag whose DEFAULT may later move. It is also printed as a `git:` line at startup, so a running log carries it. **The 15 pre-09-18 checkpoints keep no hash**, so a re-train of those still has to pin one commit by hand; the 7 sim checkpoints of the current package do carry one (`d7492c7`/`bea5f04` for the twins, `7ea0a86`/`d8926cb` for the members, each `-dirty`). |
| 6.8 | **The archives, the staging README and the handoffs live outside the git tree** (`../sub_candidates/`, `../handoffs/`); the sim-map caches and `val_*.npz` sit on purgeable scratch, and nothing asserts a rebuilt cache matches the one a checkpoint saw. | State the first in the fact sheet. The key covers `(phases, frames, mode, real_exclude)`; add `stride` and a data md5 if it matters. `/fs/scratch/.../realpde/README.md` says which two caches are live. |

---

## 7. Nondeterminism, and what `--seed` actually controls

`train2.py` sets `torch.manual_seed`, `torch.cuda.manual_seed_all` and `np.random.seed` from `--seed` (default
`-1` = unseeded). It does **not** set `torch.use_deterministic_algorithms` or `cudnn.deterministic`, and does
not fix a cuDNN benchmark mode.

A seed therefore makes an arm and its control share **initialization and batch order** — what it exists for
(seed σ = 0.017 `wfinal`, the size of several shipped effects) — but promises nothing across a different GPU,
CUDA/cuDNN version or `OMP_NUM_THREADS`. A re-train on the organizers' A800s will give *different weights* at
the same seed, so the acceptance test must be a **score** test, not a checksum: rebuild, run both strict
gates, and require the four error subscores inside the seed spread (±0.017 `wfinal` per member, ~±0.01 at
four). Everything downstream of training *is* deterministic: the band solve is a sort, `_fno.fuse()` is exact
to ~1e-8 with TF32 off, int8 packing is deterministic, and `AMP` self-checks at import.

Also worth telling the organizers: **`time` is measured on their cluster**, so `board_rounds.py`'s latency
constants do not transfer, and `AMP_ACTIVE` is the only fingerprint of which precision path ran.
