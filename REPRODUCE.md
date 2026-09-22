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

Built and gated 2026-09-20 (job `rp_pkg_r34`, 7412354) and **scored the same day: 81.418569 on Track 1 and
81.333776 on Track 2, the best on both boards** (`board_rounds.py` R30 / T2R17). It is R28 / T2R13 with
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

`reproduce.sh` next to this file is a sequential driver for the whole thing. **It has now been run**: end to
end in a 2-epoch smoke (every stage, both builds, both strict gates) and, for the F-FNO twin seed 1, at full
fidelity against the shipped checkpoint, and its build re-creates **both uploaded archives array for array**
— **§8** is the procedure, the five knobs that make it runnable without touching the shipped tree, and what
came out. Throughout:
`PY=…/.conda/envs/realpde/bin/python`, `REAL=$RPDE/train_real/train_real`, `SIM=$RPDE/train_sim/train_sim`,
`U=Track_1/unet` for the code and `OUT=${RPDE_REPRO_OUT:-$U}` for the checkpoints; training commands run
from `Track_1/unet/`.

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

**All four ViTs were written by an older `train2.py`** (verified key by key against today's parser, §8):
every one of them lacks `drop_path`, `max_epoch`, `oracle_cond` and `teacher*`, and `arch_vit_d384.pth` —
the oldest — also lacks `lam_cov`, `cov_c`, `mixup`, `sam_rho`, `subpix`, `aug_mask`, `mode_drop`,
`lam_tail`, `sim_map`, `sim_p_*`, `lam_p_sim` and `aux_*`. Today's default for each is the off value, so the
*recipe* is recoverable and the commands above are exact; what is not pinned is the *code* that made the
weights (§6.7), and a feature that is off today may still consume a random draw that the older code did not,
so a re-train is a score match, not a checksum match (§7).

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

Those seven jobs also passed **`--ckpt-every 20 --resume`** (it is in their `train2_argv`), which
`reproduce.sh` does not: it writes an epoch-boundary resume file so a preempted 4½-hour run restarts where
it stopped. Every one of the seven logs `resume: no checkpoint … starting from scratch`, so **none of them
resumed and the two flags changed nothing about the weights** — but add them to any re-train on a
`--requeue` partition.

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
    --fill-unscored --centre-alpha 0 --out Track_2/submit/band8.npz
```

**r35 (2026-09-21) adds `--centre-alpha 0`:** the SPS interval is centred on the pre-compose member mean
y(α=0) instead of on the prediction (`lower=min(c−h, pred)`, `upper=max(c+h, pred)`), and `base` is solved on
|y(0) − y|. The value is stored in the band (`centre_alpha`), and both `submission.py` files read it at load,
so the centre cannot disagree with the band; a band without the key reproduces r34 exactly. The point
prediction is unchanged (pkg_identity 0.000e+00 vs R30 / T2R17). Evidence: held-out Δsps +0.61, 8/8
trajectories, pre-registered (#battleground-0921:ac7fc4f3).

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
the checkpoints. A reproduction must **reuse the committed band** (`bands/r35_centre0_fill.npz` for r35;
`bands/r34_p2ffnoL8sim_fill.npz` for R30 / T2R17; `bands/r24_xfam6.npz` for R25; `bands/t2r13_p2ffnoL8x4.npz` for R28 / T2R13), or it is
reproducing a different package.

---

## 4. Stage D — the build

**Both tracks, r35 — the method of record** (`reproduce.sh` → `build_r35`; built by run
20260921-180408, gates 2/2, pkg_identity prediction 0.000e+00 vs R30 / T2R17): r34's members and flags plus
`RPDE_CENTRE_ALPHA=0` and the r35 band:

```bash
RPDE_A="$A" RPDE_M="$M" RPDE_CENTRE_ALPHA=0 RPDE_BAND=Track_1/bands/r35_centre0_fill.npz \
  RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
  RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_TAG="r35_centre0_fill" \
  bash Track_1/submit_fno/build_round13.sh 0.60
```

(`$A`, `$M` as below.) The r35 logic lives in the repository (`_fno.predict_fno(centre_out=…)`, the bounds
block in both `submission.py`, `fit_band.py --centre-alpha`); the baked constants in a shipped zip (`AMP =
"fp16"`, `FFNO_PATH = "dense"`, …) are NOT committed — `build_round13.sh` injects them at build time (§6.6),
so the repository's defaults differing from a shipped archive is expected.

**Superseded — both tracks, R30 / T2R17** (job `rp_pkg_r34.sbatch`, 7412354; `reproduce.sh r34`) — one
invocation of the script builds and gates both archives from one member set, with `U=Track_1/unet`:

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
| 6.1 | **Every shipped checkpoint exists only as an untracked `.pth` under `Track_1/unet/`** (`.gitignore` line 2) — 15 for the superseded packages, plus the 7 sim files of the current one. They were produced by one-off sbatch jobs on scratch, none of which is in the repo. | **HALF CLOSED 09-20 (verifier, §8):** `reproduce.sh` now runs — every stage, both builds, both gates, in a 2-epoch smoke; its build re-creates both uploaded archives array for array from the shipped `.pth` (§8.4); and one training stage re-run at 400 epochs reproduces its shipped checkpoint's score to four decimals (§8.5). What is still unrun is the other 10 full-fidelity stages, ~32 GPU-h. The `.pth` files themselves are still untracked and still exist in one place. |
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
to ~1e-8 with TF32 off, int8 packing is deterministic, and `AMP` self-checks at import — **measured 09-20**,
both archives rebuilt to arrays identical to the uploaded ones at 0.0e+00 (§8.4).

**Measured on the same GPU model, 09-20 (§8.5):** re-training one twin from the same code and seed on
another node of the same cluster is bit-identical for ~40 epochs and then drifts in the 5th decimal, and
after 400 epochs it scores **the same `wfinal` to four decimals** (81.9865). So the nondeterminism above is
real but far below the seed floor; the cross-*hardware* case is the one that still needs the score test.

Also worth telling the organizers: **`time` is measured on their cluster**, so `board_rounds.py`'s latency
constants do not transfer, and `AMP_ACTIVE` is the only fingerprint of which precision path ran.

---

## 8. Running `reproduce.sh` — the knobs, the smoke, and what it proved (2026-09-20)

### 8.1 The five knobs

All five are no-ops when unset, so the default invocation is byte-for-byte the pipeline of record.

| knob | what it does |
|---|---|
| `RPDE_REPRO_OUT=<dir>` | write **and read back** every checkpoint there instead of `Track_1/unet`. `U` stays the *code* dir (`train2.py` imports `model.py` as a sibling); `OUT` is the *output* dir, and `--out` is now the absolute path. Before this they were one variable, and the `P2` knob §6.2 documents was broken: with `P2` outside `U` the run wrote into `U` and then failed its own existence check. |
| `RPDE_REPRO_EPOCHS=<n>` | `--epochs` for every training stage. 400 is the recipe; anything else is a **smoke** and the script then refuses to write into `Track_1/unet`, or to run at all in a tree that has `../sub_candidates` beside it. |
| `RPDE_REPRO_STAGE="a b"` | run only these stage functions, in order, and no build — the header's "split across the job cap", which until now had no way to call one function. |
| `RPDE_REPRO_SEEDS="1 2"` | narrow every seed loop (default `1 2 3` twins, `1 2 3 4` members). |
| `RPDE_REPRO_FITBAND=1` | **fit** the band from the twins this run trained (`RPDE_BAND_FILL=1`, `--fill-unscored`) instead of reusing the committed one. Reuse is right for reproducing a *scored* package (§3); the fit is what a new member set needs, and it is what the smoke exercises. |

`reproduce.sh` also reads the two `SUBMITTABLE` lines out of the build's own log and **fails if either gate
did not pass** — `build_round13.sh` does not propagate its gates' exit codes (CLAUDE.md), and before this
the script printed `DONE` over a failed gate.

### 8.2 The smoke must run from a copy of the repo

`build_round13.sh` stages both archives into `../sub_candidates` and **its `SUB` is not overridable**; the
band it writes lands in `Track_2/submit/band8.npz` and it `sed`s both `submission.py` files (§6.6). So a
smoke runs in a detached worktree on scratch, where `../sub_candidates` does not exist and nothing it
touches is the working tree:

```bash
git worktree add --detach /fs/scratch/PAS1187/qwu/tmp/realpde/repro/repo HEAD
cd /fs/scratch/PAS1187/qwu/tmp/realpde/repro/repo
ln -s /users/PAS1187/qwu/workspace/realpde/main/Track_1/val_{seen,unseen}.npz Track_1/   # derived caches
RPDE_REPRODUCE_GO=1 RPDE_REPRO_EPOCHS=2 RPDE_REPRO_FITBAND=1 \
  RPDE_REPRO_OUT=/fs/scratch/PAS1187/qwu/tmp/realpde/repro/smoke_unet bash reproduce.sh both
```

on one GPU node (`pz_repro_smoke`, `-p gpu-exp -c 4`). The sim-map caches (§2) and `val_{seen,unseen}.npz`
are absolute scratch paths, so they are found from anywhere; nothing else in the pipeline is.

The smoke's two archives carry the package-of-record's `RPDE_TAG`, because the tag is the recipe's, not the
run's — they are renamed `SMOKE_2EPOCH_*` on scratch so nothing can confuse them with an upload.

### 8.3 The smoke: every stage runs (job `pz_repro_smoke` 54416675, V100-32G, 1 513 s = 0.42 GPU-h)

`reproduce.sh both` at `--epochs 2` exited **0** with **2/2 SUBMITTABLE**, having run, in order: the 2 ViT
twins, the 3 F-FNO+sim twins, the 2 ViT members, the 4 F-FNO+sim members (11 checkpoints, ~2 min each),
`fit_band.py --fill-unscored` (touched 3 436 of 81 920 elements, `untouched elements bit-identical: True`),
the ALPHA / FUSE / AMP / GRAPH / SPEC_PATH / HOST_WRITE / `FFNO_PATH` bake, `build_zip.py` +
`validate_strict.py` (36.40 MB extracted, all four contract checks, order-independence at 0.0e+00 drift),
and `build_t2.py` + `validate_t2.py` (336 streamed steps, no aliasing, CUDA/CPU paths bit-identical,
stateless replay bit-identical). Nothing reached `Track_1/unet`, `Track_1/bands` or `../sub_candidates`.

The gates' error subscores are **worthless** at 2 epochs (rel_l2 0.2100 against the shipped 0.0592) — they
are there to prove the code path, not the model. **Neither timing is board evidence**: `2.696 ms/window` on
Track 1 and `4.932 ms/step` on Track 2 are a V100S, where the board is A100-class and the private set is an
A800, and the 6 members were 2-epoch weights.

Three plumbing bugs the smoke and the static pass found, all fixed in `reproduce.sh` and none in a recipe:
the `--out` basename that broke `P2`/`RPDE_REPRO_OUT` (§8.1), the unchecked gate exit codes (§8.1), and the
missing way to run one stage. **The recipes themselves need no change**: all 11 command lines were parsed
with today's `train2.py` parser and compared key by key against the shipped checkpoints' `config` —
**0 mismatches** on every shared key, and the only absent keys are the ViT schema drift of §1.3.

### 8.4 The build, at full fidelity: the archives rebuild bit for bit (job `pz_repro_ident` 54416918, 47 s)

The smoke proves the build *runs*; this proves it rebuilds **the package of record**. The same
`reproduce.sh both`, with `RPDE_REPRO_OUT` pointed at the shipped `Track_1/unet` so all eleven training
stages `SKIP  … (exists)`, ran the band reuse, the constant bake and both gates in **47 s** from the
worktree of HEAD — **2/2 SUBMITTABLE** — and `verify/pkg_identity.py` then compared the two archives it
produced against `sub_candidates/submitted/track{1,2}_r34_p2ffnoL8sim_fill.zip`:

| | prediction | lower | upper | verdict |
|---|---|---|---|---|
| Track 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | **IDENTICAL** (tolerance 1e-5) |
| Track 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | **IDENTICAL**, CUDA graph active in both |

So everything downstream of the weights — band, ALPHA, `fuse`, int8 packing, `AMP`, `SPEC_PATH`,
`HOST_WRITE`, `FFNO_PATH`, both entry points — is reproduced exactly by the committed script, on a GPU that
is not the one the package was built on. The rebuilt Track-1 zip is even the same 30 282 465 bytes.

This also measures §6.6: the committed `submission.py` carries **`AMP = None` and no `FFNO_PATH`** — the
tree was restored after the r34 build — so the shipped constants exist *only* as `build_round13.sh`
arguments. `reproduce.sh` passes all of them, and the identity above is the proof that it passes them
right; editing the constants by hand instead is how a reproduction silently ships fp32.

### 8.5 Fidelity: one stage at 400 epochs against its shipped twin (job `pz_repro_fid` 54416670)

The build reproduces exactly because nothing in it is stochastic. Training is the part that can drift, so
the script's **own** `twins_ffno` stage was re-run at full fidelity —
`RPDE_REPRO_STAGE=twins_ffno RPDE_REPRO_SEEDS=1`, 400 epochs, into a scratch `RPDE_REPRO_OUT` — and scored
with the instrument the shipped twin was scored with, `screen_sets.py --arm repro_p2_ffno_l8_sim_s1`.
Same code (worktree of 9e8f868 against the shipped `d7492c7-dirty`), same seed, same cluster and same GPU
model (Pitzer V100S-32G), 15 750 s against the shipped 15 743 s.

| `p045_full`, K1 | shipped `p2_ffno_l8_sim_s1` | re-trained | Δ |
|---|---|---|---|
| rel | 0.075748 | 0.075748 | **+0.000000** |
| tke | 0.495154 | 0.495176 | +0.000022 |
| mvpe | 0.068372 | 0.068371 | −0.000001 |
| sps | 45.5413 | 45.5419 | +0.0006 |
| **wfinal** | **81.9865** | **81.9865** | **+0.0000** |

against a seed spread of **0.011** over the three sim twins (81.9865 / 81.9944 / 81.9973) — the
reproduction is three orders of magnitude inside it. The training trace is *bit-identical* for the first
~40 epochs (`loss 4.73662 → 2.70188 → 1.74383 …`, every component) and then drifts in the 5th decimal,
which is cuDNN's non-determinism (§7), not the recipe: the final epoch reads `loss 0.76131 field 0.06190
var 0.37984 tke 0.37793` on both runs and differs only in `div`, at 1.12507 against 1.12508.

**What this does and does not prove.** It proves the driver trains *the shipped recipe* — one member, on
the same hardware. It does not prove the other ten stages (~32 GPU-h, unrun), and it says nothing about
the organizers' A800s, where §7's score test is still the acceptance criterion.

### 8.6 What the audit cost, and what it left unverified

Three Pitzer jobs, **4.87 GPU-h**: `pz_repro_smoke` 54416675 (0:25:14), `pz_repro_ident` 54416918
(0:01:31), `pz_repro_fid` 54416670 (4:24:56). Scratch under `/fs/scratch/.../realpde/repro/`, sbatch
scripts in `/fs/scratch/.../realpde/jobs/pz_repro_*.sbatch`.

Still unverified, in the order it would matter if a shortlist arrives:

1. ~~**The other ten training stages at 400 epochs** (~32 GPU-h).~~ **CLOSED 2026-09-22 (Ilse), §9.**
   All eleven stages were re-trained at 400 epochs on a *different* stack and compared to the shipped
   checkpoints: recipes identical key by key, scores inside the seed spread, and the package rebuilt from
   the eleven retrained weights passes both gates with the same score as the shipped one.
2. **`time` on the evaluation GPU.** Every latency number in this file's §8 is a Pitzer V100S. The board
   is A100-class, the private set an A800, and `validate_strict`'s ms/window is not a board prediction
   even on the right card — the gate exists for the contracts, not the clock.
3. **Anything trained before 09-18 cannot be pinned to a commit** (§6.7), so an exact re-creation of the
   four ViTs is a score match at best.
4. **`../sub_candidates` is still not redirectable** without editing `build_round13.sh`; the worktree of
   §8.2 is a workaround, not a fix.
5. **The derived caches** `Track_1/val_{seen,unseen}.npz` and the two sim maps were reused, not rebuilt.
   **Partly closed 2026-09-22 (§9):** every job of that campaign rebuilt `val_{seen,unseen}.npz` from the
   released HDF5 and got the same sha256 each time (`a518c91d…` / `1a7e85af…`), and the sim map rebuilt
   under the shipped twin's own cache key `_x476193e300`. What is still unasserted is that a rebuilt sim
   map equals the one a shipped checkpoint actually saw — that file lives on the author's scratch.


---

## 9. Reproduced end to end on a foreign machine (2026-09-22, Ilse)

Eleven Pitzer V100S jobs, ~40 GPU-h, on a stack the repo does **not** pin: torch 2.7.1 / numpy 2.2.6
(`environment.yml` pins torch 2.2.2 / numpy<2). Every stage of §1 was re-trained at 400 epochs from the
released data and compared with the shipped checkpoint in the same process, then the whole package was
rebuilt from the retrained weights alone.

**Result: the method reproduces.**

| what | result |
|---|---|
| recipes (`config`, key by key, retrain vs shipped) | identical for all 11; only `epochs`/`out`/`git` and the inert `ckpt_every`/`resume` differ |
| single-model score, 7 F-FNO stages | ≤ 0.0002 `wfinal` |
| single-model score, 4 ViT stages | −0.051 … +0.007 (ViT seed spread is 0.095) |
| all 11 signed deltas | mean −0.0068, sd 0.0173, se 0.0052 — no systematic shift |
| **package from the 11 retrained weights** | **both gates SUBMITTABLE**; ensemble 84.3051 vs shipped 84.3029, twins 82.3596 vs 82.3600 |

Run ids: `20260921-1712..1720xx-*-ilse-repro-<stage>`, `20260921-0624/0625-*-ilse-repro-vit-*`,
end-to-end `20260922-111450-*-ilse-repro-phase3-build`.

### 9.1 Three things that break on any machine but the author's — fix these before handing over

1. **`reproduce.sh` defaults `PY` to the author's conda env** (`/users/PAS1187/qwu/.conda/envs/realpde/bin/python`,
   line 40). Anywhere else it dies at `PREFLIGHT FAILED: train2.py` before doing anything. Pass `PY=…` explicitly,
   or make the default fall back to `python3`.
2. **`Track_2/submit*/{_fno.py,submission.py}` are symlinks into `Track_1/submit_fno/`.** `build_round13.sh`
   bakes constants into Track 1's `_fno.py` and then checks Track 2's copy against it. Any transport that
   dereferences symlinks — `tar -h`, `zip`, a payload size limit, some file-copy UIs — turns them into real
   files, and then **Track 1 builds while Track 2 refuses** with `submit/_fno.py is not Track 1's _fno.py;
   the tracks would diverge`. Ship the tree with symlinks intact, or recreate them before building.
3. **`validate_strict.py` needs `Track_1/val_unseen.npz` at GATE time**, not only for scoring. Build the
   validation sets (`make_val.py`) *before* `reproduce.sh`, or the Track 1 gate fails with
   `FileNotFoundError` on a package that is otherwise fine.

None of the three affects the shipped archives; all three stop a from-scratch reproduction on a fresh machine,
which is exactly what the organizers will attempt.
