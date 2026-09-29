# Decision-phase campaign, opened 2026-09-28

Owner's grant (2026-09-28): **300 GPU-hours, at most 20 jobs at a time.** Planner: the main session. Executors: Opus
agents. State is in `STATE.md` next to this file; results go to `results/`. Scratch: `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/`.

Every criterion below was written on 2026-09-28 before any job of this campaign had finished.

## 1. What is being optimised

Track 2 is shortlisted (#8); Track 1 is not (#20). The organizers retrain the method from scratch and score a private
set at novel AoA / Re, once. So the objective is **the Track-2 composite on conditions the models never saw**, and the
development board is no longer a target. Track 1 reads come for free from the same ensembles and are reported, not
acted on.

**Composite** = the fitted `final` without its `time` term, weights from `Track_1/final_v9.W0`, read on the real code
path (`Track_2/stream_eval.py` on a built package, revealed target normalised as the official runner does).
Latency is reported separately in A100 ms per step and never added to an expected value.

## 2. Why anything is reopened

`ROADMAP.md` Part 2 closed the training side and most of the band **in distribution**, where the residual is limited by
information (§1.37). Out of distribution the error budget is different: at a held-out AoA 20 the composite falls from
77.1 to 72.9, the mvpe error doubles (0.066 → 0.122), and the two levers that were read there paid 5 – 10× their
in-distribution value (sim co-training +1.12 per member, the stream stack +0.45 … +0.55). The new evidence that
reopens a direction is therefore **a read in the held-out regimes at the member count that ships**, which no
training-side arm except sim has ever had. A direction whose mechanism cannot differ out of distribution stays closed.

Known weaknesses of the existing out-of-distribution reads, which wave 1 removes:

* the ensembles are stand-ins (3 – 4 members, one seed each, FNO w96 as the second family, no ViT);
* their band was not refitted to them;
* no ensemble exists for extrapolation in Re, although the release's own hints point there (sim covers Re 15225 and
  27975, which real does not; the Track-2 example file is Re 2968);
* a single model's seed spread is 12 – 58× larger out of distribution than in it (§1.12), so one run is not a read.

## 3. Regimes

Held-out lists are derived in code, never typed (rule 3). Every regime model also excludes `model.VAL_TRAJ`, which is
its in-distribution fit split for the band (minus any trajectory of the held-out regime). Sim trajectories at a
held-out condition are excluded too (`--sim-exclude-held`): the lower bound of sim's value.

| regime | held out of training | evaluation trajectories | stands for |
|---|---|---|---|
| `A20` | every real and sim trajectory at AoA 20 | 16 real at AoA 20 | a novel angle, extrapolated |
| `A10` | every real and sim trajectory at AoA 10 | 18 real at AoA 10 | a novel angle, interpolated |
| `RE` | `re_split.excluded()` (Re 3750, 5025, 12675, 19050, 25425, 26700) and sim at those Re | `relo` 9, `rehi` 7, `rein` 10 | Re 2968-like, Re 27975-like, and the interior control (Re 15225-like) |
| in distribution | `VAL_TRAJ` | the 8 trajectories, and their AoA 15 – 20 slice | the floor; the twins of record |

"OOD mean" = the unweighted mean of `A20`, `A10`, `relo`, `rehi`. `rein` is reported with the in-distribution reads.

## 4. Wave 1 — the instrument (about 65 GPU-h)

Regime ensembles at the recipe that ships (`reproduce.sh`: `FFNO + SIMF`, `VITP`, `VITC + SIMF`), 400 epochs.

| regime | F-FNO + sim seeds | ViT | reused |
|---|---|---|---|
| `A20` | 1 – 6 | `VITP` s1, `VITC` s3, `VITC` s1 | `screen/aoa/aoa20_ffnosim_s1,s2.pth` if their config equals the recipe |
| `A10` | 1 – 6 | the same three | `screen/aoa/aoa10_ffnosim_s1.pth`, same condition |
| `RE` | 1 – 6 | the same three | none |

Seeds 1 – 4 with `VITP` s1 and `VITC` s3 are the ensemble that mirrors the package of record. Seeds 5 – 6 and `VITC` s1
are there for arms T1 and T2 and for the seed spread of a four-member draw.

**Gate G0 (the instrument must show what the report claims).** On the mirror ensembles, with the band refitted to each
and `PREV_SPACE = "target"` on both sides: the shipped stream stack against asymmetric R1 alone reads **≥ +0.15 composite
on `A20` and on `A10`, positive on ≥ 80 % of trajectories**. If it reads below +0.10 on either, the out-of-distribution
claims of the phase report are withdrawn in `STATE.md` and every stream arm below is re-priced before it runs.

## 5. Arms and their criteria

Deltas are composite, against the mirror ensemble with the package of record's constants plus `PREV_SPACE = "target"`
(the control). "In distribution" is the twins-of-record package on all 8 `VAL_TRAJ` trajectories.

### Inference side (no training; about 25 GPU-h in all)

| id | change | ships if |
|---|---|---|
| I1 | `PREV_SPACE = "target"` | a fix, already +0.0064 (8/8) in distribution; confirm ≥ 0 in every regime |
| I2 | `STATS_MASK = True` | ≥ −0.005 in every regime, **and** on a synthetic re-blanked recording (a different masked region) the masked build beats the unmasked one. Robustness, not score |
| I3 | FAMW, residual accumulated after the reweighting | OOD mean ≥ +0.05 with `A20` and `A10` both positive; in distribution ≥ −0.01; ≤ +0.15 ms per step |
| I4 | early-window prior: widen the interval in windows 1 – 3 by the family disagreement | ΔC×100 ≥ +0.20 over windows 1 – 3 in ≥ 3 of the 4 OOD regimes; in distribution ≥ −0.01 composite |
| I5 | the stream constants (`D1C_N0`, `D1C_EB_N0`, `D1C_LEAD_N0`, `VC_N0`, `R1_N0`, `R1_BLK_N0`) chosen leave-one-regime-out | the constants picked on three regimes gain ≥ +0.03 on the fourth, in all four rotations, and in distribution ≥ −0.01 |
| I6 | band base and SIGBLEND κ, q refitted on pooled held-out-condition errors, leave-one-regime-out | ΔC×100 ≥ +0.30 OOD mean with ≥ 3 of 4 regimes positive; in distribution `sps` ≥ −0.05. Stated in ΔC; local containment over-reads (§1.34), so the bar is twice the in-distribution one |

**Added 2026-09-28 05:58 EDT (commit d701287), before any read of it:**

| id | change | why it is new | ships if |
|---|---|---|---|
| I7 | zero propagation: an element that is the physical zero in every one of the 20 input frames is predicted as the physical zero in all 20 output frames. Uses the window's own input only, so it is legal on both tracks. A knob, bit-identical when off | the scorer's `rel`, `tke`, `mvpe` and the accuracy factor of `sps` are norms over the **whole** field, blanked elements included (`scoring.py` lines 87 – 91, 206 – 210); only the interval average and the coverage leave them out. The documents said blanked elements "are not scored", so nothing ever asked what the members predict there. A novel angle has a different blanked region | OOD mean ≥ +0.03 with ≥ 80 % of trajectories positive; in distribution ≥ 0.00 with ≥ 6 of 8 trajectories not negative; on the three synthetic re-blanks of arm I2, better than the control |

I7 is screened first on dumped predictions (where the squared error sits, by the state of the input at that element),
then, if the screen shows at least half of the bar, built as a knob and read on the real code path.

### Training side (about 75 GPU-h)

Seed-matched to the control, read at ensemble level (the arm's four F-FNO members with the control's two ViTs), band
refitted to each ensemble.

| id | change | new evidence that reopens it | ships if |
|---|---|---|---|
| T1 | six fused F-FNO members instead of four | member count was closed at constant latency in distribution; the seed spread out of distribution is 12 – 58× larger, and averaging is what removes it | OOD mean ≥ +0.05, in distribution ≥ 0, and the A100 latency delta is stated |
| T2 | both ViTs co-trained with sim (`VITC` s1 + s3) | one shipped ViT never saw sim; sim's value is out of distribution and was never read for the ViT pair | OOD mean ≥ +0.05 and in distribution ≥ −0.02 |
| T3 | sim dose `--lam-sim` 1.0 (shipped 0.3) on the F-FNO members | the dose was never opened because sim at the held-out **cell** was a wash in distribution (−0.004); the regime effect it rests on (+1.12) was never dosed | `A20` ≥ +0.10 and the mean of `relo`, `rehi` ≥ +0.10; in distribution ≥ −0.03 |
| T4 | Re-similarity augmentation: a real window resampled in time by a factor s and scaled in amplitude by s, s log-uniform in [0.8, 1.25] (every sample traces back to a released trajectory) | `--aug-scale` alone was null in distribution; it changes the amplitude without the time scale, so it never produced a physically consistent neighbour in Re | mean of `relo`, `rehi` ≥ +0.10; `A20`, `A10` ≥ −0.02; in distribution ≥ −0.03 |

**Added 2026-09-28 11:03 EDT (commit 43322eb)**, after T4's single-member screen on RE (+0.120 rein, +0.523 relo, +0.208 rehi, 4 of 4 seeds
each, against seed spreads of 0.02 – 0.08) and before any read of what follows. T4's own criterion is unchanged.

| id | change | why | ships if |
|---|---|---|---|
| T4 (continued) | the same arm on A20 and A10 (seeds 1 – 4), and three twins on `VAL_TRAJ` for the in-distribution read | required by T4's criterion | as registered above, at ensemble level on the full stream stack |
| T4v | the two ViT recipes with `--aug-resim 0.25` (RE first: ViT-C s1, s3 and ViT-P s1) | the ViT family is the weak member out of distribution, worst where the amplitude is extrapolated (relo: 4.4 – 7.1 behind an F-FNO); the augmentation widens exactly that range | single-member screen on RE: each of the three ≥ +0.20 on the mean of relo and rehi against its seed-matched control, and rein ≥ −0.05. If it clears, A20, A10 and the two ViT twins are trained, and the read is the ensemble whose six members all carry the augmentation: OOD mean ≥ +0.10 over the control ensemble and ≥ +0.03 over the ensemble with only the F-FNO members augmented; in distribution ≥ −0.03 |
| T4b | `--aug-resim 0.5` (s in [0.67, 1.5]) on the F-FNO members, RE seeds 1 – 4 | the dose of a lever that paid; one step, not a sweep | replaces 0.25 only if it beats 0.25 by ≥ +0.05 on the mean of relo and rehi with 4 of 4 seeds and rein is not worse by more than 0.02; otherwise 0.25 stands |

T3 and T4 run only on the regimes their criterion names. An arm that does not move its own training loss or its
single-member held-out read by more than the control's seed spread is closed without an ensemble read.

### Closing gates (about 85 GPU-h, reserved)

* **R1. From-scratch reproduction** of whatever is proposed (`reproduce.sh`, clean clone, all eleven trainings, band
  refit if members changed): package composite within ±0.005 of the build from the campaign's checkpoints. 37 – 48 GPU-h.
* **R2. A different card.** The full Track-2 package on a Pitzer V100: every load-time choice (fp16 check, graph capture,
  arena) must fall back to the same numbers.
* **R3. If a training arm ships**: all-data members and twins of the new recipe, trained once. About 48 GPU-h.

## 6. Budget

| item | GPU-h |
|---|---|
| wave 1, instrument | 65 |
| inference-side arms | 25 |
| training-side arms | 75 |
| closing gates | 85 |
| reserve | 50 |
| **total** | **300** |

The ledger is `sacct` over job names `rp_dp_*` (Ascend) and `pz_dp_*` (Pitzer), elapsed × GPUs, recorded in
`STATE.md` at every wave. At most 20 jobs of this campaign run at once (wave-1 training 18, instrument work 2); jobs
beyond the cap wait on a Slurm dependency.

## 7. Decision rule

* The package of record stays the default. A change enters the proposal only if it passed its own criterion.
* A switch in doubt stays off. Nothing is staged for upload; the owner decides, and HOLD is valid.
* Every change that enters is a knob that reproduces the package of record bit for bit when off, is built by
  `build_final.sh`, and has its band in `Track_1/bands/` with a `MANIFEST.md` entry.
* The proposal carries two sets of predicted subscores: in distribution and held-out regimes.

## 8. Rules of the campaign

* Jobs run from a worktree pinned to a commit on scratch (`dec0928/wt_base`), never from the working tree.
* Checkpoints, logs, dumps and job scripts on scratch. Nothing generated under `$HOME`.
* Every `sbatch` excludes the owner's desktop node. Other projects' jobs are never touched.
* No upload, no push. `Track_1/unet/`, `Track_1/submit_fno/*.py`, `Track_2/submit/*.py` and the build scripts in the
  working tree are not edited by this campaign; new code lives under `experiments/decision_20260928/` or in a scratch
  worktree until an arm passes.
