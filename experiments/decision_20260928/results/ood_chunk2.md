# OOD instrument, chunk 2 (2026-09-28, executor "gauge")

Stand-ins: A20 = `aoa20_ffnosim_s1, s2 + aoa20_fno_s1`; A10 = `aoa10_ffnosim_s1, aoa10_ffno_s1 + aoa10_fno_s1,
aoa10_nosim_s1`, each with the band refitted to it. ID = the five twins of record with the band of record, on the 8
`VAL_TRAJ` trajectories.

Composite = `final_v9.W0` · (rel, tke, mvpe, sps), without the time term. Deltas are pooled; (n/N) counts the trajectories
whose own delta is > 0. Reports are in `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs/{id_twins,a20_standin,a10_standin,*_shipband}/report.md`.

**GPU use for this chunk: 0.78 GPU-h** (sacct, 5 jobs). Both chunks together: 1.13 GPU-h.

## A. Which windows the shipped band is fitted on

**My earlier sentence was correct.** The static band's error field comes from windows that the twins trained on.

**Static band (`r38_a080_c0_fill.npz`, the base of `k27_sigblend_lam0.5.npz`).**
* **Which windows:**
  * The build calls `fit_band.py --ckpt $A --err-ckpt $M` without `--fit-on` (`Track_1/submit_fno/build_round13.sh:105`,
    `reproduce.sh:393`).
  * The defaults are `--fit-on seen` and `--fit-aoa 15 20` (`fit_band.py:93`, `:98`).
  * `fit_d = ds` = `Track_1/val_seen.npz` (`fit_band.py:191`, `:211`). The error field is
    `Pf = infer(m_err, fit_d["x"])[keep]` on the AoA 15-20 windows (`:212-214`), computed from the five twins (`--err-ckpt`).
  * `sigma_ref` comes from the all-data members on the same windows (`:220-221`).
* **Whether the twins trained on them: yes.**
  * `val_seen.npz` is `make_val.SEEN` (`make_val.py:28-29`, `:94`). The file says: "SEEN: 8 trajectories that WERE in our
    models' training data" (`make_val.py:4`).
  * The twins were trained without `--exclude` (their `config["exclude"]` is None). train2.py then excludes only
    `VAL_TRAJ` (`train2.py:841`), so every SEEN trajectory is in their training set.
  * `MANIFEST.md:236-238` states the construction (`--err-ckpt <the five held-out twins>`) but not the fit split. "Held-out
    twins" means held out of `VAL_TRAJ`, not of the fit split.
* **How much:** SEEN ∩ AoA 15-20 = 10125_20, 12675_20, 22875_15. That is **3 trajectories, 126 windows** (42 each), all
  in sample. The 1.39 inflation is applied on top of these in-sample errors.

**SIGBLEND constants (κ per frame and channel, q per channel).**
* `dump_twins.py` defaults to `--val Track_1/val_unseen.npz` (`dump_twins.py:17`), and `make_k0_band.py` fits on that dump.
  `MANIFEST.md:247` says "κ and q were fitted on the eight `val_unseen` trajectories".
* That is **8 trajectories, 336 windows, all AoA, out of sample** (= `VAL_TRAJ`).
* The evidence for κ and q is the leave-one-trajectory-out table.

**How the regime band differs from the shipping construction.**

| | static band error field | SIGBLEND κ, q |
|---|---|---|
| shipped | val_seen AoA 15-20: 3 trajectories, 126 windows, **in sample** | val_unseen: 8 trajectories, 336 windows, out of sample |
| regime A20 (instrument) | VAL_TRAJ \ A20, AoA 15-20: 1 trajectory (21600_15), 42 windows, **out of sample** | VAL_TRAJ \ A20: 6 trajectories, 252 windows, out of sample |
| regime A10 | 3 (8850_20, 11400_20, 21600_15), 126 windows, out of sample | 6, 252, out of sample |
| regime RE | 3 (same three), 126 windows, out of sample | 6 (VAL_TRAJ minus 3750_0, 26700_0), 252, out of sample |
| ID read | the band of record itself (in sample on val_seen) | the band of record |

**What this does to the comparison.** An out-of-sample static fit should give a wider band than the in-sample one. So
the regime band could flatter containment relative to what an organiser rebuild of the same members would ship, and
relative to the in-distribution read. I measured the size of this with a new option, `OOD_BANDSTYLE=ship`. It fits the
static band the shipping way: `--fit-on seen`, SEEN AoA 15-20 minus the regime. These are trajectories the regime
ensemble trained on: 1 at A20 (22875_15), 3 at A10. SIGBLEND is unchanged.

| stand-in | static band median, held-out / ship | control composite, held-out / ship | sps | control − base, held-out / ship |
|---|---|---|---|---|
| A20 | 0.01169 / 0.01153 | 73.4329 / 73.4209 (−0.012) | 45.562 / 45.514 | +0.4529 / +0.4601 |
| A10 | 0.00937 / 0.00930 | 76.0100 / 76.0082 (−0.002) | 52.465 / 52.458 | +0.5673 / +0.5718 |

The choice moves the level by at most 0.012 and the stream-stack gain by less than 0.01, so the design's out-of-sample
fit is not a material advantage. This was measured on stand-ins; `ood_run.sh` can repeat it on the mirror ensembles at
A20 and A10. At RE the shipping-style split cannot be selected by AoA (12675_20 is in RE), so `ood_lib` refuses rather
than fit the wrong split. What does differ between a regime and the ID read is the amount of fit data at A20: one static
trajectory against three.

## B. Arm I2: STATS_MASK on a re-blanked recording

**Masks.** Each mask is set to the physical zero in u and v of every input frame and every target frame, constant in
time (`ood/ood_reblank.py`). The flow runs toward the higher column index, and the airfoil's zero set is at columns 0-20.
* **m1:** downstream border, columns 58-63 (192 px).
* **m2:** the recording's static zero set dilated by 3 px (258-372 px, of which 205-265 are new).
* **m3:** near wake, rows 12-21 × columns 20-39 (200 px). It covers mvpe probes.

**Result.** Composite Δ vs `control` of the same mask, with sps Δ; brackets are n+ over trajectories.

| regime | mask | statsmask − control: composite | sps | prev_input − control: composite | sps |
|---|---|---|---|---|---|
| ID all 8 | plain | −0.0012 (4/8) | −0.003 (5/8) | −0.0064 (0/8) | −0.026 (0/8) |
| ID all 8 | m1 | **−4.768 (0/8)** | −7.00 (0/8) | −0.0115 (0/8) | −0.046 (0/8) |
| ID all 8 | m2 | −0.096 (0/8) | −0.174 (0/8) | −0.0046 (0/8) | −0.019 (0/8) |
| ID all 8 | m3 | −2.404 (0/8) | −3.87 (0/8) | −0.0100 (0/8) | −0.040 (0/8) |
| ID AoA 15-20 | m1 / m2 / m3 | −3.184 / −0.059 / −0.732 (0/3 each) | −4.39 / −0.094 / −1.23 | −0.0069 / −0.0037 / −0.0072 (0/3) | |
| A20 | plain | −0.0006 (2/16) | −0.001 (4/16) | −0.0023 (0/16) | −0.009 (0/16) |
| A20 | m1 | −0.850 (0/16) | −1.15 (0/16) | −0.0064 (0/16) | −0.026 (0/16) |
| A20 | m2 | −0.016 (0/16) | −0.024 (0/16) | −0.0044 (0/16) | −0.018 (0/16) |
| A20 | m3 | −0.600 (0/16) | −1.04 (0/16) | −0.0082 (0/16) | −0.033 (0/16) |
| A10 | plain | −0.0001 (11/18) | +0.001 (14/18) | −0.0064 (0/18) | −0.026 (0/18) |
| A10 | m1 | −1.121 (0/18) | −1.75 (0/18) | −0.0087 (0/18) | −0.035 (0/18) |
| A10 | m2 | −0.113 (0/18) | −0.225 (0/18) | −0.0043 (0/18) | −0.017 (0/18) |
| A10 | m3 | −2.341 (0/18) | −4.17 (0/18) | −0.0117 (0/18) | −0.047 (0/18) |

**I2 fails its own criterion.** The masked build does not beat the unmasked one on any re-blanked recording; it loses on
every trajectory under every mask. This is far worse than the arm's hypothesis allowed, so I checked the mechanism before
accepting it:
* Containment C is unchanged (|ΔC×100| ≤ 0.01 in every case). The whole loss is in rel / tke / mvpe. Example, A10 m3:
  rel error +0.022, tke +0.171, mvpe +0.083.
* The kit computes these three metrics, and the accuracy factor inside sps, over the **whole** field, blanked elements
  included (`scoring.py`: "The accuracy factors … are not masked"). A blanked element therefore has target 0 and still
  counts.
* Without STATS_MASK, D1c's per-trajectory bias and VARCAL's variance ratio learn the current recording's blanked region
  from its own revealed targets. They pull the prediction toward 0 there within a few windows, and fluctuations there
  are clipped down (the gain floor is 0.25). STATS_MASK turns exactly that learning off.
* Because the statistics are per trajectory and reset between trajectories, "learning the mask" is learning **this**
  recording's mask. That is what is wanted on a private set whose blanking is different. The audit's premise (M2) does
  not hold under the kit's scoring.

The effect is small on m2, which mostly borders the recording's own zero set, and large on m1 / m3, which blank new flow
regions. `prev_input` loses on every trajectory under every mask as well. Its unscored test never fires, so R1 counts the
blanked elements as scored; its loss grows with the mask from 0.006 to 0.012. **This supports I1 (`PREV_SPACE = "target"`).**

**Transient zeros.** Across all 82 recordings on the 32 × 64 grid:
* 1.2-6.0 % of pixels are static zeros; 3.2-16.0 % (mean 8.5 %) are transient, i.e. zero in some but not all frames;
  the maximum is 26700_15.
* 63 % of all zero elements sit at transient pixels.
* Within one 20-frame block, 0.9-5.6 % of pixels are partly zero.
* 1.4 % of elements are zero in only one of u, v.

How the package treats them:
* **R1's unscored test and the kit's test are elementwise per frame and per channel.** R1 uses
  `(gyp − r1_z).abs() >= 1e-6`; the kit uses `t != 0`. A transient zero is unscored in exactly the frames where it is
  zero, in both.
* **STATS_MASK in D1c** zeroes that element-frame's error, then averages over all 20 frames. A pixel scored in k of 20
  frames therefore has its time-mean error diluted by k/20, rather than averaged over its scored frames.
* **STATS_MASK in VARCAL** accepts a pixel's variance pair only from windows where all 20 of its frames are scored. A
  pixel with any transient zero drops out for that window.
* **Without STATS_MASK,** transient zeros enter D1c and VARCAL as targets of 0. By the result above, this helps the
  scored metrics.
* **Under `PREV_SPACE = "input"`,** which is the uploaded package, the test never fires on the board, and every zero is
  treated as scored by R1.

## C. Arm I5 dry run: one stream constant at a time (Δ vs control)

Format: Δ (n+) · Δ over the first 3 / first 14 windows. Every variant = control with one constant moved.

| variant | ID all 8 | ID AoA 15-20 | A20 | A10 |
|---|---|---|---|---|
| c_r1_n0_1 | +0.0028 (4/8) · +0.0170 / +0.0065 | −0.0024 (0/3) · −0.0039 / −0.0067 | +0.0045 (16/16) · +0.0244 / +0.0110 | +0.0025 (14/18) · +0.0271 / +0.0066 |
| c_r1_n0_4 | −0.0075 (3/8) · −0.0261 / −0.0170 | +0.0004 (2/3) · −0.0067 / +0.0017 | −0.0103 (0/16) · −0.0360 / −0.0241 | −0.0075 (0/18) · −0.0387 / −0.0192 |
| c_r1_blk_n0_10 | −0.0003 (2/8) · +0.0052 / +0.0020 | −0.0027 (0/3) · +0.0019 / −0.0032 | +0.0051 (16/16) · +0.0080 / +0.0077 | +0.0003 (11/18) · +0.0079 / +0.0032 |
| c_r1_blk_n0_40 | −0.0023 (1/8) · −0.0031 / −0.0037 | −0.0007 (1/3) · −0.0017 / −0.0003 | −0.0083 (0/16) · −0.0044 / −0.0081 | −0.0030 (0/18) · −0.0048 / −0.0049 |
| c_d1c_n0_10 | −0.0076 (2/8) · −0.0108 / −0.0134 | −0.0141 (0/3) · −0.0216 / −0.0216 | −0.0015 (6/16) · +0.0076 / −0.0005 | +0.0017 (16/18) · +0.0131 / +0.0044 |
| c_d1c_n0_15 | −0.0035 (2/8) · −0.0047 / −0.0059 | −0.0065 (0/3) · −0.0095 / −0.0093 | −0.0002 (12/16) · +0.0049 / +0.0007 | +0.0009 (17/18) · +0.0071 / +0.0021 |
| c_d1c_n0_60 | +0.0006 (4/8) · +0.0019 / +0.0017 | +0.0013 (2/3) · +0.0042 / +0.0022 | −0.0009 (1/16) · −0.0030 / −0.0010 | −0.0005 (2/18) · −0.0042 / −0.0011 |
| c_d1c_n0_100 | +0.0007 (4/8) · +0.0028 / +0.0022 | +0.0016 (2/3) · +0.0062 / +0.0027 | −0.0013 (0/16) · −0.0045 / −0.0017 | −0.0007 (1/18) · −0.0060 / −0.0017 |
| c_d1c_eb_n0_0.5 | −0.0005 (2/8) · −0.0002 / −0.0016 | −0.0009 (0/3) · −0.0016 / −0.0022 | +0.0010 (13/16) · −0.0069 / +0.0009 | +0.0019 (15/18) · +0.0002 / +0.0028 |
| c_d1c_eb_n0_2 | +0.0002 (4/8) · −0.0012 / +0.0009 | +0.0010 (2/3) · +0.0012 / +0.0020 | −0.0035 (1/16) · +0.0046 / −0.0052 | −0.0055 (1/18) · −0.0048 / −0.0103 |
| c_d1c_lead_n0_2.5 | +0.0000 (3/8) · +0.0033 / +0.0002 | −0.0008 (0/3) · +0.0004 / −0.0010 | +0.0018 (16/16) · +0.0027 / +0.0031 | +0.0011 (11/18) · +0.0019 / +0.0034 |
| c_d1c_lead_n0_10 | −0.0006 (4/8) · −0.0030 / −0.0010 | +0.0006 (3/3) · −0.0004 / +0.0002 | −0.0037 (0/16) · −0.0026 / −0.0050 | −0.0029 (5/18) · −0.0020 / −0.0065 |
| c_vc_n0_10 | **−0.0403 (0/8)** · −0.0210 / −0.0233 | −0.0489 (0/3) · +0.0116 / −0.0350 | −0.0161 (3/16) · +0.0166 / +0.0079 | +0.0004 (8/18) · +0.0128 / +0.0350 |
| c_vc_n0_15 | −0.0207 (0/8) · −0.0085 / −0.0078 | −0.0261 (0/3) · +0.0086 / −0.0141 | −0.0035 (6/16) · +0.0113 / +0.0119 | +0.0071 (10/18) · +0.0088 / +0.0276 |
| c_vc_n0_60 | +0.0042 (6/8) · +0.0020 / −0.0040 | +0.0097 (3/3) · −0.0069 / −0.0006 | −0.0164 (2/16) · −0.0084 / −0.0195 | **−0.0270 (0/18)** · −0.0066 / −0.0301 |
| c_vc_n0_100 | +0.0005 (5/8) · +0.0022 / −0.0086 | +0.0096 (3/3) · −0.0103 / −0.0039 | −0.0345 (1/16) · −0.0124 / −0.0320 | −0.0519 (0/18) · −0.0098 / −0.0475 |

This is a dry run, not a decision. What it shows about the machinery:
* Every variant built, streamed and was checked (knobs, live flags, bit-identical first window).
* Every constant except `VC_N0` sits on a plateau within ±0.01 of the shipped point in all four reads.
* `VC_N0` is the only constant that pulls in opposite directions: lower is better out of distribution (at A10, 10 and 15
  read +0.000 and +0.007, and +0.028 to +0.035 over 14 windows), higher is better in distribution (10 reads −0.040, 0/8).
  A leave-one-regime-out pick on the full-size ensembles will be decided mainly by `VC_N0` and `R1_N0`.
* `R1_N0` 1 reads +0.002 to +0.005 in three of the four reads (not on AoA 15-20) and is concentrated in the first 3
  windows.

## D. Bound for arm I4: gain by number of revealed windows n

Composite of the windows with n revealed targets before them (n = the window's index in its stream).

| n | ID all 8: base / control / gain | A20: base / control / gain | A10: base / control / gain |
|---|---|---|---|
| 0 | 76.607 / 76.607 / 0 | 72.298 / 72.298 / 0 | 75.151 / 75.151 / 0 |
| 1 | 76.575 / 76.585 / +0.010 | 72.307 / 72.398 / +0.091 | 75.266 / 75.328 / +0.063 |
| 2 | 76.763 / 76.772 / +0.008 | 72.733 / 72.844 / +0.111 | 75.203 / 75.414 / +0.211 |
| 3 | 77.086 / 77.134 / +0.048 | 72.879 / 73.119 / +0.240 | 75.036 / 75.297 / +0.261 |
| 4-7 | 76.869 / 76.920 / +0.051 | 72.616 / 72.952 / +0.336 | 75.452 / 75.881 / +0.429 |
| 8-13 | 77.289 / 77.354 / +0.065 | 72.916 / 73.330 / +0.414 | 75.366 / 75.941 / +0.575 |
| 14-27 | 77.292 / 77.375 / +0.083 | 73.195 / 73.722 / +0.526 | 75.546 / 76.182 / +0.636 |
| 28+ | 77.216 / 77.313 / +0.097 | 73.035 / 73.572 / +0.538 | 75.449 / 76.108 / +0.659 |

**Hindsight bound.** The whole-stream gain if the windows with n ≤ 2 had the 8-13 bin's gain, to first order:
Σ_{b=0,1,2} (N_b/N)(g_8-13 − g_b). The first-order form is exact to 3e-4 on these data: the window-share-weighted bin
composites reproduce the pooled composite to 3e-4.

| regime | full streams | first 14 windows of each stream |
|---|---|---|
| ID all 8 | +0.004 | +0.013 |
| ID AoA 15-20 | −0.002 (the 8-13 gain is negative there) | −0.005 |
| A20 stand-in | **+0.026** | +0.075 |
| A10 stand-in | **+0.035** | +0.104 |

**On full streams the bound is below +0.03 at A20 (+0.026) and just above it at A10 (+0.035).**
* It is an upper bound on anything a window-1-3 correction can buy: it gives the early windows the full gain of an
  8-13-window history.
* On 14-window streams it is about 3× larger.
* The hidden streams are 14-99 windows, so what I4 is worth depends on how short they are.
* The mirror ensembles will give the real read. Their gain curve can differ from the stand-ins'.

## E. Prepared commands (not run)

From `/users/PAS1187/qwu/workspace/realpde/main/experiments/decision_20260928/ood/`.

**Readiness check (login node, no GPU).**
```bash
bash ood_ready.sh A20            # also A10, RE; optional layout list, default "mirror six vitc2"
```
* It prints `READY` or `NOT READY <layout>: <member>(no .pth | no sentinel)` per layout, and exits 0 only if every
  requested layout is ready.
* A member is ready when its `.pth` exists and either `dec0928/logs/{rp,pz}_dp_<regime>_<member>.done` exists or the
  member is listed in `ckpt/<REGIME>/REUSED.txt` (a20_ffnosim_s1, a20_ffnosim_s2, a10_ffnosim_s1).
* Status at 05:55: A20 has s1, s2 (reused), A10 has s1; everything else is still training.

**One command per regime.**
```bash
bash ood_wave.sh A20
bash ood_wave.sh A10
bash ood_wave.sh RE
# or all at once, in two afterany chains:          bash ood_wave.sh A20 A10 RE
# after a given job:                               OOD_DEPEND=<jobid> bash ood_wave.sh A20
# dry run:                                         OOD_WAVE_DRY=1 bash ood_wave.sh A20 A10 RE
```
* For each regime it submits the layouts that `ood_ready.sh` reports READY, skipping the others by name, in at most two
  concurrent `afterany` chains.
  * **`<r>_mirror`** (limit 1:45): band refit, build, then `frozen base control prev_input statsmask famw`, the 16 `c_*`
    constants, and `control` / `statsmask` under m1, m2 and m3. That is 28 streams, about 30 min at A20 and A10 and about
    45 min at RE.
  * **`<r>_six`** and **`<r>_vitc2`** (limit 0:45): band refit, build, then `frozen base control`.
* **Outputs:** `runs/<r>_<layout>/report.md` and `summary.json`.
* **Where G0 is read:** the `control` row's `Δ base (n+)` in `<r>_mirror`.
* **Re-running a single piece by hand:** `OOD_CSET=1 OOD_REBLANK="m1 m2 m3" OOD_REBLANK_VARIANTS="control statsmask"
  bash ood_submit.sh a20_mirror A20 mirror 01:45:00`.
* **Estimated cost:** about 0.5 + 0.2 + 0.2 GPU-h per regime, about 3 GPU-h for all three regimes.

## Verified, and not verified

**Verified**
* Part A, from the code with file and line; my earlier sentence stands.
* The shipping-style static fit, measured on both stand-ins: it moves the composite by at most 0.012.
* Every new variant and the re-blank path ran on the real package code:
  * the diffs are checked;
  * the first-window identity to `frozen` holds for every `c_*` and every non-FAMW variant;
  * the kit self-check passes to 1e-8.
* The I2 result was traced to its mechanism: C is unchanged, and the loss is in the accuracy metrics that the kit
  computes over blanked elements.
* The first-order I4 bound: exact to 3e-4.
* `ood_ready.sh` and `ood_wave.sh` were tested on a mock checkpoint tree (readiness, skipping, chain construction).

**Not verified**
* Nothing here has run on a full-size regime ensemble yet (none exists).
* `ood_wave.sh` has not submitted a real job; its submit path is `ood_submit.sh`, which has.
* The re-blank masks are synthetic, and m3's rectangle is placed by eye from the zero sets.
* The `c_*` reads are single stand-in ensembles, one per regime, not a decision.
* The shipping-style band comparison is not available at RE.
