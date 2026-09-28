# OOD instrument, chunk 4 (2026-09-28, executor "gauge")

Code: `experiments/decision_20260928/ood/`.
* New: `ood_blank.py`, `ood_i6dump.py`, `ood_i6.py`, `ood_i5.py`, `ood_arms.py`, `ood_layouts.py`, `ood_poll.sh`,
  `ood_after.sh`, `ood_batch5.sh`, `ood_timing.sh`.
* New stages in `ood_run.sh`: `i6dump`, `i6fit`, `idread`.
* New layouts in `ood_lib.py`: ffno4, ffno6, six_vitc2, t3, t4, twins_ffno.
* FAMW's per-window family weight is recorded in `ood_stream.py`, and the interval width in `ood_score.py`.
* Patch: `patches/fit_band_from_dump.patch` with a README; it is applied only inside the I6 runs' own source trees.

Every job now runs from a snapshot of the instrument's code (`ood/jobs/snap/…`). Two jobs died on 09-28 06:31 because I
edited `ood_run.sh` under them; bash reads a script while it runs. Both were re-run.

## Gate G0 (written as each mirror's control and base streams landed)

| regime | control − base (n+) | min per trajectory | first 3 / 14 | criterion (≥ +0.15, ≥ 80 % positive) |
|---|---|---|---|---|
| A10 mirror (job 7571943, read 08:00) | **+0.5386 (18/18)** | +0.254 | +0.098 / +0.393 | pass on A10 |
| A20 mirror (job 7571976, read 08:10) | **+0.6917 (16/16)** | +0.563 | +0.147 / +0.512 | pass on A20 |

**G0 PASSES** at the member count that ships: on both held-out angles the shipped stream stack reads 3.6-4.6× the bar,
with every trajectory positive. A20 levels: base 72.966, control 73.658 (stand-in: 72.980 / 73.433).
Reported, not gated: relo +1.146 (9/9), rehi +0.292 (6/7), rein +0.079 (9/10).

## Part 3: the mirror wave and the arms

**What ran.** Every regime ensemble is complete. All four RE members and the mirror, six, vitc2, ffno4, ffno6,
six_vitc2, T3 and T4 layouts were read. Nothing was substituted.
* The three reused checkpoints come from other commits: `a20_ffnosim_s1` and `a10_ffnosim_s1` from `d7492c7-dirty`,
  `a20_ffnosim_s2` from `cddecd9-dirty`. The members trained in this campaign come from `4e5ed37`; the T4 members come
  from `4e5ed37-dirty` (the T4 patch). They were accepted as they are.
* Every build ended `BUILD_FINAL OK`.
* Causality passed on every `control`.
* The I6 identity check was bit-identical on all three mirror ensembles and on both stand-ins.
* The kit self-check stayed at 1e-8 or better.

Full tables are on scratch: `runs/arms.md` (`ood_arms.py`), `runs/layouts.md` (`ood_layouts.py`) and
`runs/<run>/report.md`.

| arm | criterion (PLAN.md) | read | verdict |
|---|---|---|---|
| G0 | control − base ≥ +0.15 on A20 and A10, ≥ 80 % positive | +0.692 (16/16), +0.539 (18/18) | **PASS** |
| I1 PREV_SPACE target | ≥ 0 in every regime | A20 +0.0020, A10 +0.0066, relo +0.0071, rehi +0.0040, rein +0.0086 (control never worse on any trajectory); in distribution +0.0064 (8/8) | **PASS** |
| I3 FAMW (FW_N0 20) | OOD mean ≥ +0.05, A20 and A10 > 0, in distribution ≥ −0.01, ≤ +0.15 ms | A20 +0.0198 (14/16), A10 −0.0030 (5/18), relo +0.0105 (6/9), rehi −0.0009 (4/7); OOD mean +0.0066; in distribution −0.0007 (2/8); local step +0.014 / −0.010 / +0.020 / +0.020 ms | **FAIL** |
| I4 early-window prior | ΔC×100 ≥ +0.20 over windows 1-3 in ≥ 3 of 4 (not built); coordinator's bar: close if the hindsight bound < +0.03 | bound on full streams A20 +0.039, A10 +0.032, relo +0.053, rehi +0.013 (mean +0.034); first 14 windows +0.11 / +0.10 / +0.16 / +0.04 | **bound above +0.03 in 3 of 4: not closed on the bound** |
| I5 constants, leave one regime out | picks gain ≥ +0.03 on the left-out regime in all 4 rotations, in distribution ≥ −0.01 | left-out A20 +0.017 (16/16), A10 +0.004 (15/18), relo +0.035 (8/9), rehi −0.020 (1/7); in distribution −0.007 (2/8), −0.027 for the rehi picks | **FAIL** |
| I6 band on held-out-condition errors | ΔC×100 ≥ +0.30 OOD mean, ≥ 3 of 4 positive, in-distribution sps ≥ −0.05 | with the stream: −0.018 (1/4 regimes); frozen −0.114 (2/4); in distribution sps −0.103 (0/8) | **FAIL** |
| I7 zero propagation (Z1, post hoc) | OOD mean ≥ +0.03, ≥ 80 % positive; in distribution ≥ 0 with ≥ 6/8; better than control on the re-blanks | plain +0.0003 / +0.0003 / +0.0002 / +0.0001; re-blanks m1 +0.25 to +0.85, m2 +0.008 to +0.041, m3 +0.14 to +0.55, all positive | **FAIL** as a score lever; insurance as in chunk 3 |
| T1 six F-FNO | OOD mean ≥ +0.05, in distribution ≥ 0, latency stated | A20 +0.033 (13/16), A10 +0.024 (18/18), relo +0.087 (9/9), rehi +0.023 (6/7); **OOD mean +0.042**; in distribution +0.004 (regime ensembles, band in sample); **+0.357 ms per step** [+0.354, +0.360] local A100, 4 interleaved rounds | **FAIL** (narrowly) |
| T2 both ViTs with sim | OOD mean ≥ +0.05, in distribution ≥ −0.02 | A20 −0.019 (7/16), A10 +0.008, relo +0.054, rehi +0.019; OOD mean +0.015; in distribution +0.008 | **FAIL** |
| T3 sim dose 1.0 | A20 ≥ +0.10 and mean(relo, rehi) ≥ +0.10; in distribution ≥ −0.03 | A20 +0.013 (9/16); relo −0.082 (0/9), rehi −0.050 (0/7); in distribution −0.005 | **FAIL** |
| T4 Re-similarity augmentation (RE only) | mean(relo, rehi) ≥ +0.10; A20, A10 ≥ −0.02; in distribution ≥ −0.03 | **relo +0.334 (9/9), rehi +0.046 (6/7), mean +0.190**; rein +0.025 (9/10); in distribution (REID) +0.037 (6/6); frozen relo +0.505, rehi +0.174. A20 and A10 not measurable (no T4 ensembles there) | **PASS on every measured term; the A20 / A10 term is open** |

**Checks on T4**, whose gain is far larger than any other arm's:
* The T4 members and the mirror's F-FNO members have identical exclusion lists (all RE trajectories, `VAL_TRAJ`, and four
  names with no recording). They are seed-matched (s1-4), carry the same two ViTs, and have the same sim settings.
* T4's config adds only `aug_resim 0.25, p 0.5`.
* The gain appears in every subscore at relo (rel +0.174, tke +0.502, mvpe +0.084, sps +0.773) and in the frozen read
  (+0.505). It also holds in distribution (+0.037, 6/6).
* The relo trajectories are Re 3750 and 5025, below every training Re of the RE ensemble. That is where a resampling
  augmentation that creates lower-Re-like windows should pay.
* I found nothing wrong with the harness. The verdict still rests on one ensemble per arm (single seed set); a
  replication and the A20 / A10 reads the criterion requires are needed before T4 is proposed.

### Arm I6 in detail

Band fitted on the pooled held-out errors of the OTHER regimes (each from its own ensemble), against the regime's own
band. Real path, same ensemble.

| regime | frozen ΔC×100 (n+) | frozen Δsps | control ΔC×100 (n+) | control Δsps (n+) | control Δcomposite | first window ΔC×100 | width ratio frozen / control |
|---|---|---|---|---|---|---|---|
| A20 | +0.788 (16/16) | +0.565 | +0.151 (16/16) | +0.110 (16/16) | +0.0274 | +0.807 | 0.930 / 0.986 |
| A10 | −1.156 (0/18) | −0.869 | −0.068 (6/18) | −0.053 (6/18) | −0.0131 | −1.061 | 1.047 / 1.004 |
| relo | −0.132 (3/9) | −0.112 | −0.152 (0/9) | −0.117 (0/9) | −0.0292 | −0.117 | 1.067 / 1.019 |
| rehi | +0.043 (3/7) | +0.032 | −0.003 (3/7) | −0.002 (3/7) | −0.0005 | +0.299 | 0.987 / 0.992 |
| in distribution, all 8 (twins, band pooled over all four regimes) | −1.051 (0/8) | −0.819 | −0.129 (0/8) | −0.103 (0/8) | −0.0256 | −0.979 | n/a |
| OOD mean | −0.114 (2 of 4 regimes positive) | | −0.018 (1 of 4) | | | | |

The pooled band moves each regime toward the pooled error level. A20, the hardest regime, gets a band 7 % narrower and
gains containment, which local containment over-reads. The others get a wider band and lose. R1 takes back most of the
difference in the stream. In distribution, the band of pooled out-of-distribution errors is too wide.

### Ensemble composition (coordinator's addition): each layout − mirror, control variant

| layout | A20 | A10 | relo | rehi | rein | OOD mean | in distribution (regime ensembles, band in sample) | frozen (Track 1): A20 / A10 / relo / rehi |
|---|---|---|---|---|---|---|---|---|
| ffno4 (no ViT) | −0.176 (1/16) | −0.061 (0/18) | +0.187 (6/9) | −0.097 (0/7) | −0.094 (0/10) | −0.037 | −0.110 (0/18) | +0.073 / −0.013 / +1.121 / +0.011 |
| ffno6 (no ViT) | −0.114 (3/16) | −0.033 (2/18) | +0.189 (6/9) | −0.056 (2/7) | −0.061 (1/10) | −0.004 | −0.092 (0/18) | +0.142 / +0.031 / +1.137 / +0.046 |
| six | +0.033 (13/16) | +0.024 (18/18) | +0.087 (9/9) | +0.023 (6/7) | +0.016 (10/10) | +0.042 | +0.004 | +0.125 / +0.056 / +0.490 / +0.063 |
| vitc2 | −0.019 (7/16) | +0.008 (14/18) | +0.054 (9/9) | +0.019 (5/7) | +0.014 (8/10) | +0.015 | +0.008 | +0.016 / −0.011 / +0.254 / +0.059 |
| six_vitc2 | +0.014 (10/16) | +0.029 (18/18) | +0.123 (9/9) | +0.036 (7/7) | +0.025 (10/10) | +0.050 | +0.009 | +0.135 / +0.048 / +0.628 / +0.102 |
| t3 (A20, RE) | +0.013 (9/16) | — | −0.082 (0/9) | −0.050 (0/7) | −0.025 (2/10) | −0.039 (3 regimes) | −0.005 | +0.122 / — / +0.107 / +0.008 |
| t4 (RE) | — | — | +0.334 (9/9) | +0.046 (6/7) | +0.025 (9/10) | +0.190 (relo, rehi) | +0.037 | — / — / +0.505 / +0.174 |

In distribution on the twins of record, both bands refitted the shipping way: the F-FNO-only twins − the five twins read
**−0.140 (0/8)** on all 8 (rel −0.045, tke −0.143, mvpe −0.076, sps −0.392) and −0.181 (0/3) on AoA 15-20. That matches
the 0.155 the board charged in T2R15.

**The ViT question.**
* Dropping the ViTs wins big only at relo: +0.19 with the stream, +1.12 frozen. With the stream it loses at A20
  (−0.18 / −0.11), rehi, rein and A10, and it loses 0.09-0.14 in distribution.
* Out of distribution the ViTs hurt the frozen forecast at relo and A20. The stream stack repairs most of that at relo
  and more than all of it at A20.
* **No fixed composition beats the mirror everywhere.** `six_vitc2` is the best static layout (+0.050 OOD mean, every
  regime positive, +0.009 in distribution). It adds two members' latency (six alone: +0.36 ms).

**FAMW's learned weight of family B (the ViTs)**, mean over windows k ≥ 28 (u / v), against the prior w0; famw − control
in brackets:

| layout | A20 | A10 | relo | rehi | rein | w0 |
|---|---|---|---|---|---|---|
| mirror | 0.260 / 0.295 (+0.020) | 0.332 / 0.343 (−0.003) | **0.204 / 0.271** (+0.011) | 0.334 / 0.397 (−0.001) | 0.383 / 0.379 (+0.000) | 0.333 |
| six | 0.221 / 0.255 (+0.022) | 0.294 / 0.300 (−0.004) | 0.168 / 0.227 (+0.004) | 0.299 / 0.350 (+0.004) | 0.342 / 0.335 (+0.002) | 0.250 |
| vitc2 | 0.285 / 0.322 (+0.016) | 0.344 / 0.370 (−0.000) | 0.205 / 0.285 (+0.003) | 0.349 / 0.410 (+0.003) | 0.399 / 0.404 (+0.001) | 0.333 |
| six_vitc2 | 0.245 / 0.282 (+0.018) | 0.306 / 0.326 (+0.000) | 0.168 / 0.242 (+0.000) | 0.314 / 0.363 (+0.009) | 0.358 / 0.360 (+0.005) | 0.250 |
| t4 | — | — | 0.211 / 0.243 (+0.023) | 0.340 / 0.368 (−0.002) | 0.371 / 0.369 (+0.003) | 0.333 |

FAMW moves towards the F-FNO family where the single-member reads said the ViTs are weak: at relo the u weight goes
0.333 → 0.20 and at A20 to 0.26. It moves the other way at rein and rehi. The move is too slow and too partial to pay:
+0.011 at relo against the +0.19 a fixed ffno4 gets there. It is also paid for in the regimes where the ViTs are fine,
so FAMW nets +0.007. FAMW's weight is per pixel; these are means.


## Part 1: whose property is the blanked region?

**Mostly the angle's, and the members generalise natural blanking.**
* The static zero set is the airfoil's shadow. Its size is set by the AoA (AoA 10: 94-125 px in u; AoA 20: 47-90 px) and
  varies with Re within an angle.
* Only 6 (u) and 7 (v) of 2048 pixels are static-zero in all 82 recordings. The union over all 82 is 232 (u) and 311 (v).
* A held-out AoA-20 recording carries 5-10 novel static-zero pixels per channel (static in no training recording). They
  sit in a band along the upper edge of the known shadow (rows 11-14, columns 8-21). A held-out AoA-10 recording carries
  3-8, none of them strictly novel.
* The stand-in members predict |P| = 0.0005 m/s (A20) and 0.0003 m/s (A10) there, against 0.0003 / 0.0002 on the
  shadow pixels the training set shares, and a flow scale of 0.08-0.10 m/s. They predict the physical zero at naturally
  novel blanking.
* The synthetic masks of arms I2 and I7 blank regions far from the shadow (the downstream border, the near wake), where
  the members predicted 0.004-0.012 m/s. So they overstate the exposure to blanking that behaves like the rig's.

(a) Static zero set, pixel count, channel u (v and the "mostly zero" sets are in `ood/blank/blank.md` on scratch):

| AoA \ Re | 3750 | 5025 | 6300 | 7575 | 8850 | 10125 | 11400 | 12675 | 13950 | 16500 | 17775 | 19050 | 20325 | 21600 | 22875 | 24150 | 25425 | 26700 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 63 | 38 | 46 | 46 | 33 | 39 | 31 | 25 | 31 | 32 | - | 64 | 62 | 75 | 42 | 66 | 68 | 102 |
| 5 | 111 | 99 | 83 | 71 | 72 | 69 | 52 | 55 | 56 | 25 | 35 | 29 | 35 | 101 | - | - | - | - |
| 10 | 125 | 112 | 113 | 105 | 114 | 105 | 101 | 94 | 107 | 82 | 94 | 102 | 109 | 124 | 106 | 113 | 105 | 112 |
| 15 | - | 63 | 84 | 67 | 56 | 70 | 44 | 53 | 51 | 48 | 63 | 54 | 57 | 70 | 74 | 67 | 82 | 50 |
| 20 | 82 | 83 | 76 | 57 | 76 | 66 | 47 | 47 | 62 | 68 | 58 | 56 | 66 | 51 | - | 90 | 50 | - |

(b) Union and intersection, static zero set: u 232 / 6, v 311 / 7. "Mostly zero" (> half the frames): union u 334 / v 359,
intersection u 68 / v 73. Within one AoA, static in u or v:

| AoA | recordings | intersection | union | mean per recording |
|---|---|---|---|---|
| 0 | 17 | 25 | 261 | 143.5 |
| 5 | 14 | 22 | 255 | 150.1 |
| 10 | 18 | 76 | 193 | 139.1 |
| 15 | 17 | 22 | 198 | 114.9 |
| 20 | 16 | 29 | 141 | 94.8 |

(c) Novel static-zero pixels in held-out recordings (training set = all 82 minus `VAL_TRAJ` minus the regime).
"Strict" = not even mostly zero in any training recording.

| regime | held-out recordings | novel per recording, u / v | strict per recording, u / v |
|---|---|---|---|
| A20 | 16 | 5-10 / 5-9 | 4-8 / 3-7 |
| A10 | 18 | 4-8 / 3-4 | 0 / 0 |
| relo | 9 | 0-3 / 0-1 | 0 / 0 |
| rehi | 7 | 0-3 / 0-2 | 0-1 / 0-1 |
| rein | 10 | 0-2 / 0-1 | 0 / 0 |

Map for A20 ('#' novel in some held-out recording, '+' static-zero in some training recording; rows 11-31 shown, rows
0-10 are empty):
```
.................##.............................................
...........#######..#...........................................
........#+++++++++##.+..........................................
.....#+++++++++++++++...........................................
..+++++++++++++++++++...........................................
.+++++++++++++++++++++..........................................
+++++++++++++++.++++++..........................................
+++++++++++++++++++++...........................................
++++++++++.....+++++............................................
+++++++++++....+++++............................................
+++++++++++....++++.............................................
++++++++++++..++++..............................................
+++++++++++++++++...............................................
++++++++++++++++................................................
++++++++++++++++................................................
+++++++++++++++.................................................
++++++++++++++..................................................
+++++++++++++...................................................
+++++++++++++...................................................
++++++++++++....................................................
+++++++++++.....................................................
```

(d) Stand-in `frozen`, mean |P| in m/s:

| regime | novel static-zero pixels (element-frames) | strictly novel | shared with training | flow scale (mean abs Y, non-zero) |
|---|---|---|---|---|
| A20 | 0.0005 (179 780) | 0.0005 (137 000) | 0.0003 (1 816 520) | 0.0795 |
| A10 | 0.0003 (129 360) | none | 0.0002 (3 548 160) | 0.0955 |

Per recording, A20 novel pixels read 0.0003-0.0008 and A10 0.0002-0.0005 (`blank.md`). The input is zero there, so part
of this is the members carrying zeros from input to output. The synthetic masks also zero the input, yet they drew
0.004-0.012 m/s, so the difference is the location, not the zero input.

## Part 2: arm I6 machinery, and the stand-in dry run

**Pipeline** (`ood_after.sh i6 X`):
1. Every mirror run dumps what a band fit needs on its held-out windows (`ood_i6dump.py`): P at α 0.8, y0 at α 0, the
   members' σ, Y, and the regime band's σ_ref. It is computed with the repository's `_fno.predict_fno` exactly as
   `fit_band.py` computes it.
2. For target X, the dumps of the OTHER regimes are pooled. Each window's σ ratio is taken against its own ensemble's
   σ_ref.
3. The static base is solved by `fit_band.py` itself, patched with `--from-dump`. The solver, smoothing and fill are
   unchanged; without the flag the file behaves as before.
4. The SIGBLEND κ and q come from the unchanged `make_k0_band.py`, run on the pooled dump.
5. The band carries X's own σ_ref.
6. X's ensemble is built with that band and streamed on the **real code path**: frozen, base (asymmetric R1) and control
   (the full stream). I did not have to replay R1: the real package runs it.

**Identity check (both stand-ins): bit-identical.** Fed a regime's own static fit split, the `--from-dump` path
reproduces the static band that `fit_band.py` fitted by running the models: max |diff| 0 over 122 880 elements.

**Dry run.** A20 fitted on A10's held-out errors, and the reverse (one source regime each, stand-in ensembles). Values
are Δ vs the regime's own band, with per-trajectory counts:

| target | variant | ΔC×100 | Δsps | Δcomposite | width ratio (I6 / regime band) |
|---|---|---|---|---|---|
| A20 (from A10) | frozen | +1.138 (16/16) | +0.815 (16/16) | +0.203 (16/16) | 0.896 |
| A20 | frozen, first window | +1.093 (16/16) | +0.779 | +0.194 | 0.895 |
| A20 | base | +0.328 (16/16) | +0.234 | +0.058 | 0.975 |
| A20 | control | +0.186 (16/16) | +0.135 | +0.034 | 0.981 |
| A10 (from A20) | frozen | −1.122 (0/18) | −0.841 (0/18) | −0.210 (0/18) | 1.063 |
| A10 | frozen, first window | −1.122 (1/18) | −0.844 | −0.210 | 1.062 |
| A10 | base | −0.075 (7/18) | −0.058 | −0.014 | 1.018 |
| A10 | control | −0.033 (8/18) | −0.026 | −0.006 | 1.009 |

**What the dry run shows.**
* With one source regime, the refit mostly transfers the source's error level. A10's smaller errors give A20 a band 10 %
  narrower; A20's larger errors give A10 a band 6 % wider.
* The stream's R1 then takes back most of the difference: ΔC×100 is +0.19 / −0.03 with the stream, against +1.14 / −1.12
  frozen.
* The A20 gain comes from a narrower interval, which local containment over-reads.
* The dry-run OOD mean with the stream is +0.08 ΔC×100, against the criterion's +0.30.
* The full-size read pools three regimes for each target and follows in Part 3.

## Ledger

* **GPU:** 6.42 GPU-h on Ascend in chunk 4 (52 jobs, sacct), plus a 2.5-minute Pitzer CPU job for Part 1. All chunks
  together: 7.85 GPU-h.
* **Wall clock:** the whole wave ran between 07:55 and 11:39 with at most 2 of my jobs running until 11:05, and at most 6
  after the coordinator's addition.
* **Scratch:** runs under `dec0928/ood/runs/`. The I7 dumps of the mirror runs were deleted after analysis. The chunk-3
  stand-in dumps (19.4 GB) and the I6 held-out dumps (about 3 GB) are still there.

## Verified, and not verified

**Verified**
* G0, I1, I3, I5, I6, I7, T1, T2, T3 and T4 were read on the real package code, on complete regime ensembles, each
  against its own refitted band.
* Every build passed both gates and causality.
* The patched `--from-dump` band fit is bit-identical to the model-run fit on 5 ensembles.
* The I5 combined reads match their additive predictions to 0.004.
* The FAMW weight capture leaves every output unchanged: the composites are identical before and after it.
* T4's exclusion lists match the control's.

**Not verified**
* Every arm is one ensemble, i.e. one seed set; T4's gain is not replicated.
* T4 was trained only at RE, so its A20 / A10 term cannot be read.
* The in-distribution reads of the regime ensembles use `VAL_TRAJ` minus the regime. The regime band is fitted on those
  trajectories, so their sps part is in sample. For the twin sets the band was refitted the shipping way, which is in
  sample on val_seen as shipped.
* Local step times come from shared nodes. FAMW's re-streamed runs carry the weight capture's synchronize, so the I3
  latency column quotes the first mirror runs.
* The I4 bound is first order, and an upper bound.

