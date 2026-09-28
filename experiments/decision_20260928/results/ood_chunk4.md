# OOD instrument, chunk 4 (2026-09-28, executor "gauge")

Code: `experiments/decision_20260928/ood/`.
* New: `ood_blank.py`, `ood_i6dump.py`, `ood_i6.py`, `ood_i5.py`, `ood_arms.py`, `ood_poll.sh`, `ood_after.sh`,
  `ood_timing.sh`.
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

<!-- PART3 -->

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

<!-- END -->
