# Arm I7 (zero propagation): screen on dumped predictions (2026-09-28, executor "gauge")

The screen ran on the ID twins (band of record) and on the A20 and A10 stand-ins (refitted bands), for `frozen` and
`control`, on plain streams and under the chunk-2 re-blanks m1, m2 and m3.

The dumps come from the real package code (`ood_stream.py --dump`). They reproduce the stream json's per-window records
exactly (max |diff| 0 in every set). The rules are applied post hoc and rescored with the kit's per-window decomposition
(`ood_score.py`). Code is in `ood/ood_i7.py`; full tables are in `runs/{id_twins,a20_standin,a10_standin}/i7.md` and
`i7.json` on scratch. Dumps are 3.8 / 7.2 / 8.4 GB under `runs/*/dump/`.

**GPU use: 0.30 GPU-h** (3 jobs; 2.21 GPU-h used over chunks 1-3).

## Verdict

**On the recordings as they are, the rule has nothing to remove.**
* The members already predict the physical zero at every element that is zero in all 20 input frames (S20): mean |P| is
  0.0001-0.0004 m/s, against a flow of about 0.15.
* Target-zero elements carry 0.01-0.06 % of the squared error, and the S20 part of that is 0.003-0.014 %.
* The best rule, Z1, reads **+0.0002 (A20, 16/16), +0.0001 (A10, 18/18), +0.0001 in distribution (7/8 positive, 1
  negative)**.
* The Stage C gate (+0.015 OOD mean) is **not met on the regime reads**. The PLAN criterion's first term (OOD mean
  ≥ +0.03) fails by two orders of magnitude.

**On a recording blanked differently from the training data, the rule is worth a great deal.**
* Z1 gains against `control` on every trajectory under every mask:

  | mask | ID all 8 | A20 | A10 |
  |---|---|---|---|
  | m1 | +0.88 | +0.16 | +0.15 |
  | m2 | +0.020 | +0.006 | +0.022 |
  | m3 | +0.57 | +0.29 | +0.48 |

* Without a stream (Track 1) it gains +0.9 to +5.5 under m1, and +0.8 to +2.6 under m3.
* The members have learned the training recordings' blanking. The stream stack (D1c and VARCAL) learns a new blanking
  from revealed targets over several windows. Z1 knows it at window 0. The gain over the first 3 windows is 4-5× the
  full-stream gain.
* The third term of the PLAN criterion (better than control on all three re-blanks) passes. The in-distribution term
  (≥ 0.00, at least 6 of 8 not negative) passes: +0.0001, 7 of 8.

**Recommendation.** I7 is a **robustness knob, not a score knob.** It costs nothing on our data (it never loses in
distribution or out of distribution). It is insurance against a private recording whose blanked region differs from
ours. Its value is exactly as large as the probability of such a recording, which our data cannot measure. That is the
same premise as I2 (STATS_MASK), but here the sign is right. If the owner wants that insurance, **Z1 is the rule**:
* it is never worse than Z3_m5 or Z3_m10 anywhere, and slightly better on m2;
* Z3_m1 loses in distribution (−0.011, 0/8), because a single zero input frame is a transient zero about 5 % of the time
  (item 3);
* Z2 (the first k output frames only) loses badly on masks, because zeroing only part of the window creates a
  fluctuation (tke −0.1 to −1.4).

Stage C is not prepared, because its gate is written on the OOD mean. If the coordinator counts the re-blank reads as
the relevant read, a design is sketched at the end; it takes a few lines.

## Stage A: where the squared error sits

**Item 1, plain streams.** For elements whose target is zero, by the state of the element in the window's own input.
Each class gives: share of all elements / share of the squared-error sum / mean |P| in m/s.

| regime | variant | target-zero elements | their share of SE | S20 | LAST (zero in last input frame, not all 20) | SOME (zero in some input frame, not the last) | NEVER |
|---|---|---|---|---|---|---|---|
| ID all 8 | frozen | 9.95 % | 0.061 % | 8.50 % / 0.014 % / 0.0003 | 0.96 % / 0.012 % / 0.0005 | 0.45 % / 0.022 % / 0.0011 | 0.04 % / 0.013 % / 0.0033 |
| ID all 8 | control | 9.95 % | 0.059 % | 8.50 % / 0.005 % / 0.0001 | 0.96 % / 0.015 % / 0.0006 | 0.45 % / 0.025 % / 0.0012 | 0.04 % / 0.014 % / 0.0035 |
| ID AoA 15-20 | frozen | 7.75 % | 0.028 % | 6.69 % / 0.006 % / 0.0003 | 0.61 % / 0.004 % / 0.0005 | 0.39 % / 0.008 % / 0.0010 | 0.06 % / 0.010 % / 0.0025 |
| ID AoA 15-20 | control | 7.75 % | 0.027 % | 6.69 % / 0.003 % / 0.0002 | 0.61 % / 0.004 % / 0.0005 | 0.39 % / 0.009 % / 0.0010 | 0.06 % / 0.010 % / 0.0026 |
| A20 | frozen | 6.40 % | 0.015 % | 5.79 % / 0.007 % / 0.0004 | 0.38 % / 0.001 % / 0.0006 | 0.20 % / 0.003 % / 0.0011 | 0.03 % / 0.004 % / 0.0024 |
| A20 | control | 6.40 % | 0.012 % | 5.79 % / 0.003 % / 0.0002 | 0.38 % / 0.001 % / 0.0005 | 0.20 % / 0.003 % / 0.0011 | 0.03 % / 0.005 % / 0.0027 |
| A10 | frozen | 9.96 % | 0.032 % | 8.63 % / 0.007 % / 0.0003 | 0.85 % / 0.005 % / 0.0006 | 0.44 % / 0.013 % / 0.0013 | 0.04 % / 0.007 % / 0.0032 |
| A10 | control | 9.96 % | 0.038 % | 8.63 % / 0.004 % / 0.0002 | 0.85 % / 0.007 % / 0.0006 | 0.44 % / 0.017 % / 0.0014 | 0.04 % / 0.010 % / 0.0039 |

**Item 1, re-blanked.** Target-zero share of SE, with the S20 part in brackets:

| regime | mask | frozen | control | control mean |P| on S20 target-zero |
|---|---|---|---|---|
| ID all 8 | m1 / m2 / m3 | 77.9 % (77.9) / 2.2 % (2.2) / 30.1 % (30.1) | 24.4 % (24.4) / 0.51 % (0.47) / 8.5 % (8.4) | 0.0050 / 0.0005 / 0.0024 |
| A20 | m1 / m2 / m3 | 27.0 % (27.0) / 0.34 % (0.34) / 10.0 % (10.0) | 5.9 % (5.9) / 0.11 % (0.10) / 3.6 % (3.6) | 0.0043 / 0.0005 / 0.0031 |
| A10 | m1 / m2 / m3 | 25.5 % (25.5) / 0.98 % (0.97) / 26.8 % (26.8) | 4.5 % (4.4) / 0.24 % (0.22) / 7.0 % (6.9) | 0.0027 / 0.0005 / 0.0029 |

Under a new mask almost all of the target-zero error is S20, which the rule removes at its root. The stream (`control`)
removes about three quarters of it over time; the remainder, and all of it in the first windows, is what Z1 removes.

**Items 2 and 3, plain streams, control.** Values at lead t = 0 / 1 / 2 / 4 / 9 / 19:

| regime | S20 share of elements | S20 with target NOT zero | P(target 0, given S20) | P(target 0, given zero in the last input frame) |
|---|---|---|---|---|
| ID all 8 | 8.63 % | 0.004 / 0.008 / 0.012 / 0.017 / 0.017 / 0.017 | 0.996 / 0.992 / 0.988 / 0.983 / 0.983 / 0.983 | 0.983 / 0.971 / 0.959 / 0.947 / 0.948 / 0.947 |
| ID AoA 15-20 | 6.77 % | 0.003 / 0.006 / 0.009 / 0.012 / 0.011 / 0.013 | 0.997 / 0.994 / 0.991 / 0.988 / 0.989 / 0.987 | 0.979 / 0.964 / 0.952 / 0.940 / 0.942 / 0.939 |
| A20 | 5.84 % | 0.002 / 0.005 / 0.007 / 0.010 / 0.010 / 0.011 | 0.998 / 0.995 / 0.993 / 0.990 / 0.990 / 0.989 | 0.989 / 0.979 / 0.972 / 0.963 / 0.962 / 0.960 |
| A10 | 8.74 % | 0.003 / 0.007 / 0.010 / 0.014 / 0.014 / 0.014 | 0.997 / 0.993 / 0.990 / 0.986 / 0.986 / 0.986 | 0.984 / 0.971 / 0.960 / 0.949 / 0.948 / 0.948 |

* Persistence after S20 falls only from 0.997 to about 0.985 and is flat after lead 4. There is no lead-time structure
  that would favour Z2.
* A zero in the last input frame alone persists at about 0.95. That is why Z3_m1 touches too many live elements
  (0.2-0.5 % of all elements have a non-zero target).

**Item 4, control by revealed windows n.** Target-zero share of SE, with the S20 part after the slash:

| regime | n 0 | n 1-3 | n 4-13 | n 14+ |
|---|---|---|---|---|
| ID all 8 | 0.050 % / 0.013 % | 0.069 % / 0.011 % | 0.057 % / 0.006 % | 0.060 % / 0.004 % |
| A20 | 0.013 % / 0.006 % | 0.012 % / 0.005 % | 0.011 % / 0.003 % | 0.013 % / 0.003 % |
| A10 | 0.042 % / 0.009 % | 0.031 % / 0.006 % | 0.035 % / 0.004 % | 0.040 % / 0.004 % |

D1c halves the small S20 part with time. Nothing of consequence is left for a window-0 rule on these recordings.

## Stage B: the rules

**The rules.** Each channel is treated on its own:
* **Z1:** P = 0 in every output frame where the input is zero in all 20 frames.
* **Z2_k:** the same, for the first k output frames only.
* **Z3_m:** P = 0 where the input is zero in its last m frames.

The interval becomes [min(L, 0), max(U, 0)]. Entries are Δ composite vs `control` (n+ / n−); the rel / tke / mvpe / sps
deltas; Δ over the first 3 / first 14 windows; and the Track-1 effect, which is the rule applied to `frozen` vs
`frozen`.

| regime | stream | Z1 | Z1 rel / tke / mvpe / sps | Z1 first 3 / 14 | Z1 on Track 1 | Z2_k10 | Z3_m1 | Z3_m5 | Z3_m10 |
|---|---|---|---|---|---|---|---|---|---|
| ID all 8 | plain | +0.0001 (7/8 / 1) | +0.000 / −0.000 / +0.000 / +0.000 | +0.0003 / +0.0002 | +0.0005 (8/8) | +0.0001 (8/8) | −0.0114 (0/8) | −0.0041 (2/8) | −0.0012 (5/8) |
| ID all 8 | m1 | +0.878 (8/8) | +0.568 / +2.544 / +0.000 / +1.425 | +4.193 / +1.771 | +5.505 (8/8) | +0.038 (6/8) | +0.867 (8/8) | +0.874 (8/8) | +0.877 (8/8) |
| ID all 8 | m2 | +0.0203 (8/8) | +0.015 / +0.018 / +0.024 / +0.036 | +0.087 / +0.037 | +0.104 (8/8) | +0.0072 (7/8) | +0.0089 (6/8) | +0.0168 (7/8) | +0.0192 (8/8) |
| ID all 8 | m3 | +0.574 (8/8) | +0.196 / +1.876 / +0.608 / +0.965 | +2.149 / +1.133 | +2.615 (8/8) | +0.107 (7/8) | +0.565 (8/8) | +0.571 (8/8) | +0.573 (8/8) |
| ID AoA 15-20 | plain | +0.0001 (3/3) | +0.000 / −0.000 / +0.000 / +0.000 | +0.0002 / +0.0001 | +0.0002 (3/3) | +0.0000 (3/3) | −0.0007 (0/3) | −0.0001 (2/3) | +0.0001 (3/3) |
| ID AoA 15-20 | m1 / m2 / m3 | +0.597 / +0.0158 / +0.287 (3/3 each) | | | +3.736 / +0.067 / +0.973 | | | | |
| A20 | plain | +0.0002 (16/16) | +0.000 / −0.000 / +0.000 / +0.000 | +0.0003 / +0.0002 | +0.0005 (16/16) | +0.0001 (16/16) | −0.0002 (9/16) | +0.0001 (15/16) | +0.0001 (16/16) |
| A20 | m1 | +0.156 (16/16) | +0.195 / +0.103 / +0.000 / +0.210 | +0.685 / +0.269 | +0.909 (16/16) | +0.012 (11/16) | +0.156 (16/16) | +0.156 (16/16) | +0.156 (16/16) |
| A20 | m2 | +0.0055 (16/16) | +0.005 / +0.002 / +0.007 / +0.008 | +0.015 / +0.008 | +0.017 (16/16) | +0.0026 (16/16) | +0.0055 | +0.0055 | +0.0055 |
| A20 | m3 | +0.288 (16/16) | +0.150 / +0.329 / +0.609 / +0.512 | +0.634 / +0.397 | +0.789 (16/16) | +0.058 (16/16) | +0.287 | +0.288 | +0.288 |
| A10 | plain | +0.0001 (18/18) | +0.000 / −0.000 / +0.000 / +0.000 | +0.0003 / +0.0002 | +0.0003 (18/18) | +0.0001 (18/18) | −0.0006 (4/18) | +0.0000 (13/18) | +0.0001 (16/18) |
| A10 | m1 | +0.150 (18/18) | +0.151 / +0.192 / +0.000 / +0.233 | +0.879 / +0.323 | +1.133 (18/18) | +0.005 (16/18) | +0.149 | +0.150 | +0.150 |
| A10 | m2 | +0.0216 (18/18) | +0.013 / +0.021 / +0.037 / +0.041 | +0.090 / +0.038 | +0.106 (18/18) | +0.0069 (18/18) | +0.0211 | +0.0216 | +0.0216 |
| A10 | m3 | +0.475 (18/18) | +0.193 / +1.039 / +0.711 / +0.867 | +2.009 / +0.907 | +2.391 (18/18) | +0.086 (18/18) | +0.475 | +0.475 | +0.475 |

**Notes on the table.**
* Z2_k5 is in `i7.md`; it is dominated by Z2_k10 or Z1 everywhere.
* ΔC (containment) is 0 to ±0.002 for every rule. The rule acts through the accuracy norms, and through sps's accuracy
  factor, not through the interval.
* **"Share of the blanking loss recovered" is not a usable number.** It is the rule's composite on the masked stream
  relative to `control` on the plain stream. Under m2 and m3 the masked stream scores **higher** than the plain one
  (masked targets are the physical zero, and the wake region they replace is the hardest to predict), so the ratio
  changes sign. It is in `i7.md` for m1 only, where it reads 0.33-0.76. The meaningful measure is the S20 share of
  target-zero SE in the item 1 re-blank table: Z1 removes all of it (P = 0 = Y) except the 0.4 % of S20 elements whose
  target turns non-zero.

**OOD mean for the Stage C gate** (A20 and A10; no RE stand-in exists), plain streams: Z1 +0.00015, 34 of 34 positive.
The gate is +0.015.

## Pitfalls

**(i) Exact zero against the normalised zero.**
* The release stores u and v as float64. Across all 82 recordings (32 × 64 grid) there are 9.6 % exact zeros in u and
  10.8 % in v (15.4 M of them negative zeros, which compare equal to 0), and **~1 100 values in (0, 1e-12) per
  channel**. There are no values in [1e-12, 1e-8).
* The package sees the input normalised with the input statistics. The physical zero maps to −1.6007417 (u) and
  0.0322037 (v) in float32. A value of 1e-23 m/s is indistinguishable from it.
* So the package's test must be "|x_norm − z_c| < tol". I use **tol = 1e-6 in normalised units**, which is 9.7e-8 m/s for
  u and 1.6e-8 m/s for v. That is about 8 ulp for u and 270 ulp for v at the normalised zero, and it separates cleanly
  because nothing in the release lies between 1e-12 and 1e-8.
* The dumps compute the input state with exactly that physical test on the raw input the harness normalises. It differs
  from the exact test at 209-389 input elements per regime (the denormal-size values), recorded in
  `dump/*/meta.json`. The kit's target test is exact (`t != 0`), so those few are "zero" to the rule and "scored" to the
  kit.

**(ii) u and v separately.** Every state, class and rule is evaluated per channel. 1.4 % of elements are zero in only
one channel (chunk 2).

**(iii) mvpe probes** (rows 8-24 even, columns 13 / 21 / 29 / 37).
* Z1 touches probe elements on the plain streams (counted over window × 36 probe pixels × 2 channels × 20 leads): 5.3 %
  in ID (25 460 of 483 840), 2.9 % in A20 (27 080 of 925 920), 6.1 % in A10 (66 280 of 1 088 640). Its mvpe delta there
  is +0.000, because the members already predict about 0.
* Under m3 (which covers columns 21-37) the rule sets the probes to their target, 0. That is where mvpe gains +0.6 to
  +0.7.
* Z3_m1 costs mvpe −0.014 in distribution through transient zeros at the probes.

## Stage C sketch (conditional; not prepared as a build)

**`ZERO_PROP`**, a module-level constant, default `False` (bit-identical: no op is added to the graph when off).

**Track 2 (`Track_2/submit/submission.py::_step`, inside the captured graph):**
* Compute `zm = (|gx[..., c] − z_in_c| < 1e-6).all(frames)` per channel from `self.gx`, the window's own input. It has
  shape (1, 1, H, W, 2).
* Apply it at the very end of the step, after the containment lines (`gl = min(c − half, go)`,
  `gh = max(c + half_hi, go)`), and after `gC`, `gH`, `gC0` and `gVp` have been stored. D1c, VARCAL and R1 therefore keep
  their statistics on the unmodified fields.
* The three writes are: `go[..., :2] = where(zm, z_out, go)`, `gl = min(gl, z_out)`, `gh = max(gh, z_out)`, with
  `z_out = −MEAN_TGT/STD_TGT` (float32).
* It goes before the M3 guard tail, so the finiteness guards still run.

**Track 1 (`Track_1/submit_fno/submission.py::predict`, and `_fno.predict_fno`'s band path if the interval is built
there):** the same test, per window of the batch, on that window's own input rows. There is no pooling across windows.

**Files a build variant would touch:**
* both `submission.py`;
* `Track_1/submit_fno/_fno.py` (only if the Track-1 interval lives there);
* `build_round13.sh` (one `sed` line, `RPDE_SHIP_ZERO_PROP`);
* `build_final.sh` (only if it ships);
* a verifier in `Track_1/verify/`, because no local stream exercises the rule except a re-blanked one.

## Added to the prepared mirror runs

`ood_wave.sh` now gives the `<r>_mirror` runs the stage `i7`: frozen and control dumps, plain and m1-m3,
`ood_i7.py`, then the dumps are deleted. The full-size read of every rule comes with the wave at about +10 min per
regime.

## Verified, and not verified

**Verified**
* The dumps equal the real-path stream records exactly (max |diff| 0 in all 15 sets checked).
* The rules are applied to P and to the interval as the package's containment would apply them, and scored with the
  kit's decomposition (self-check 1e-8, chunk 1).
* Every number here has the sign of its mechanism. Z1 cannot lose except where an S20 element's target turns non-zero,
  which is 0.2-1.7 % by lead time. I did not trace which elements make the one negative in-distribution trajectory; its
  size was not extracted (the pooled in-distribution delta is +6e-5).
* The input-zero tolerance was checked against the whole release.

**Not verified**
* No rule has run inside the package: this is post hoc. The real-path read needs the knob.
* There is no RE stand-in, so the "OOD mean" is A20 and A10 only.
* The value of I7 depends entirely on whether the private recordings blank a region different from ours, which no local
  data measures. The masks are synthetic.
* Track 1 is read as the `frozen` Track-2 package, which matches the Track-1 package to 3e-6, not bit for bit.
