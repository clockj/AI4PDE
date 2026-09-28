# Shipping-code audit (read-only) — 2026-09-27

Scope: `Track_1/submit_fno/`, `Track_1/sigblend/`, `Track_2/`. Ground truth for "final" = the files inside
`../sub_candidates/final/track1_k27fin_a080.zip` (T1, board 81.671149) and `.../T2_MRGP2_track2_MRGP2.zip` (T2, 82.055553).
Both archives carry the **same** `_fno.py` and `band.npz` (md5 `e8309d9f`, byte-identical to the untracked
`Track_1/bands/k27_sigblend_lam0.5.npz`: `base (1,20,32,64,3)`, `sigma_ref (1,1,32,64,3)`, `p = 0.225`, `centre_alpha = 0`,
unused `clip_lo/clip_hi`). The shipped `submission.py` files equal the working tree plus the sed-baked constants, plus a
5-line `FFNO_PATH` block inserted by `bake_ffno_path.py` after T1 `submission.py:124` / T2 `submission.py:104`.
**Line numbers:** T1 `submission.py` = working tree (shipped file = +5 after line 124); **T2 `submission.py` = the SHIPPED
file** (working tree = −5 after line 104); `_fno.py` is identical in tree and archives except the two sed'd constants.

---------------------------------------------------------------------------------------------------------------------

## 1. Knob table

L = live under the final constants, D = dead (unreachable, or a no-op), F = only a fallback runs it.
"Other setting" names the historical round given in the comments.

### Track 1 `Track_1/submit_fno/submission.py`

| knob | line | final | L/D | LOC guarded | other setting used by |
|---|---|---|---|---|---|
| `_TF32` (env `RPDE_TF32`) | 83 | False | L (TF32 off) | 3 | drivers only |
| `HOST_WRITE` → `_fno._HOSTW` | 110,123 | "torch" | **D**: the final path `continue`s before the only write that reads it (`_fno.py:2627-2637`) | ~40 incl. comments | R17+ "torch"; pre-R17 "" |
| `SPEC_PATH` → `_fno._SPEC` | 111,124 | "bmm" | **D**: read only by `StackedFNO._spec` (`_fno.py:497`); there are no FNO2d members | ~40 incl. comments | R17 T2 arm |
| `FFNO_PATH` → `_fno.FFNO_PATH` | shipped 125-129 | "dense" | L | spectral path of 4 F-FNO | T2R13 used dense via a scratch helper |
| `FUSE` | 122 | True | L | 2 | R23 False (invocation experiment) |
| `AMP` → `_AMP_DT` | 136-137 | "fp16" | L (behind `_amp_verified`) | ~25 | R24 None; R25+ fp16 |
| `GRAPH` / `GRAPH_FP` / `_Graphed` / `_capture_graph` | 156,216-218,405-533 | False | D | ~150 | R26/R27 True (+0.565 board ms, closed) |
| `GPU_BAND` | 162 | True | L | the whole final path | D2, 09-25 |
| `DIAG` | 166 | True | L (stdout only) | ~15 | — |
| `BATCH_FP` | 172 | (0,0) | D | ~15 | audit-2 fingerprint builds |
| `GPU_FP` | 179 | False | D | ~20 (+`_fno._PRED_FP`) | t1_curve 09-26 |
| `EBAND` | 185 | `(((0.24232559,0,0,0,0,0),(0.24535683,0,0,0,0,0)),'mult',None)` | L (logsig fast path only) | ~15 here, ~45 in `_fno` | K-sprint 09-27 |
| `WARM_TAILS` | 191 | True | L (import only) | ~20 | 09-26 |
| `LN_MANUAL` → `_fno.LN_MANUAL` | 194-195 | False | D | 8 in `_fno` | K1 09-27 arm |
| `BATCH_CAP` | 199 | 56 | L | 2 | 0 = uncapped (all rounds before 09-27) |
| `SMALL_GRAPH`, `SMALL_N`, `SMALL_GRAPH_LAZY` | 205-215 | False, 4, False | D | ~75 | L3 arms 09-26 |
| `_ERR` (from `band.npz` `err_mode`) | 221, 618-653 | None (band has no key) | D | ~45 | QD recipe (`reproduce.sh --recipe QD`) |
| `_BAND_LEGACY` `band.npy` | 222, 631-632 | absent | D | 3 | July packages |
| `ALPHA` | 252 | 0.80 | L | — | 0.60 (r25-r35), 0.90 (R13) |
| `CENTRE_ALPHA` (from npz) | 262, 615 | 0.0 | L (`centred` = True) | — | = ALPHA before 09-21 |
| `JOINT_GAIN` (+ `_fno.JOINT_GAIN`) | 265, 579-580 | True | L | assert + 5 | False = per-channel compose (all rounds ≤ 09-26) |
| `BLEND_BETA`, `BLEND_REF`, `TRAIN_SPEED` | 321-323, 330-347, 853 | 0.0 | D (also makes `use_gpu_band` possible) | ~80 | R14 0.15 |
| `PIN_OUT` | 715 | 128 | L (first 128 windows per process) | ~30 | K1 09-27 |
| host band path (`_predict` 824-917) | — | — | **D**: with `EBAND` set, `use_gpu_band` False raises (784) instead of falling here | ~95 | every round before D2 |

### Track 2 `Track_2/submit/submission.py`

| knob | line | final | L/D | LOC | other setting used by |
|---|---|---|---|---|---|
| `_TF32` | 78 | False | L | 3 | drivers |
| `SPEC_PATH` | 92,104 | "bmm" | **D** (no FNO2d) | ~12 | T2R5 arm |
| `FUSE` | 103 | True | L | 2 | R23 |
| `FFNO_PATH` | shipped 105-109 | "dense" | L | — | "fft" T2R13-era |
| `AMP` → `_AMP` | 119-122 | "fp16" | L (behind `_amp_verified`) | ~25 | None ≤ T2R11 |
| `ALPHA` | 138 | 0.80 | L | — | 0.60 |
| `_ERR` | 139, 925-932 | None | D | ~40 | QD |
| `PAR_STREAMS` | 145 | True | L | ~20 | 09-23 |
| `GRAPH_TOL`, `GRAPH_WORST`, `GRAPH_ERR` | 151-153 | 0.0 | L (capture acceptance) | 5 | N6 proposal >0 |
| `DIAG` | 157 | True | L | ~12 | — |
| `FALLBACK_FP` | 162 | 0.0 | D | 3 | audit-2 fingerprint |
| `STACKED_OUT` | 167 | False | D | 5 | Corin 09-26 arm |
| `ARENA_STEPS` | 173 | 1024 | L (CPU-input return, first 1024 steps) | ~15 | 0 before 09-27 |
| `NUMPY_OUT` | 177 | True | L (CPU input) | 1 | False before 09-26 |
| `CENTRE_ALPHA` (npz) | 180, 918 | 0.0 | L | — | = ALPHA before 09-21 |
| `JOINT_GAIN` | 183 | True | L | ~20 | False |
| `R1` | 195 | True | L | whole adaptive stack | False = frozen T2 (all rounds before the 09-25 R1 package) |
| `R1_KS` (9-pt grid 0.5..2) | 196 | — | L | — | — |
| `R1_SNAP` | 197 | False | D (`kfield` only) | 4 | legacy snap |
| `R1_N0` | 199 | 2.0 | L | — | — |
| `R1_ASYM` | 200 | True | L | ~50 | symmetric R1 (09-25 boards) |
| symmetric `update`/`update_gated`/`kfield` | 337-364, 563-566 | — | D under ASYM | ~35 | 09-25 boards |
| `D1C_N0` | 203 | 30.0 | L | ~30 | 0 before 09-27 |
| `SIGEMA` (+`_BETA`,`_N0`,`_DDOF`) | 207-217 | False | D | ~35 | never shipped |
| `SIGPOW` | 213 | (0.242325588, 0.245356822) | L | 8 | None before 09-27 |
| `VC_N0`, `VC_CLIP` | 218-222 | 30.0, (0.25,4) | L | ~25 | 0 |
| `FW_N0` | 223 | 0.0 | D | ~35 | never shipped |
| `R1_TSM` | 227 | 3 (any >1 means 3×3) | L | 1 | 1 |
| `R1_BLK_N0`, `R1_BLK` | 230, 235 | 20.0, 4 | L | ~20 | 0 |
| `R1_UNSCORED_TOL` | 236 | 1e-6 | L (see defect M1) | 1 | — |
| `R1_FP32` | 237 | True | L | 1 | fp64 = reference-exact |
| `D1C_EB`, `D1C_EB_N0` | 240, 245 | True, 1.0 | L | ~15 | False |
| `D1C_LEAD_N0` | 246 | 5.0 | L | ~6 | 0 |
| `HIST_BETA`, `BLEND_REF`, `TRAIN_SPEED` | 397-399 | 0.0 | D | ~55 incl. comments | never shipped |
| per-channel compose / `CENTRE_ALPHA == ALPHA` branch | 689-693, 730-735 | — | D | ~12 | ≤ 09-26 |

### `Track_1/submit_fno/_fno.py` (shared by both archives and ~25 repo tools)

| knob | line | final | L/D | LOC | note |
|---|---|---|---|---|---|
| `_PAD_BATCH` (env) | 40 | False | D | ~12 in `predict_fno` | env read in shipped file |
| `_AMP` (env) | 41 | None | D (callers pass `amp=`) | 1 | — |
| `_SPEC`, `_TAIL` (env), `_res_gelu` | 49-50, 58-74 | overridden / "" | D | StackedFNO only | — |
| `_HOSTW` | 55 | set to "torch" by T1 | D on final path | ~12 | — |
| `SWIN_IMPL` | 1001 | "auto" | D (no Swin) | ~600 (Swin family) | 09-14 screen |
| `LN_MANUAL` | 1327 | False | D | 8 | — |
| `FFNO_IMPL` | 1669 | "auto" | L: vendored copy in the archive, `models_alt` inside the repo (`_fno.py:1859-1861`) | — | — |
| `FFNO_PATH` | 1673 | set "dense" by both tracks | L | ~30 fft path dead | — |
| `FFNO_TILE` (env, default 16) | 1691 | 16 | L on T1 (b56 → 4 tiles of 14); T2 b1 untiled | 20 | env read in shipped file |
| `D2_DIRECT_PRED` | 2383 | **True** (sed) | L on T1 | 10 | False = D2 path |
| `JOINT_GAIN` | 2387 | **True** (sed) | L on T1 (T2 has its own compose) | 6 | — |
| `_PRED_FP` | 2390 | 0.0 | D | 6 | GPU_FP |
| `MEAN_SN`/`STD_SN`/`_self_norm_of`/`sn` branches | 104-127, 2469-2494 | — | D | ~40 | `--norm self` (closed) |
| Model families not shipped: `SpectralConv2d`/`FNO2d`/`StackedFNO` 129-569, `UNetTwoHead` 570-614, `UFNO2d` 615-677, `AFNO2d` 678-751, Swin 980-1326 + 1415-1659, `_fno_from_blob` 2148-2183 | | | D in the archive | **~1,450 of 2,645 lines** | all needed by repo tools / `reproduce.sh r25` |

---------------------------------------------------------------------------------------------------------------------

## 2. Live-path walkthrough

Common load (both tracks): `load_any` (`_fno.py:2292`) → `torch.load` → `dequant8` (2201: int8 × fp16 scale per
(out,in) pair; Linears one scale per tensor, `linear="tensor"`) → arch `"ffno"` → `_ffno_member` (1853, vendored FFNO2d
because `../unet/models_alt.py` is absent next to the archive) → `set_spectral_path("dense")`; arch `"vit"` → `ViT2d`
(heads = width//64 = 6). `fuse` (2069) groups by state-dict signature → `[StackedFFNO(4), StackedViT(2)]`, `n_members` 4 / 2.

**Forward of one group, member-averaged** (`StackedFFNO.forward` 2047): input `prev` = normalized `(B,40,32,64)` (u,v
frames, frame-major) + 2 grid channels → lift (1×1 conv 42→128, GELU, grouped 1×1 128→40 per member) → zero-pad (H+4, W+8)
→ 8 × `_StkFFNOLayer` (1967): dense separable spectral op (rfft-as-matmul basis, modes 16 × 32, real 2×2 block bmm, tiled
over 4 equal batch tiles), then `x + LN(FF(s))` with FF = Linear 40→80, GELU, 80→40 → crop → LN → heads
(1×1 128 hidden). Returns the member **mean** of the field and `log(mean_k exp(clamp(logstd_k,-8,2)))`.
`StackedViT.forward` (944): 4×4 patch embed, 8 pre-LN blocks (SDPA, 6 heads), same return contract. `predict_fno`
multiplies each group by `n_members` and divides by 6 → uniform 6-member mean `y` (normalized input space, `(B,20,H,W,2)`)
and `sig` = uniform mean of the 6 sigma heads (normalized, `(B,1,H,W,2)`).

### Track 1 — `predict(X)` with N in 14..99

1. `predict` (`submission.py:742`) → `_predict` (760). `use_gpu_band` (780) is True (cuda, band present, p=0.225,
   no graph/err/blend, `CENTRE_ALPHA == 0`). Input kept f32/f64, contiguous (791).
2. Outputs: `_pool_take(N)` (730) hands three consecutive slices of the 128-window pinned arena if they fit, else
   `np.zeros`/`np.empty` (792-798). `sig` host buffer `(N,1,32,64,2)`.
3. `_fno.predict_fno(..., batch=56, band_gpu=(base_dev, sigma_ref, 0.225, lower, upper, True, (beta, 'mult', None)))`
   (`_fno.py:2440`). Per chunk of ≤ 56 windows (2478):
   - upload, `[..., :2].float()`, `prev = ((x - MEAN)/STD)` permuted to `(B,40,32,64)` (2479, 2495);
   - groups under `torch.autocast(fp16)` (2519), cast back to f32, member average (2505-2538);
   - `c_dev = y*STD + MEAN` = **y0**, the pre-compose mean, physical (2552);
   - **compose, JOINT_GAIN** (2565-2573), per window/pixel, normalized space, population std over t:
     `m = mean_t y`, `s_c = std_t y_c`, `t_c = α·σ_c + (1-α)·s_c` (α = 0.8),
     `g = sqrt(Σ_c t_c² STD_c²) / (sqrt(Σ_c s_c² STD_c²) + 1e-7)`, `y ← m + (y - m)·g` (one gain for u and v; the
     per-channel physical variances are no longer each matched, only their sum — the TKE numerator);
   - `sigma_out = sig*STD` to host (2581); `yp = y*STD + MEAN` (2599);
   - **band** (2600-2626): `r3 = clip(σ_phys/σ_ref, 0.3, 3)^0.225` computed **on the host in numpy** (2605), uploaded;
     `half = base' · r3`; EBAND fast path (2610-2614, taken because features 1..4 are zero):
     `fac = exp(log(σ_phys + 1e-8)·b_c + 0)`, b = (0.24232559, 0.24535683) → `half_uv = half_uv · (σ_phys+1e-8)^b_c`,
     channel p half = 0;
   - `lower = min(y0 - half, P)`, `upper = max(y0 + half, P)` (channel p: 0/0/0), copied into `lower/upper` (2620-2626);
   - `D2_DIRECT_PRED`: `p3` (u, v, 0) copied straight into the output (2627-2637).
4. Return `{"prediction", "lower", "upper"}` (822); `predict` applies nothing else (`BATCH_FP` 0).

So T1's half-width is `half = base'_{t,y,x,c} · clip(σ/σ_ref, .3, 3)^{0.225} · (σ + 1e-8)^{0.2423 | 0.2454}` with
`base' = base^{0.5} κ_{t,c}^{0.5}` baked into the npz (SIGBLEND, λ = 0.5; `Track_1/sigblend/README.md`).
Import-time (untimed): `_load` (576) → `_amp_verified` (349; fp16 vs fp32 on 8 Gaussian windows, rel-L2 < 2e-2) →
`_pick_batch` (372; times 14/28/56/112 on a zeros probe **through the non-band path**) → `min(·, 56)` → `_pool_init`
(189 MB pinned) → `_warm_tails` (689; `predict` on 1..56 rows, f32 and f64).

### Track 2 — `ttt_step(input_norm, prev_target_norm)`, batch 1 (all inside one CUDA graph, `_body` 548)

Load: `get_ttt_model` (969) → members, fuse → `_load_band` (909) → `_amp_verified` per group (942) → `FrozenEnsemble`
(402): `half0 = base[...,:2]/STD_TGT` (target-normalized), pinned staging, 1.5 GB pinned arena, R1/D1c/VARCAL state →
`_capture` (749): 5 eager warm-ups on a side stream, capture, eager-vs-replay probe, accept iff max rel diff ≤ 0 → state
reset → 3 warm-up `ttt_step`s (prev None) → `arena_i = 0`.

Per step (`ttt_step` 827): copy input to `gx`; if `prev_target_norm` is given and a previous step exists in this
trajectory (`r1_prev`), copy it to `gyp`, `gflag = 1`, else `gflag = 0`; replay. `_body`, in order:

1. **R1 update + multipliers** (on `r1_stream`, 553-562). The previous window's `P = go`, `C = gC`, `h = gH` (all
   target-normalized, pre-multiplier half) are scored against `ŷ = gyp·A + B` (A = STD/STD_TGT, B = (MEAN-MEAN_TGT)/STD_TGT,
   i.e. the code assumes `gyp` is INPUT-normalized — see M1), `scored = |gyp - (-MEAN/STD)| ≥ 1e-6`.
   `update_gated_asym` (317): for the 9-point grid k_a, `lo_a = min(C - k_a h, P)`, `hi_b = max(C + k_b h, P)`,
   `A_a = exp(-(C-lo_a)·STD_TGT/σ_G)·[ŷ<C ? ŷ≥lo_a : 1]·scored`, `B_b = exp(-(hi_b-C)·STD_TGT/σ_G)·[ŷ<C ? 1 : ŷ≤hi_b]`,
   `cumG[a,b,g] += Σ_{t∈block g} A_a B_b · flag`, `cum2 = Σ_g cumG`, `n += flag` (σ_G = 0.0563870259; the product is the
   scorer's per-element `inside·exp(-(hi-lo)/σ)` without the per-window `(1-pm)` weight). `kfield_asym` (295):
   `(a*,b*) = argmax box3_{HW}(cum2)` (TSM), `k_lo = 1 + (box3(k[a*]) - 1)·n/(n+2)`, same for `k_hi`; BLK: per 4-frame
   block `(a_g,b_g) = argmax box3(cumG_g)`, `k_lo(t) = clamp(k_lo + (box3(k[a_g]) - k_lo)·n/(n+20), 0.5, 2)`. `n = 0` ⇒
   exactly 1. First max wins ties.
2. **D1c** (567-593): `ef = gC0 - ŷ` (previous window's uncorrected y0 in target-norm space minus the revealed target,
   **all pixels, no scored mask**), `e = mean_t ef`; `dsum += e·flag`, `dn += flag`, `dsq += e²·flag`,
   `dsumf += ef·flag`. `raw = dsum/max(n,1)`; EB: `se2 = mean_px(max(dsq - n·raw²,0)/max(n-1,1)/n)`,
   `τ² = max(mean_px(raw²) - se2, 0)` (per channel, pooled over all 32×64 pixels), `eb = (n≥2) ? τ²/(τ²+se2) : 0`,
   `eb *= n/(n+1)`; `shr = max(eb, n/(n+30))`; `b = raw·shr`; LEAD: `a(t) = Σ_px (dsumf/n)(t)·raw / Σ_px raw²`,
   `b(t) = b·clamp(1 + (a(t)-1)·n/(n+5), 0, 3)`; `gD = -b` `(1,20,H,W,2)`.
3. **VARCAL** (603-610): `cT += var_t(gyp)` (population, all pixels), `cP += gVp` (previous window's uncalibrated composed
   variance), `g = clamp(box3(cT)/box3(cP), .25, 4)`, `g = 1 + (g-1)·n/(n+30)`, `gG = sqrt(g)` per pixel **and channel**.
4. Forward (620-657) on two side streams, uniform 6-member mean `y`, `sig`; `y0 = y`.
5. **Compose, JOINT_GAIN + VARCAL** (665-688): `gj` as in T1; `gVp = (s·gj)²` (per channel); `y ← m + (y - m)·gj·gG`.
   Note `gG` is per channel, so the "one gain per pixel for both channels" property no longer holds once VARCAL moves.
6. Join R1 stream (698-699). `gC0 = y0·A + B`; **P** `= go[..,:2] = y·A + B + gD` (700-702).
7. **Band** (705-721): `half = half0 · clamp(σ_phys/σ_ref, .3, 3)^{0.225} · (σ_phys + 1e-8)^{q_c}` (SIGPOW, pow — T1
   uses exp·log, so the tracks agree to ~1 ulp, not bitwise); `gH = half`; `half_lo = half·k_lo(t)`, `half_hi = half·k_hi(t)`.
8. **Centre** (736-746): `c = y0·A + B + gD` (CENTRE_ALPHA 0); `gC = c`; `lower = min(c - half_lo, P)`,
   `upper = max(c + half_hi, P)`; channel p of go/gl/gh stays 0 (never written).
9. `_fetch` (870): CPU input → copy into arena slot `arena_i` (pinned, caller-owned), sync; after 1024 slots → pinned
   staging + `.clone()`; CUDA input → `.clone()` on the GPU. `NUMPY_OUT` → numpy views. Return
   `(pred, {"adapt_loss": None, "lower", "upper"})`.

`reset_ttt_state` (797) zeroes R1 tables/`n`, `gk/gk2 = 1`, `gflag = 0`, `r1_prev = False`, D1c sums, VARCAL sums, `gG = 1`.
`gC0/gVp/go/gC/gH` are not cleared; they are only read with `gflag = 1`, which needs a step after the reset — correct.

---------------------------------------------------------------------------------------------------------------------

## 3. Defects and risks in the live path

No BLOCKER found for the scored configuration. Nothing on T1 reads across windows (compose, band, EBAND are per window;
N only changes fp16 kernel choice and tiling). `lower ≤ upper` holds by construction on both tracks as long as values are
finite (`half ≥ 0` everywhere: base ≥ 0 checked at load, ratio clip positive, `(σ+1e-8)^q > 0`, k ∈ [0.5, 2]).

**MAJOR**

- **M1 — T2 revealed target is assumed INPUT-normalized; the official runner uses TARGET statistics.** Kit
  `official/t2_v6/.../local_eval.py:114` normalizes `tgt` with `mean_tgt/std_tgt`, and `:199,207` passes that as
  `prev_target`. Ours applies `yp·A + B` (T2 `submission.py:559,564,569`) and tests "unscored" against
  `r1_z = -MEAN/STD` (472-473). Numbers: the affine error is tiny (A-1 = -4.9e-5 / -1.9e-4, B = -1.8e-5 / +2.4e-4 in
  normalized units, ~1e-5 m/s). **The mask is the material part**: a physical-0 target arrives as
  `-MEAN_TGT/STD_TGT = (-1.600681, 0.032436)` against our `r1_z = (-1.600742, 0.032204)` — a gap of 6.1e-5 / 2.3e-4,
  60-230× `R1_UNSCORED_TOL`. So on the board **every blanked element counts as scored in the R1 table** (measured 8.9 % of
  u,v target elements are exactly 0 in `train_real`, ~5 % of pixels always-zero, the rest intermittent), and the 3×3
  TSM/box smoothing spreads that into neighbours. Local instruments hide it: `stream_eval.py:66`, `validate_t2.py:135` and
  `verify/pkg_identity.py` (which passes `prev=None`) all use `normalize_in`. Fix: in T2 `submission.py` replace `yp*self.Ad
  + self.Bd` by `yp` at 559, 564, 569, compute `r1_z` from `_MEAN_TGT/_STD_TGT` at 472-473, and in VARCAL (605) either
  accept the 1-A² ≈ 4e-4 scale error or divide by A². Add `t2_common.normalize_tgt` and use it at `stream_eval.py:66`,
  `validate_t2.py:135` (and `pkg_identity.run_t2` once it passes targets). Behaviour-changing: re-read on val_unseen
  twins with the fixed harness before shipping; the caller's -0.0015 `final` estimate is not verifiable from code.
- **M2 — D1c-EB and VARCAL statistics include unscored (blanked) pixels.** `ef` (569), the EB pooled `τ²` (581) and `cT`
  (605) use every pixel. At an always-blanked pixel `raw ≈ C` (≈ 1.6 normalized for u if the model does not predict 0 there),
  which dominates `mean_px(raw²)`, drives `eb → 1` from window 2, and applies the per-pixel shift at full strength
  everywhere. The comment (245-249) describes a bias-vs-scatter test; with blanking present it is mostly a blanking
  detector. VARCAL sees `var_t y = 0` at blanked pixels → `g` clipped to 0.25 and box-spread. These were the conditions the
  local gains were measured under, but a private recording with a different blanking pattern (new AoA → new shadow)
  changes the behaviour. Suggest: measure `eb` with and without a `scored` mask on the val_unseen stream before the
  Decision phase; if they differ, mask (same mask as M1, fixed).
- **M3 — No non-finite guard on either track; one NaN zeroes every subscore.** T1 `submission.py:910-914` deliberately
  skips the scan; T2 `_body` has none; T2 D1c/R1/VARCAL accumulate `gyp` with `·flag`, so a NaN in one revealed target
  poisons the rest of the trajectory. The fp16 probe (`_amp_verified`) runs on 8 in-range Gaussian windows only; the
  private set is OOD in Re (raw speed ∝ Re). Fix (bit-identical when finite): T1 in `predict_fno` band block, before the
  copies (2625-2636): `p3 = nan_to_num(p3, 0,0,0)`, `lo = where(isfinite(lo), lo, p3)`, `hi = where(isfinite(hi), hi, p3)`,
  `lo = minimum(lo, p3)`, `hi = maximum(hi, p3)`; T2 the same on `go/gl/gh` at the end of `_body`, and
  `gyp = nan_to_num(gyp)` + `flag *= isfinite(prev).all()` in `ttt_step` (846). ~4 kernels inside the graph.
- **M4 — T1 has no degraded path: without CUDA, `predict()` raises.** `submission.py:783-784`: `EBAND` requires the GPU
  band path, which requires `dev == cuda`. A container where `torch.cuda.is_available()` is False (driver/CUDA mismatch on
  a new cluster) crashes every call instead of running slow. T2 degrades to CPU correctly. Fix: in the host path
  (878-882) multiply `r3[..., :2]` by `np.exp(np.log(sig + 1e-8) * beta[:,0])` when EBAND is logsig-only, and drop the
  raise (or raise only for a non-logsig EBAND).
- **M5 — The final recipe is not in version control and is not what `reproduce.sh` builds.** `build_final.sh`,
  `Track_1/sigblend/`, `Track_1/bands/k27_sigblend_lam0.5.npz` and `r38_a080_c0_fill.npz` are untracked (`git status`);
  `.gitignore` has `**/*.npz`, so the bands need `git add -f` like `Track_1/bands/r24_xfam6.npz`. `REPRODUCE.md:10-11`
  still reads `FINAL_RECIPE = <TO BE DECIDED>`; `reproduce.sh` recipes A/QD/QD_STATIC build neither EBAND/SIGPOW nor the T2
  stack. `Track_1/sigblend/{make_k0_band.py:20,23, bandlab.py:17, dump_twins.py:9-14, rp_k27_final.sh:12-26}` hard-code
  scratch/home paths, and the json's `shipped_band` points at scratch. Irreproducible = disqualified.

**MINOR**

- **m1 — T2 `device` with an index ≠ current device.** `torch.cuda.Stream()` (479, 521) and `torch.cuda.synchronize()`
  (899, 903) act on the *current* device; with `device="cuda:3"` the arena copy is not waited for (host reads a slot
  before the D2H copy lands) and capture/streams target the wrong GPU. Fix: `torch.cuda.set_device(dev)` in
  `get_ttt_model` when an index is given, or `torch.cuda.synchronize(self.device)`. (The kit passes `"cpu"`,
  `local_eval.py:160`; the board evidently passes cuda.)
- **m2 — Pinned allocation.** T1 `_pool_init` (720) and T2 arena (448-453) are in try/except (fallback correct).
  T2 staging `self.host` (445-446, `pin_memory=cuda`) is **not**: a runtime that refuses page-locked memory makes
  `get_ttt_model` raise. Wrap it, fall back to pageable.
- **m3 — T2 graph acceptance is bit-exact (`GRAPH_TOL = 0.0`, 151).** On another GPU a replay-vs-eager ulp difference
  silently drops to eager (~3× step time; `FALLBACK_FP` 0 so the board cannot tell). Consider `GRAPH_TOL = 1e-5`. The
  eager path with `gflag = 1` (R1 on its side stream, record_stream only on group outputs, 636-640) is not exercised by any
  gate (`validate_t2`, `stream_eval` run the graph).
- **m4 — T2 step 1025+.** Arena exhausted → pinned staging + clone (901-907): correct, historical path, slower
  (~+0.2 ms). `arena_i` is process-global (not per trajectory). No gate reaches it (`validate_t2` streams 336 × 2 = 672
  steps); run `validate_t2.py --repeat 4` once. 1.5 GB pinned host memory is also a load-time assumption.
- **m5 — T1 `_pick_batch` (372-402) times the non-band path on a zero probe with 2 repetitions**, then `BATCH_CAP`
  clips to 56. On other hardware a noisy read can pick 14 or 28 (more chunks, different fp16 kernels/tiling ⇒ ~1e-4
  output drift, slower). Since the cap is fixed, restrict candidates to ≤ 56 and time the band path, or bake 56. Any
  exception returns 64 even after smaller sizes succeeded (401).
- **m6 — `PIN_OUT` helps only the first 128 windows of a process** (732); if the evaluator calls `predict` per
  trajectory (N 14-99), later calls allocate. Correct, just less benefit than measured.
- **m7 — Environment reads inside the shipped `_fno.py`**: `RPDE_PAD_BATCH`, `RPDE_AMP`, `RPDE_SPEC`, `RPDE_TAIL`,
  `RPDE_HOSTW` (40-55), `RPDE_FFNO_TILE` (1691, live), and `RPDE_TF32` in both `submission.py`. Violates "only baked
  constants ship"; harmless unless the organizers' environment sets them.
- **m8 — `FFNO_IMPL = "auto"` depends on the filesystem** (`_fno.py:1859`): if the organizers unpack the archive
  inside our repo tree (e.g. `Track_1/submit_fno/x/`), `../unet/models_alt.py` does not exist there, but a package run
  directly from `Track_1/submit_fno/` uses the training definition. Bake `"vendored"` in the archive.
- **m9 — T2 auto-reset.** State persists across trajectories unless `reset_ttt_state` is called; a runner that signals
  a new trajectory only by `prev_target_norm=None` would leak D1c/R1/VARCAL state across trajectories. Cheap,
  bit-identical under the kit protocol: in `ttt_step`, if `prev_target_norm is None and self.r1_prev`, call the reset.
- **m10 — Build hygiene.** `build_final.sh` greps the gates but its exit code ignores them (`FAIL` lines do not fail the
  script; `build_round13.sh` does not propagate gate codes). The T2 sed at `build_round13.sh:129`
  (`s/^R1_ASYM = .*/...`) also eats the trailing comment, leaving orphan comment lines in the shipped file (shipped 201-202).
  T1 EBAND exponents and T2 SIGPOW differ at 1e-9 (json float64 vs float32 source) — harmless, but the two tracks are not
  bit-identical in the band.
- **m11 — T2 DIAG** prints "first ttt_step" during the warm-up in `get_ttt_model`, not on the first real step (862-865).

---------------------------------------------------------------------------------------------------------------------

## 4. Duplication and structure

- **Compose** is implemented three times: `_fno.predict_fno` 2565-2575, T1 `_Graphed.body` 452-462 (dead), T2 `_body`
  665-693 (with VARCAL/SIGEMA variants). JOINT_GAIN epsilon 1e-7 vs per-channel 1e-6 is repeated in each.
- **Member loop + weighting** three times: `predict_fno` 2505-2540, `_Graphed.body` 431-449, T2 `_group`/`_body` 529-657.
- **Band loading**: T1 `_load` 610-653 vs T2 `_load_band` 909-939 (same err-head parse, `centre_alpha`, JOINT_GAIN
  assert, finiteness; different shape checks and fallbacks). **Band formula** twice (T1 `predict_fno` 2600-2618 +
  host path 878-894; T2 `_body` 705-721), with SIGBLEND as EBAND on T1 and SIGPOW on T2 — two constants for one number.
- **AMP verification** twice with different probes (T1 via `predict_fno` at batch 8; T2 per group at batch 1).
- **TF32 / SPEC_PATH / FUSE comment blocks** verbatim in both `submission.py` (~60 lines each).
- **Checkpoint discovery** `fno.pth, fno_2..16` in both; `TRAIN_SPEED/BLEND_REF` in both (both dead).
- **Packers**: `build_zip.py` and `Track_2/build_t2.py` duplicate `_linear_for`, the pack loop and the size cap.
- **Streaming loops**: `stream_eval.run`, `validate_t2.main` 133-170, `verify/pkg_identity.run_t2` — three copies, all
  with the M1 normalisation choice (pkg_identity additionally never reveals targets).
- **R1**: `r1_ref/causal_band_t2.py` (symmetric, fp64) vs the shipped class, which added ASYM/TSM/BLK — the reference no
  longer covers the shipped estimator. `_r1_box3` ≡ `_r1_boxn(·,0,1)`.
- **EBAND**: `band_feats.py` (numpy reference, 5 features) vs `_fno.eband_features` — only the logsig term ships.

Natural boundaries, keeping the 4-file archive: `_fno.py` = models + fusion + load + **shared math** (`member_mean`,
`compose(y, sig, alpha, joint, gain=None)`, `band_half(base, sig, sref, p, q)`), called by both `predict_fno` and T2
`_body` (plain tensor ops, capturable); per-track `submission.py` = contract, state, I/O only; a separate repo-only
`_fno_zoo.py` (or keep in `_fno.py`) for families that do not ship, so tools keep working.

---------------------------------------------------------------------------------------------------------------------

## 5. Dead files (nothing live uses them)

| file | evidence |
|---|---|
| `Track_1/submit_fno/build_round10.sh`, `build_round11.sh`, `build_round12.sh`, `finalize_round11.sh` | referenced only by each other, `plan/`/`report/` notes, `closed/README.md`, `Track_2/closed/t2_latency.py`, README "kept for provenance"; not in `reproduce.sh`/`REPRODUCE.md` |
| `Track_1/submit_fno/_package/` (tracked `_fno.py`, `submission.py`; untracked round-11 `fno*.pth`, `band.npz`) | round-11 staging; tracked despite `.gitignore:17 _package*/`; `build_zip.py:22` uses `_package` as its default `--stage` and would `rmtree` it; every live caller passes `--stage _package8` |
| `Track_1/submit_fno/band.npy` (tracked) | only `submission.py:222/631` legacy fallback when `band.npz` is absent (never in a package), `closed/calib_board.py`, `Track_1/submit/` |
| `Track_1/submit_fno/band.npz`, `band10.npz`, `submission.zip`, `submission10.zip`, `_package10/` (untracked, ignored) | August artifacts; no code reference except `build_zip.py` defaults and `verify/cpu_smoke_submission.py`'s old default |
| `Track_1/submit_fno/_package8/`, `submission8.zip`, `Track_2/_pkg/`, `Track_2/submission_t2.zip`, `Track_2/submit/band8.npz` | NOT dead: regenerated by every `build_round13.sh` run (by-products; `build_final.sh`'s restore does not cover them) |
| `Track_2/submit_ho/`, `Track_2/submit_ho5/` (tracked symlinks) | round-13 width-24 held-out twins; used only by `verify_space.py` default and README/EVALUATION examples |
| `Track_2/submit/fno*.pth` symlinks | round-13 all-data w24 members; dev copy, never packaged (build_t2 takes `--ckpt`) |
| `Track_2/eval/` | empty directory |
| `Track_2/t1_vs_t2_score.py`, `t2_lat17.py`, `step_breakdown.py` | instruments cited in CLAUDE.md/EVALUATION.md/README; not in any build/gate; candidates for `closed/` |
| `Track_2/r1_ref/` | verification reference cited by `REPRODUCE.md` and T2 comments; covers symmetric R1 only (see §4) — keep, mark partial |
| `Track_1/submit_fno/band_feats.py` | only its `const_literal` helper was ever used by the build; the final EBAND literal comes from the sigblend json |
| `__pycache__/` in `submit_fno`, `Track_2`, `submit`, `submit_ho*` | caches |

---------------------------------------------------------------------------------------------------------------------

## 6. Refactor proposal

**Tier 1 — moves/deletions, cannot change a package.** Commit first what M5 needs (`build_final.sh`, `sigblend/`,
the two bands with `git add -f`, the four uncommitted source files). Then: `git rm` `build_round1{0,1,2}.sh`,
`finalize_round11.sh`, `_package/` (tracked copies), `band.npy` → or `git mv` into `Track_1/submit_fno/closed/`; delete
untracked `band.npz`, `band10.npz`, `submission*.zip`, `_package10/`, `Track_2/eval/`, `__pycache__/`; move
`t1_vs_t2_score.py`, `t2_lat17.py`, `step_breakdown.py`, `submit_ho*/` into `Track_2/closed/` (update `verify_space.py`
default). Change `build_zip.py` defaults (`STAGE_DEFAULT`, `ZIP_DEFAULT`, `DEFAULT_CKPT`) to require explicit args.
Check: `bash Track_1/submit_fno/build_final.sh` with `RPDE_REF_T1/T2` → "ALL 9 FILES BYTE-IDENTICAL" on both.

**Tier 2 — non-shipped tooling.** Parameterize `sigblend/*` paths (env/args, no scratch defaults); add
`t2_common.normalize_tgt` and switch `stream_eval.py:66`, `validate_t2.py:135` (the M1 harness half — it changes local
numbers, not packages); make `pkg_identity.run_t2` reveal targets and run T1 at N ∈ {14, 56, 99} in successive calls
(exercises the pool, tails and `PIN_OUT` exhaustion); make `build_final.sh` fail on any `FAIL` or missing
`SUBMITTABLE`; share `_linear_for`/pack loop between `build_zip.py` and `build_t2.py`; fix the `R1_ASYM` sed pattern
(`[^ ]*` like its neighbours); write `REPRODUCE.md` FINAL_RECIPE = `build_final.sh` and add `reproduce.sh --recipe FINAL`.
None of these touches an archive (except the sed fix, which only changes a comment in the shipped T2 file — verify
by content diff, not md5).

**Tier 3 — inside shipped `submission.py` / `_fno.py`.**
Candidate cuts (all unreachable under the final constants, no data-dependent branch):
- T1: `GRAPH` (+`_Graphed`, `_capture_graph`), `SMALL_GRAPH*`, `BATCH_FP`/`GPU_FP`, `BLEND_*`, legacy `band.npy`,
  `LN_MANUAL` — ≈ 330 lines of 920. Keep the host band path and `_ERR` only if QD stays a supported recipe
  (`reproduce.sh --recipe QD` needs the err-head host path on T1 and the direct band on T2).
- T2: `SIGEMA*`, `FW_N0`, `HIST_BETA`, `STACKED_OUT`, `FALLBACK_FP`, `R1_SNAP` — ≈ 200 of 1004. Symmetric R1 and the
  per-channel compose are what A/QD/QD_STATIC ship; keep them if those recipes stay.
- `_fno.py`: **do not prune.** ~1,450 dead lines are inert in the archive, but the same file is imported by ~25 repo
  tools (`fit_band.py`, `aoa_holdout_eval.py`, `predict_board.py`, `qd_k/*`, `verify/fuse_*`, …) and by
  `reproduce.sh r25` (FNO w96), and `fit_band.py` drives `predict_fno`'s `err_out`/`centre_out` branches.

Verification recipe for any Tier-3 edit (all must hold): (1) rebuild with `build_final.sh` +
`RPDE_REF_T1/T2` — only `submission.py` may differ; (2) `Track_1/verify/pkg_identity.py --track t1 --tol 0` against the
extracted uploaded T1 archive at N = 14, 56, 99 and 336 (successive calls in one process); (3) T2 adaptive identity at
`--tol 0` with targets revealed (needs the Tier-2 pkg_identity change) plus `stream_eval.py --causal-check`;
(4) `validate_strict.py`; (5) `validate_t2.py --adaptive --repeat 4` (reaches the arena fallback);
(6) `verify/cpu_smoke_submission.py` on both archives; (7) run `reproduce.sh --recipe A` build step (`RPDE_REPRO_STAGE`
skipped, band reused) to confirm the knobs A relies on still exist.

**Recommendation.** Pruning the shipped files buys readability only; the score cannot improve and a re-created package
would no longer be byte-identical to the scored one. Before the Decision phase I would **not** prune `_fno.py`, and I
would spend the Tier-3 budget on the behaviour-preserving hardening instead: M3 (finite guards), M4 (T1 CPU path for
EBAND), m2 (T2 staging pin fallback), m1 (`set_device`), m9 (auto-reset) — each bit-identical on finite GPU inputs and
checkable with the recipe above. If a prune is wanted for the method release, the safest high-value subset is the T1
`GRAPH`/`SMALL_GRAPH`/`BATCH_FP`/`GPU_FP`/`BLEND`/`band.npy` blocks and the T2 `SIGEMA`/`FW`/`HIST`/`STACKED_OUT`/
`FALLBACK_FP` blocks: never shipped in a final, no recipe in `reproduce.sh` sets them, removal is mechanical, and the
identity check at tol 0 is conclusive. M1's package half is a behaviour change and belongs in a measured arm, not a
refactor.

---------------------------------------------------------------------------------------------------------------------

## 7. Doc drift (false under the final packages)

`Track_1/submit_fno/README.md`
- "As of R24 / R25 (xfam6)" table: members are now 4 F-FNO w40 L8 + sim (1.5 MB int8 each) + 2 ViTs, 36.3 MB extracted
  (not "4 w96 L8 FNOs … 29 MB each … 146 MB"); band is `k27_sigblend_lam0.5.npz`, not `r24_xfam6.npz`; `_fno.py` row
  omits `FFNO2d`/`StackedFFNO`.
- "What `predict()` does": missing JOINT_GAIN, the alpha-0 interval centre with min/max, SIGBLEND via EBAND, GPU-side band,
  direct write, `PIN_OUT`, `WARM_TAILS`, `BATCH_CAP`; step 5's band formula is incomplete.
- Baked-constants table: `ALPHA 0.60` → 0.80; AMP/GRAPH rows describe R24-R27; no rows for FFNO_PATH, EBAND, JOINT_GAIN,
  WARM_TAILS, PIN_OUT, BATCH_CAP, D2_DIRECT_PRED, SMALL_GRAPH.
- Band section: `--alpha 0.60`; the final band is a SIGBLEND transform of a `--centre-alpha 0 --fill-unscored` fit at 0.80.
- Build section: the one-call build is `build_final.sh`; "five stages … sed ALPHA, FUSE, AMP, SPEC_PATH" omits ~35 knobs.
- "`_pick_batch` chooses the batch size": then capped at 56.

`Track_2/README.md`
- "Conclusion: Track 2 is Track 1's model, streamed. Ship the forecaster frozen." and the closed-table rows "Online SPS
  band calibration" and "Residual EMA / output bias correction" — the final ships R1 (causal online band calibration),
  D1c (causal per-pixel bias removal) and VARCAL.
- "The package" bullets: no mention of R1/D1c/VARCAL/SIGPOW/arena/NUMPY_OUT; "`submit/` … band8.npz is whatever the last
  build wrote" (still true) but `band.npz → band8.npz` symlink now holds the SIGBLEND band after `build_final.sh`.
- Latency section describes StackedFNO(4) steps and T2R12; layout table lists `submit_ho*` as live.

Code comments
- T1 `submission.py:1-40` docstring (w24 FNO, `fno.pth`, `band.npy`, "86.68 local"); T2 `submission.py:1-36` ("Round T2-1.
  No test-time adaptation at all", "8-member group").
- T1 `submission.py:157-161` "else the host path below runs unchanged" — with EBAND set it raises (783-784).
- T1 `submission.py:633-637` "Nothing downstream can break it" — true only for a finite prediction (M3).
- `_fno.py:1-15` "Round 12 ships a MIXED ensemble"; `_fno.py:1672` and `StackedFFNO` docstring 1997-1999 "fft for Track 2's
  batch 1" — T2 ships dense; `_fno.py:2461-2464` "Bit-identical to the host path" — the EBAND factor has no host twin.
- T2 `submission.py:193` "Reference: … causal_band_t2.py (this class verbatim …)" — the class now has ASYM/TSM/BLK.
- T2 `submission.py:218-221` VARCAL: "vpred = (ALPHA·σ + (1-ALPHA)·s)²" — under JOINT_GAIN the code stores `(s·gj)²`
  (676-685) and the gain becomes per channel.
- T2 `submission.py:227` `R1_TSM` "3 = …" — any value > 1 means 3×3 (298).
- T2 `submission.py:82-92` SPEC_PATH rationale (einsum copy "27.6 % of a step") — inert, no FNO2d member.
- T2 `_fetch` docstring 870-887 "The clone is not optional" — the default CPU path is now the arena, no clone.
- T1 `_pick_batch` docstring "falls back to 64" — then clipped to 56.
- `REPRODUCE.md:10-11` FINAL_RECIPE "TO BE DECIDED"; CLAUDE.md *Current state* still names r34 / ALPHA 0.60 as the
  method of record and "Track 2 … Ship frozen" (outside scope, but it is what every session reads first).
