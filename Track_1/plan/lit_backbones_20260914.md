# Literature triage — backbones, Track 2 adaptation, sim2real (scout, 2026-09-14)

**Inputs.** `pde_lttta.jsonl` (1087) + `pde_sim2real.jsonl` (354) = 1284 unique titles; keyword triage; 160 abstracts read. Web checks:
RealPDEBench paper and HF card, The Well code and README, OpenSTL code and model zoo, Poseidon config, DPOT/MPP/Walrus cards, PDE-Arena code,
2606.10848, 2407.14129. "(unv.)" marks a number I could not check.

**FLOPs.** Measured with `torch.utils.flop_counter` on CPU, B=1, one window (20 frames at 32×64). The shipped FNO w96 L8 is 0.525 GFLOPs
(FFTs uncounted) and defines 1×; the shipped ViT d384 L8 p4 is 4.9×. Candidates come from minimal re-implementations; "(a)" marks analytic
mult-adds. **FLOPs are not latency:** one ViT (4.9×) cost +0.316 A100 ms/window, i.e. +1.02 board ms on T1 (ROADMAP §1.25a), and
depthwise-conv and FFT ops cost more per FLOP than matmuls. Forge's interleaved timing decides.

## 0. Five findings that frame everything below
1. **Real, noisy, fixed-grid data favours local conv/window models; synthetic periodic NS favours FNO.**
   * RealPDEBench Foil (real PIV airfoil): U-Net first on every metric, incl. frequency error (2601.01829). PIV cylinder wake: U-Net lowest
     error on every metric except TKE magnitude, against FNO/FCN/DMD at ~60M params (2606.10848).
   * DLWP study: TFNO best on synthetic NS; ConvLSTM and Swin best on real WeatherBench short range (2407.14129). Fixed-grid nets beat
     grid-invariant operators (2410.07472). The Well: CNextU-Net best one-step on 8/17 (2412.00568).
2. **TKE loss comes from the loss function and appears in every architecture; it is not a backbone property.**
   * Every MSE-trained surrogate under-predicts planar TKE on PIV, worse with Re (2606.10848). MSE spectral blurring, with error shared across
     10 weather architectures (2604.01215); the conditional-mean barrier (2605.28076).
   * No paper in the set shows a backbone fixing TKE under an MSE-type loss: expect rel and containment; TKE only via better σ-head features.
3. **Everything saturates.** No DLWP backbone scales with params, data or compute (2407.14129). REALM: inductive bias, not parameter count,
   controls accuracy (2512.18595). Consistent with our closed capacity ladder.
4. **Pretrained PDE foundation models (FMs) are the wrong shape and size.** All are next-frame autoregressive at 128²: DPOT (10 frames), MPP
   (16), Poseidon (1 state + lead time), Walrus (1.3B). On Foil, fine-tuned DPOT-L (674M) trails a 23M U-Net by 11 % (2601.01829).
5. **Board priors the literature cannot override.** Partner families pay through containment (ViT +0.18, U-FNO +0.46 sps); conv families
   over-read locally (U-Net +0.09 proxy became +0.015 on the board; U-FNO missed by −0.45). Replicate, and project every candidate as a partner.

## A. Ranked backbone candidates
| # | candidate (ids) | role | should move | FLOPs × FNO | pure torch 2.2.2 | pretrained weights | closed-list conflict |
|---|---|---|---|---|---|---|---|
| 1 | **CNextU-Net** (2412.00568) | 3rd-family partner | sps (containment), tke, rel | 1.9 (f42, 3 st) · 0.6 (f32, 2 st, r2) · 4.3 (f64) | yes | none usable | none; conv over-reading risk (ROADMAP §1.20) |
| 2 | **Swin U-Net, scOT-style, from scratch** (2405.19101) | replaces the ViT pair | time, sps, rel | ≈0.7 (e48, 2-2-2, p2, w8, a) · ≈2.8 (e96, a) | yes (~150 lines, no `transformers`) | see #11 | "ViT closed at two" (§1.25b) bars a 3rd transformer, not a swap |
| 3 | **SimVP-v2 gSTA / TAU** (2211.12509, 2206.12126) | partner, or conv-lane fallback | rel, tke, sps | 16.8 (OpenSTL cfg) · 6.1 (mid) · 1.7 (slim) | yes | OpenSTL ckpts are video/weather: no use | TAU's inter-frame regulariser ≈ closed `--lam-t` family |
| 4 | PDE-Arena modern U-Net (2209.15616) | 2nd arm of the conv lane | as #1 | 3.0 (h32) · 11.6 (h64) | yes | none | redundant with #1 |
| 5 | F-FNO (2111.13802; Res-F-FNO yGdoTL9g18) | FNO-member replacement | rel, tke | ≈2.1 (w64 L12 r2, a) · ≈7.9 (w96 L12 r4, a) | yes | none | same family (no containment); capacity/modes past w96 L8 closed |
| 6 | Local+global branch inside one FNO member (2504.04260, 2603.26800, qTvmTCDRyD, 2605.09096) | FNO-member replacement | tke, rel | ≈1.2–2 (est., unv.) | yes | none | **this is U-FNO's idea** (closed as a package, −0.45) |
| 7 | Posterior-first Re/AoA inference + FiLM, add-on to any trunk (2605.03247; CAPE 22z1JIM6mwI) | add-on | rel, mvpe | ≈0 | yes | none | not closed; leak-free when inferred from window k only |
| 8 | DPOT-Ti/S with a one-shot 20-frame head (2403.03542) | pretrained partner | rel | ≈28 (Ti, 16×32 tokens, a) · ≈126 (S) | after vendoring timm bits | HF `hzk17/DPOT`, Apache-2.0, 7M/30M, 128², 10 frames, 4 ch, patch 8 | sim pretraining +0.00; latency |
| 9 | Divided space-time ViT (MPP 2310.02994; 2507.21244; 2412.17392) | partner | tke (hypothesis) | ≈17 (d128 L6, a) · ≈49 (d192 L8, a) | yes | MPP AViT, MIT, Google Drive, 16-frame history | ViT closed at two; latency |
| 10 | ConvLSTM / MinConvLSTM / MemNO (2407.14129, 2508.03614, 2409.02313) | partner | rel | ≈260 (OpenSTL ConvLSTM, 136 GFLOPs at 12→12) | yes | none | long context closed; latency |
| 11 | Poseidon-T/B fine-tune (2405.19101) | pretrained partner | rel | ≈2.5 per lead time at 128² (a, rough), ×20 lead times | port needed (HF `SwinForPDE`) | `camlab-ethz`, 21/158/629M, **CC BY-NC-SA 4.0** (paper) / cc-by-nc-4.0 (card) | NC licence vs winners open-sourcing; latency |
| 12 | Walrus (2511.15684; 2606.01470) | distillation teacher only | — | ≫100 | no (coupled to the_well) | HF `polymathic-ai/walrus`, MIT, 1.3B | not deployable |

**1. CNextU-Net.**
* *Mechanism.* ConvNeXt blocks in a U-Net with concatenated skips: full-resolution local fields plus a hierarchy, the inductive bias that
  FNO (8×12 retained modes) and the ViT (4×4 patches) both lack.
* *Evidence.* Foil real fine-tune: U-Net rel-L2 0.0145 vs FNO 0.0206, frequency error 14.6 vs 43.8 (2601.01829). PIV wake: skip connections
  preserve fluctuating energy (2606.10848). Samudra's ConvNeXt U-Net fixed "variance collapse" in Samudra 2 with a **wider U-Net and a smaller
  block expansion factor** (2412.03795, 2606.02610): the knob to try if σ collapses.
* *Copy* (The Well benchmark config, code-verified). DW 7×7 → LayerNorm → 1×1 (×4) → GELU → 1×1, layer scale 1e-6; down = LN + conv 2×2 s2;
  up = LN + convT 2×2 s2; concat skips; init dim 42; 2 blocks per stage; 1 bottleneck block (code defaults are 32 and 1 block).
  * At 32×64 use **3 stages** (4×8 bottleneck) or 2; 4 stages over-downsamples to 2×4.
  * Input 20×2 frames + 2 coordinates = 42 channels, stacked as The Well does; zero padding; shared σ MLP head; drop-path 0.1–0.2 plus
    bottleneck dropout.
* *Pitfalls.* Depthwise 7×7 is memory-bound, so the latency ratio will exceed the FLOPs ratio (unv.). ROADMAP §1.20: the local branch
  memorises in distribution, so replicate before advancing.

**2. Swin U-Net (scOT from scratch).**
* *Mechanism.* SwinV2 windowed attention, patch merging and expanding, ConvNeXt blocks on the skips.
* *Evidence.*
  * scOT from scratch beats FNO on almost all 15 Poseidon downstream tasks (2405.19101). Swin is second on synthetic NS and first-tier on real
    WeatherBench (2407.14129).
  * OpenSTL uv10 (wind u,v) at 32×64, 12→12: Swin 1.4996 vs gSTA 1.5069 vs ConvLSTM 1.8976 (2306.11249 model zoo). CTA-Swin-UNet beat FNO,
    LSTM and a plain Swin-UNet on wall-turbulence planes (2605.17888).
  * DRIFT-Net: window attention couples globally only through depth (qTvmTCDRyD), so keep the bottleneck a single global window.
* *Copy.* Poseidon-T `config.json`, verified: embed 48, depths [4,4,4,4], heads [3,6,12,24], patch 4, window 16 at 128², mlp 4, ConvNeXt
  skip blocks [2,2,2,0]. At 32×64: patch 2 (16×32 tokens), window 8, 3 stages (16×32 → 8×16 → 4×8, the last one global window),
  embed 48–96, depths 2-2-2 or 2-6-2; SwinV2 scaled-cosine attention, log-spaced continuous relative position bias, residual post-norm.
  Shifted windows **with** the mask: the domain is non-periodic.
* *Pitfalls.* Token grids must divide by the window. Pitch it as a ViT-pair replacement, since a third transformer is closed.

**3. SimVP-v2 (gSTA) / TAU.**
* *Mechanism.* Per-frame conv encoder, then a translator of gated large-kernel conv-attention blocks on the time-stacked latent (T·hid_S
  channels), then a per-frame decoder with a skip. One-shot multi-frame by construction.
* *Evidence.* OpenSTL WeatherBench 5.625° is exactly 32×64, 12→12. t2m MSE: **gSTA 1.105 (best of 16)**, Swin 1.143, ViT 1.146, TAU 1.162,
  ConvNeXt 1.277, ConvLSTM 1.521; uv10: gSTA ties Swin. Caveat: reanalysis fields are smoother than a wake.
* *Moves.* Per-frame decoding emits frame-distinct fluctuations instead of one 40-channel readout: a TKE mechanism, but unproven.
* *Copy* (OpenSTL code, verified).
  * Conv 3×3 + GroupNorm(2) + SiLU; PixelShuffle(2) upsampling; sampling [False, True]×(N_S/2); skip = first encoder layer added before the last decoder layer.
  * gSTA block: BN → 1×1 → GELU → [DW 5×5 → DW 7×7 dilation 3 → 1×1 to 2d, split, sigmoid gate] → 1×1, residual; MLP 1×1 → DW 3×3 → GELU →
    1×1; layer scale 1e-2; drop-path 0.1.
  * WeatherBench config: hid_S 32, hid_T 256, N_S 2, N_T 8, lr 5e-3 cosine, batch 16. TAU adds SE channel attention (reduction 16) and a
    divergence regulariser on frame differences (unv.).
* *Pitfalls.* **The OpenSTL config is 16.8× FNO at T=20**, because the full-resolution per-frame encoder and decoder dominate. Use N_S 4
  (8×16 latent), hid_S 16–32, hid_T 128–192, N_T 6, MLP ratio 2 (slim 1.7×, mid 6.1×). Swap BatchNorm for GroupNorm under fp16 and small batch.

**4. PDE-Arena U-Net** (code-verified). Pre-activation residual blocks: GN(8) → GELU → conv 3×3, twice, with a 1×1 shortcut. ch_mults (1,2,2,4),
2 blocks per level, down = conv 3×3 s2, up = convT 4×4 s2, concat skips, history flattened into channels. Use hidden 32 with ≤3 levels.
Run it only as the second arm if #1 passes.

**5. F-FNO.** Separable per-axis spectral mixing plus an improved residual, x + FF(·); deep, optional weight sharing. Evidence: −83 % NS, −57 %
airfoil, −31 % elasticity vs FNO/geo-FNO (2111.13802, abstract-verified); residuals re-inject small scales lost to mode truncation, −30 % on 3D
flow (yGdoTL9g18). Moves rel and tke by affording 16×32 modes. The per-layer FF dominates its cost, so it is not a latency saver vs our single
1×1. Same family, so no containment: judge it only as a member replacement against the FNO control, and pad the non-periodic axes before the FFT.

**6. Local+global branch in one member.** LOGLO-FNO: local spectral branch plus high-frequency propagation, half the params at FNO accuracy
(2504.04260). DSO: depthwise-separable conv + MLP-Mixer, −88 % (2603.26800). DRIFT-Net: −7–54 % vs scOT (qTvmTCDRyD). SpectraNet: −20 % vs
FNO (2605.09096). All synthetic and autoregressive. It keeps one invocation, but it is U-FNO in disguise: only worth an arm if the conv lane
pays as a partner and a fused single-member version is cheaper.

**7. Posterior-first conditioning.** An auxiliary head infers (Re, AoA) from window k's own input, supervised by the training labels, and
FiLM-conditions the trunk on that estimate. On metadata-hidden PDEBench tasks, rollout nRMSE 0.175 → 0.132, 59 % of the gap to oracle
(2605.03247). Run the oracle first: condition on the TRUE (Re, AoA) in held-out twins, and drop the idea if it gains < 2 % rel (mean u
already encodes Re).

**8–12, briefly.**
* **DPOT.** On Foil, DPOT gained nothing from sim pretraining (real-only 0.0159/0.0166 vs fine-tuned 0.0161/0.0174), and a one-shot head at
  16×32 tokens is ≈28× FNO. The round-9 report §5 reached the same verdict.
* **stViT.** At 20×128 tokens it costs 17–49×. Overtone: fixed patches accumulate harmonic artefacts (2507.09264). If kept: pool time to
  ≤5 tokens, dim ≤128, conv stem.
* **ConvLSTM.** Real-data strength, but ~260× FNO in FLOPs. MemNO's memory gain is measured against 1-frame Markovian baselines, and our
  20-frame input already is memory.
* **Poseidon.** Non-commercial licence, the `transformers` dependency is absent from the eval image, 4-channel (ρ,u,v,p) 128², and 20 lead
  times per window. Not recommended.
* **Walrus.** Fine-tuned on ≤3 DNS runs, it transferred zero-shot to laboratory Rayleigh-Taylor growth rates (2606.01470). As a teacher it
  would first need fine-tuning on our windows, which is outside 250 GPU-h (unv.).

**Wave-1 verdicts.**
| arm | literature | copy |
|---|---|---|
| `cnext_unet` | **supports**: strongest evidence for our regime (Foil, PIV wake, The Well) | #1; 2–3 stages at 32×64 |
| `swin_unet` | **supports** on real data (DLWP, OpenSTL uv10, scOT vs FNO) | #2; pitch as ViT-pair replacement |
| `simvp` | **supports** on setting match (32×64, multi-frame, one-shot), **only if slimmed** | #3; OpenSTL config is 16.8× |
| `unet_mod` | supports, but **redundant** with `cnext_unet`; h64 = 11.6× | #4; hidden 32, ≤3 levels |
| `ffno` | **undermines as a new family**: same family, costlier FF, closed axes | #5; member-replacement scoring only |
| `stvit` | **undermines**: 17–49× FLOPs, ViT family closed at two, patch artefacts | #9; drop, or pool time |

## B. RealPDEBench (2601.01829), Foil rows
Main table plus App. A; the 2- and 3-round autoregressive (AR) columns are from Table 5 via `report_20260807_round9.md` §5. Rel-L2 on the
real test set.

| model | sim-only | real-only | sim→real fine-tune (FT) | FT 2-round | FT 3-round | FT freq. err | update ratio |
|---|---|---|---|---|---|---|---|
| **U-Net 23M** | 0.0745 | **0.0159** | **0.0145** | **0.0191** | **0.0227** | **14.64** | 0.51 |
| DPOT-L-FT 674M | 0.0402 | **0.0159** | 0.0161 | 0.0209 | 0.0245 | 18.24 | 1.00 |
| DPOT-S-FT 41M | **0.0397** | 0.0166 | 0.0174 | 0.0218 | 0.0253 | 26.62 | 1.00 |
| WDNO | 0.0490 | 0.0444 | 0.0181 | 0.0232 | 0.0269 | 26.05 | 0.10 |
| FNO | 0.0540 | 0.0228 | 0.0206 | 0.0271 | 0.0314 | 43.83 | 0.32 |
| CNO 8M | 0.0438 | 0.0253 | 0.0206 | – | – | 57.35 | 0.44 |
| MWT 2.9M | 0.0492 | 0.0227 | 0.0210 | – | – | 25.10 | 0.49 |
| GK-Transformer | 0.0512 | 0.0245 | 0.0237 | – | – | 60.84 | 0.59 |
| Transolver 4.3M | 0.0465 | 0.0370 | 0.0237 (unv.: duplicates GK-T's row) | – | – | 22.92 | unv. |
| DeepONet 3.6M | 0.0565 | 0.0363 | 0.0375 | – | – | 47.96 | 0.52 |
| DMD | – | – | 0.0520 | – | – | – | – |

**Foil ranking** (FT rel-L2): U-Net < DPOT-L < DPOT-S < WDNO < FNO = CNO < MWT < GK-T ≈ Transolver < DeepONet < DMD. The paper: "convolution-based
methods (U-Net, CNO) achieve lower RMSE, as the tasks resemble image processing".
* **Sim→real protocol.** Three tasks: train on real; train on sim; pretrain on sim and fine-tune on real (update ratio = fine-tune iterations
  relative to real-only). Sim pretraining helps weak models most (WDNO −59 %, CNO −19 %), strong ones little (U-Net −9 %, FNO −10 %), and
  not DPOT (+1 %, +5 %): consistent with our +0.00.
* **Foundation models.** Only DPOT-S and DPOT-L (fine-tuned); no Poseidon or MPP rows.
* **Differences from our competition** (a number trained on their Foil does not predict our board).
  * 128×256 vs our 32×64; 3990 frames/trajectory vs 607; 98 real / 99 sim trajectories vs 82/100; parameter-level split 78 / 10
    in-distribution / 10 out-of-distribution (HF card); Re 2968–17031 (site/card) vs sim up to 27975 here; site text says NACA0025 (unv.) vs our NACA4418.
  * Autoregressive rounds 1/2/3 with rel-L2 ≈0.015, vs our one-shot 20-frame horizon at board rel error 0.108; input length and horizon not
    found (unv.). Data licence CC BY-NC 4.0; training on the public release is a rules question (round-9 §9).

## C. Track 2 / test-time adaptation
**What is new on Track 2.** Only the supervised pair and the realized error (`Track_2/README.md`). Closed ceilings: an oracle per-trajectory
additive correction removes only 1.75 % rel; gradient TTT gives −1.6 % single member / −0.7 % ensemble for 9.7 ms of backward per step.

**The Track 2 leader** is +0.138 over the Track 1 leader, while our own Track 2 is −0.180 below our Track 1: a ≈0.32 relative swing.
**Latency alone can explain it:** halving our 7.06 ms step moves `time` 91.04 → 93.49 = **+0.24 `final`** (time formula, w_time 0.0965).
No paper in the set reports a TTA gain of that size on a teacher-forced one-shot forecaster. The biggest Track-2-specific lever is per-step
latency, and adaptation must cost well under 1 ms to count.

| idea | why it could beat the closed null | free offline ceiling check | per-step cost |
|---|---|---|---|
| **C1 Meta-trained closed-form Bayesian last layer, with trajectory RLS** (ALPaCA 1807.08912; R2-D2 1805.08136; MetaKoopman 2607.26345; OS-ELM 2508.08012; DA reservoir readout 2504.16767; last-layer update laws 2402.00761) | exact Bayes shrinkage instead of one tiny Adam step; features trained to be adaptable; **no trunk backward**; statistics accumulate over the trajectory | refit the last readout layer per held-out trajectory, leave-one-window-out; continue only if > 3 % rel or tke error removed | ≈0.26 GFLOPs for 6 members (128² Gram × 2048 px + Cholesky), ≈0.5 FNO; ≈0.3–0.6 A100 ms (unv.) |
| **C2 Online member/family weights** (exponentially weighted forecaster on revealed per-member errors) | static weights are closed, the per-trajectory online version is unmeasured, and family advantage may track AoA | oracle per-trajectory weights from cached held-out member outputs | < 0.05 ms |
| **C3 Amortized context encoder** over (x_{k−1}, y_{k−1}, residual) → FiLM (M3GN 2511.05234; MaNGO 2510.05874; CoSFan Dl3MsjaIdp; FOCA AW0i0lOhzqJ) | meta-trained and forward-only; distinct from closed "history" only because it consumes the **residual** | as C1, on a 16-d FiLM | 0.1–0.5 ms encoder; FOCA-style gradient on z 1–3 ms |
| **C4 TTT layers carried across windows** (REE-TTT 2601.01605; TTT-Linear 2407.04620) | fast weights trained end-to-end to adapt | closed long context and history suggest ≈0 | forward-only, about linear-attention cost |
| **C5 Gated updates** (COLoKe 2511.12760; already suggested in ROUNDS.md) | skips adaptation when the last window's error is low | — | lowers the mean cost of C1/C3 |

**C1 design.**
* Cache the last-hidden readout features Φ (128-d per pixel, per member from the fused stack) at step k−1. W_post = (ΦᵀΦ + Λ₀)⁻¹(ΦᵀY + Λ₀W₀);
  accumulate ΦᵀΦ and ΦᵀY with forgetting; drop rows whose target is exactly 0; solve in fp32 with `torch.linalg.cholesky_ex` (no host sync,
  so it stays CUDA-graphable); return fresh tensors.
* Meta-train Λ₀ and the trunk with episodes (fit on window k−1 in closed form, predict window k, backprop through the solve). The first
  window of each trajectory uses W₀, so Track 1 is unchanged.
* Honest expected value: ≤ +0.05 unless the C1 ceiling check clears 3 %. The "last"-subset gradient arm in `ttt_grad.py` did not pay, but
  it was neither meta-trained nor cheap.

## D. Sim2real beyond the closed list
**Repo evidence first** (`simreal.py` header, `sim_map.py`). Sim's time-mean field is off by rel-L2 0.400, its std field by 1.457 (0.603 after
amplitude); fluctuation energy is 2.9× real at k≈1, 5.6× at k≈10, 0.55× at k≈35. That ratio is **non-monotone**, so a PIV low-pass emulation
filter is contra-indicated, and any residual-on-sim scheme inherits a 0.40 mean-field error. Every idea below is capped by sim's measured
total of ≈ +0.02.
* **D1 Spectral equalisation (Gaussian optimal transport).** Map sim fluctuations onto the real twin's radial spectrum with a per-condition
  gain g(k) = √(E_real(k)/E_sim(k)), non-monotone allowed; optimal transport between stationary Gaussian fields is exactly this gain.
  Precedents: ReMatch (optimal transport in PCA space, 2606.30821); POTT (3CiNHxWcBr); debias, then sample (5NxJuc0T1P). New relative to
  the closed scalar and per-pixel amplitude maps, which act in space, not frequency.
* **D2 Frozen sim expert as an input feature, not a residual base** (LGFNet delta learning 2603.29303; multi-fidelity DeepONet 2310.00057;
  2606.03469). It supplies "where the structure goes next", the one thing `simreal.py` says sim is informative about, at one FNO's FLOPs
  inside the fused group. First check that it beats persistence on mapped real inputs (the kit's sim-pretrained models: 69.0 vs persistence 79.7).
* **D3 Domain-invariant trunk.** Gradient-reversal domain classifier on trunk features during co-training (unsupervised domain adaptation in 2509.10380). Cheap; small effect.
* **D4 Fisher-projected fine-tuning** (PhysGuard 2606.16602). Worth it only if sim pretraining is reopened; it protects low-frequency
  directions (−32 % low-frequency error under severe shift), which maps to rel and mvpe, not tke.

## E. With 250 GPU-h
* **E1 Conv partner: CNextU-Net, with SimVP-slim as the fallback.** The strongest evidence for our regime (Foil, PIV wake, The Well), the
  lowest FLOPs (0.6–1.9×), and the inductive bias most unlike FNO+ViT, which should pay through containment, the channel the board has
  rewarded. Screen ~40 GPU-h (two sizes × two seeds, standalone and as a partner, band refit). Advance if the partner projection reaches
  ≥ +0.10 after +1 invocation (≈ −0.083 on T1) and its FLOPs. Then ~120 GPU-h of tuning (stages, width, drop-path, Samudra-2 expansion
  factor, sim co-training) and ~60 for all-data members and a fused/graph path.
* **E2 Swin U-Net as a ViT-pair replacement.** Target: keep the pair's +0.23 (+0.18, +0.05) at ≤ ⅓ of its FLOPs, which saves ≈1.4 board ms,
  ≈ +0.10 on T1. Screen e48/e96 × depths 2-2-2/2-6-2 as "FNO×4 + Swin×2" vs the shipped set, band refit.
* **E3 SimVP-v2 gSTA, slim.** The best 32×64 one-shot multi-frame evidence (OpenSTL), and the only arm that tests per-frame decoding for σ/TKE.
  If E1 passes, E3 competes for the same slot rather than adding a fourth family.
* **Track 2, single top idea: C1**, the meta-trained closed-form Bayesian last layer with trajectory RLS, costing ~0.5 ms/step instead of 9.7.
  Gate it on the ~1 GPU-h offline readout-refit oracle. Remember the arithmetic: per-step latency, not adaptation, is where ≈ +0.24 sits.

## F. Readout-refit gate result (job `rp_t2rr` 7281140, 2026-09-14) — **C1 CLOSED**
**Verdict.** The closed-form readout refit is far below the gate (>3 % error removed): no arm removes more than ≈1 % of any error, and none improves rel and tke together.

**What was run.**
* **Script:** `Track_2/readout_refit_check.py`, on held-out twin `fno_r18_cmb3sim` and its 3-twin ensemble, over the 8 `VAL_TRAJ` trajectories.
* **Data:** 336 windows streamed in Track-2 order. Tiling asserted; the first trajectory is bit-identical to `t2_common.build_stream`.
* **Method:** from window k's revealed pair, a ridge (Bayesian posterior mean) solve of the last 1×1 convs of `head_field` (129→40) and `head_logstd` (129→2),
  plus a per-channel affine control. Memory is window k only or every revealed window of the trajectory (RLS); ρ ∈ {1, 10, 100, 1000}.
* **Frozen path:** ALPHA 0.60, TF32 off; the frozen arm equals `predict_fno` exactly at batch 1.
* **How it differs from the closed nulls:** those were a feature-blind output correction and gradient steps on parameter subsets. This is an exact
  feature-space solve with no backward pass.

**Results.** Scored on windows k ≥ 2 (328).
* **Single twin** (frozen rel 0.0777 · tke 0.5035 · mvpe 0.0699). Best arm, `both/cum/ρ=1000`: pooled %rel −0.18, %tke +0.64, %mvpe +0.06; AoA 15–20: −0.19 / +0.53 / +0.03.
  * The largest rel cut, +1.02 % (`both/cum/ρ=1`), costs −3.6 % tke.
  * Window-k-only refits at ρ ≤ 10 hurt: rel −1.6 %, tke −4.5 %.
  * The affine control stays within ±0.1 %.
* **3-twin ensemble.** Best arm, `both/cum/ρ=1000`: −0.01 / +0.48 / +0.14 pooled; AoA 15–20: −0.13 / +0.39 / +0.04.
* **Best prior = the strongest (ρ = 1000).** Even with up to 41 in-trajectory windows, the trained readout is already near the trajectory optimum.

**Update cost** (A100, batch 1, eager fp32, median of 300 reps).
* Field 0.61 ms, σ 0.62 ms, both 1.07 ms per member; 3.13 ms for 3 members.
* That is 9× (both heads) to 16× (one head) cheaper than gradient TTT's 9.7 ms, yet still ≈ −0.07 / −0.20 `final` (T2 slope, one reading) against ≤ +0.015 of
  accuracy-only gain before the proxy halving. Net −0.06 (single) / −0.19 (ensemble).

**Consequence.**
* §C's C1 and §E's Track-2 pick are withdrawn; per-step latency is the only Track-2 lever with arithmetic behind it.
* C2 and C3 remain untested, but this result and the 1.75 % additive oracle both say the per-trajectory signal is small.
* **Artefacts:** `/fs/scratch/PAS1187/qwu/tmp/realpde/logs/rp_t2rr-7281140.out` and `.../work/t2rr_r18cmb3sim.json`; ≈0.03 GPU-h in total.
  The first attempt, 7281131, aborted after 23 s: its frozen-path guard read 2.1e-3 with cuDNN's default TF32 on.
