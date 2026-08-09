# Where the remaining score is — architecture directions assessed, 2026-08-09

Written after round 11 closed every non-model lever. This is a **brainstorm record**, not a
result: nothing here was implemented. Its job is to stop the next round re-deriving the same
reasoning, and to record *why* three specific architectures were rejected so they are not
re-proposed.

Companion files: `report_20260809_round11.md` (round-11 measurements), `plan_20260810.md`
(the actionable round-12 plan). Papers referenced live in `/home/clockj/RealPDE/Papers/`.

---

## 1. The bottleneck, from our own numbers

It is **not** capacity, data volume, ensembling, the band, or latency — all closed with
measurements in rounds 8–11. What is left:

> **We predict a conditional mean and bolt a hand-tuned variance onto it, but ~40% of the
> score is a distributional quantity.**

The supporting measurements, all ours:

| measurement | source |
|---|---|
| oracle sigma is worth **+7.71 final**; oracle time-mean +2.72; oracle fluctuation *shape* only +1.95 | `oracle_split_v9.py` |
| 68% of squared error sits in the fluctuation, but only **~31% of fluctuation energy is coherently predictable** | `diag_spectral.py` |
| `rms(div u)/rms(grad u)` is 0.42–0.49 on our PIV against 0.28–0.34 on CFD → large measurement-noise component | `divergence.py` |
| replacing the fluctuation *shape* cuts rel-L2 34% and buys +1.95; replacing sigma leaves rel-L2 untouched and buys +7.71 | round 9 |
| sigma's readout cannot move the training `var` whatever its shape or inputs; `lam_v` 0.6 fits log-sigma *better* (0.361 vs 0.377) and generalizes *worse* on TKE | round 11 |

**Which subscores are distributional.** `rel_l2` and `mvpe` are point scores — the conditional
mean is optimal for them. `tke` is a second moment. `sps` is literally an interval-coverage
score. At the mid per-point weights (rel 0.36, mvpe 0.32, tke 0.24, sps 0.23, time 0.08),
**tke + sps = 0.47 of 1.23, about 38% of the score, is distributional** — and they are our two
weakest subscores (79.15, and 39.6 against a ceiling of 71.03).

Our entire answer to that is `ALPHA` plus a per-pixel Gaussian sigma head trained on **log-MSE**.
Round 11's `lam_v` U-shape is the tell: log-MSE is a *misaligned surrogate*, not a hard ceiling
on sigma. Fitting it better does not predict the amplitude envelope better.

**Independent confirmation for this problem class.** [arXiv 2606.10848] forecasts experimentally
measured cylinder wakes from PIV with FCNs, U-Nets, FNOs and DMD, and concludes the models
"struggle to preserve transient flow features and **high-frequency energy content** when faced
with **noisy measurements and incomplete state observations**." That is our wall, found
independently on our exact problem class.

---

## 2. The bet: an FGN-style probabilistic model

**Paper:** "Skillful joint probabilistic weather forecasting from marginals", arXiv 2506.10772 —
the Functional Generative Network behind WeatherNext 2. Applied form in
`Papers/Track_1/s41586-026-10953-2_reference.pdf` (WeatherNext Cyclones, Nature 2026).

Mechanism:

* sample a **32-dim Gaussian noise vector** and feed it through parameter-shared **conditional
  normalization** layers to sample a new set of *weights* — perturbation in **parameter space**,
  not output space;
* train directly on the **CRPS of per-location marginals**, fair estimator;
* **one network call per sample** (GenCast's diffusion needs 40); 8x faster to sample despite
  being ~180M against ~57M parameters.

Why it fits this competition unusually well:

1. **CRPS is a proper scoring rule for what SPS measures.** SPS is an interval score, TKE a
   second moment. Replacing log-MSE-on-sigma with CRPS aligns training with ~38% of the score.
2. **Parameter-space perturbation yields spatially coherent samples from marginal-only
   training** — the paper's central claim. Output-space noise, which is what a per-pixel sigma
   head effectively assumes, produces the high-frequency artifacts they explicitly call out.
3. **One forward pass per sample.** This project has *already rejected diffusion* on exactly the
   grounds FGN was built to fix: PDE-Refiner scored 82.53 against its own MSE control's 83.83 at
   4x compute, and autoregressive diffusion is arithmetically dead here (`time_score` ~26).
   FGN is the generative family that survives a latency-scored task.
4. **`StackedFNO` already does the hard engineering.** K noise draws share weights, so K samples
   fuse into one invocation — the same machinery that put 8 members in 3 invocations.
5. It replaces **three** hand-tuned objects with one learned one: `ALPHA`, the sigma head, and
   the fitted band. Sample mean → `prediction` (best rel-L2/MVPE); sample spread → TKE; sample
   quantiles → the interval.
6. WN-C pairs FGN (aleatoric) with a **multi-model ensemble** (epistemic). Our 8 members already
   are the second half; we are missing the first.

**The tension to think through before building.** The scorer computes `tke` on `prediction`, so
the submitted array must be simultaneously close in L2 *and* correctly dispersed. A generative
model does not dissolve that trade — `ALPHA` is our current point on it. What FGN buys is a
**better-calibrated, spatially coherent sigma**, which is where the +7.71 lives.

Fluid-side precedent worth reading: conditional flow matching with SWAG for uncertainty-aware
near-wall turbulence [arXiv 2504.14485]; DDPM vs VAE vs GAN on a von Karman street including a
real PIV wake [arXiv 2411.16417].

---

## 3. Vortex tracking — yes, but not as a tracker

WN-C's most transferable lesson is that they **do not run a tracker in the loop**. They
"griddify" the sparse cyclone state onto the grid as two auxiliary tensors — an existence
probability field (Gaussian kernel at each centre) and scalar properties broadcast within discs,
masked elsewhere — and co-train dense field + sparse structure jointly, inverting the mapping
with a heuristic tracker only at inference. Their Extended Data Fig. 4 ablation shows a
two-stage "predict field, then track" pipeline does **not** get the transfer benefit: a ~10 MB
tabular track dataset measurably improves prediction on ~20 TB of gridded data.

Three versions for us, ranked by credibility:

**(a) Phase as a co-trained second modality — the true analogue.** WN-C's second modality is a
handful of physically meaningful scalars per storm. Ours would be **shedding phase, Strouhal
number, convection speed, wake width**, recoverable from the target trajectory by signal
processing, using no extra data. The wake is a periodic vortex street; our 20-frame input spans
~1/3 of a 60–75 frame shedding cycle, so phase is observable — and sigma, the amplitude
envelope, is precisely a function of where in the cycle the output window sits. This is the one
auxiliary target that attacks the +7.71 term directly.

**(b) A phase-aligned prediction frame.** Stronger and more speculative: estimate convection
velocity and phase from the input, warp to a co-moving frame, predict there, warp back. A
translating oscillating pattern becomes closer to stationary, so the network stops re-learning
advection at every phase. Nothing like this has been tried here.

**(c) A Q-criterion / lambda-2 auxiliary loss.** Weakest. Our auxiliary-loss record is poor —
divergence, vorticity, spectral and MVPE-probe terms were all null; only velocity-gradient
matching worked (+0.74), and Q is a quadratic invariant of grad(u) so `--lam-grad` already
captures part of it. In the literature Q shows up as a *diagnostic*, not as a loss shown to pay.

**Diagnostic before building any of it (no training required):** decompose the per-window error
into a phase/advection part and an amplitude part — find the spatial shift that minimizes the
error and see how much it recovers. If a small shift buys a lot, (a) and (b) are live. If not,
drop the branch for an afternoon's work.

---

## 4. Rejected: graph neural networks

GNNs earn their keep on **irregular meshes, varying geometry, sparse scattered sensors, or a
topology a grid distorts**. We have a fixed 32x64 uniform grid, one geometry, dense
observations. GraphCast and WN-C use an icosahedral mesh because *the Earth is a sphere and
lat-lon grids degenerate at the poles* — a pathology we do not have. The flow-reconstruction
GNN in the reading list [arXiv 2411.08764] is explicitly for *very sparse data in time-varying
geometries*.

Two further reasons specific to us: the capacity ladder runs **downward** (w16 81.865 < w24
82.046 > w48 81.451) at 0.9M parameters, and a graph-transformer processor would cost real
latency at 0.431 board-ms per local-ms.

**What is worth taking from GraphCast/WN-C is not the graph** — it is multi-scale processing
(which the FNO already provides spectrally) and the generative training in §2.

---

## 5. Rejected: KANO (Kolmogorov–Arnold Neural Operator)

**Paper:** `Papers/Track_1/2509.16825v6.pdf`, ICLR 2026. A dual-domain operator parameterized in
spectral *and* spatial bases with KAN-style trainable univariate edges and symbolic
interpretability. Thesis: FNO's hard-coded spectral sparsity is maladaptive for
position-dependent (variable-coefficient) PDEs, where FNO's parameter count can grow
super-exponentially. Headline: on synthetic 1-D operators (`G1 f = x^2 f - d2f/dx2` and
friends) KANO uses **152 parameters against FNO's 566k — 0.03%** — for an order lower l2, and
recovers closed-form Hamiltonian coefficients to four decimals.

Four reasons it does not transfer, roughly in order of how decisive they are:

1. **Its selling point relieves a constraint we do not have.** KANO's pitch is *parameter
   efficiency*. We are over-parameterized already: w16 < w24 > w48 > w64, and `--point-mlp`
   (+25k params in the trunk) cost **−0.379**. 0.9M parameters on 74 trajectories where ~69% of
   the fluctuation energy is incoherent. More expressivity per parameter solves nothing here.
2. **Its core move already failed in our tree.** KANO stops privileging the spectral basis and
   gives the spatial basis real power. The FNO block already has a spatial branch
   (`nn.Conv2d(width, width, 1)`), and round 10's `--point-mlp` did exactly that upgrade — the
   single worst architecture change of the round. Not a perfect refutation (KANO's spatial basis
   is position-*dependent*, `point` is shared) but directly on point.
3. **FLOPs are a scored subscore and the authors concede the cost.** Verbatim: KANO "requires
   orders heavier FLOPs compared to FNO due to the expensive double sum nature of its KN
   quantization", and "for operators that are nearly spectrally diagonal (e.g. standard FNO
   benchmarks), FNO is theoretically much faster than KANO when the two models are in similar
   size." Their compensation is asymptotic parameter efficiency, which per (1) we cannot cash.
4. **Its demonstrated failure mode is not ours.** KANO's generalization benchmark trains on one
   family of analytic functions (Periodic, Chirped Cosine, Sine Beats) and tests on a *disjoint*
   family (Sinc Pulse, Gaussian x Hermite, Wave Packet) — out-of-distribution extrapolation with
   a noise-free symbolic ground truth. We interpolate *within* our training distribution (all
   AoAs are represented) and our board-vs-proxy degradation is **8%**. There is no closed-form
   operator here to recover.

**The part of the premise that is fair, and the one cheap test.** Our domain genuinely is
position-dependent: a solid airfoil, a PIV field-of-view boundary, ~9.9% of elements with target
identically zero. A translation-equivariant spectral kernel is a poor basis for "there is a wall
here." Two things temper it: our lift already sees 2 coordinate channels through a 128-wide MLP
(`--lift-mlp`, one of only two changes that ever worked), so position-dependence *at the
boundary* exists; and the most obviously position-dependent region — the masked/zero-target
9.9% — is **dropped from SPS entirely** and is trivially learnable for the field metrics. The
position-dependence that is easiest to see is the least valuable.

If the hypothesis is worth testing rather than the architecture: (i) error maps by region
(body / shear layer / wake / freestream) — free, no training, and kills the branch if the error
is not concentrated near the geometry; then (ii) a per-location scale/bias on the trunk (spatial
FiLM), ~2048 x 24 x 2 ~ 98k params and one elementwise multiply-add, essentially free in FLOPs.
That is the minimal slice of "spatial basis" and the only genuinely untested piece — this network
has never had per-location parameters. Prior still negative: it is trunk capacity, and trunk
capacity has lost every time.

---

## 6. Summary

| direction | verdict |
|---|---|
| **FGN-style probabilistic model (CRPS + parameter-space noise)** | **The bet.** Targets the ~38% of the score that is distributional. |
| Phase/vortex as a co-trained second modality | Maybe. Cheap diagnostic first. |
| Second model **family** for ensemble diversity | Live but budget-constrained; see `plan_20260810.md`. |
| Graph neural network | **No.** Wrong inductive bias for a dense uniform grid. |
| KANO | **No.** Solves parameter efficiency (not our constraint), costs FLOPs (scored), benchmarked on noise-free 1-D symbolic recovery, nearest analogue lost 0.379. |

The pattern: GNN and KANO are both proposals about **operator expressivity**, and five rounds of
measurement say expressivity stopped being the binding constraint a while ago. What remains is
that the fluctuation is partly unpredictable and we model that with a log-MSE sigma head instead
of a proper scoring rule.

---

## References

* Alet et al., *Operational Tropical Cyclone Forecasting with AI*, Nature 2026 —
  `Papers/Track_1/s41586-026-10953-2_reference.pdf`
* *Skillful joint probabilistic weather forecasting from marginals* (FGN / WeatherNext 2) —
  https://arxiv.org/abs/2506.10772
* Lee et al., *KANO: Kolmogorov–Arnold Neural Operator*, ICLR 2026 —
  `Papers/Track_1/2509.16825v6.pdf`, https://arxiv.org/abs/2509.16825
* *Data-driven surrogate models for forecasting experimentally measured fluid flows* —
  https://arxiv.org/abs/2606.10848
* *Flow reconstruction in time-varying geometries using graph neural networks* —
  https://arxiv.org/abs/2411.08764
* *Conditional flow matching for generative modeling of near-wall turbulence* —
  https://arxiv.org/abs/2504.14485
* *Comparison of Generative Learning Methods for Turbulence Surrogates* —
  https://arxiv.org/abs/2411.16417
