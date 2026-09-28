# Fallback packages (X1, built on Ascend 09-27 16:14-16:17 EDT, job rp_f27x_build 7566572, worktree wt_x @ 55f0e88)

Local equivalents of today's planned packages (the other machine's 0d459a82 / db3dbe42 are NOT byte-identical: band refit here,
zip timestamps). Members: ffno_all_w40l8_sim_s1..s4 + arch_vit_all_s1 + arch_vit_all_cmb3sim_s3. Band: refit here,
`band_r38_a080_c0_fill.npz` (fit_band --ckpt A --err-ckpt [p2_ffno_l8_sim_s1..s3, arch_vit_d384, arch_vit_cmb3sim] --alpha 0.80
--band-p 0.45 --centre-alpha 0 --fill-unscored, inflate 1.39; md5 0cb37d8250efc4734348737fc4edecc4).

| file | sha256 | knobs | gates |
|---|---|---|---|
| track1_x27lz_a080.zip | 49abf18af422e4053b7716b585368964cd3fa4dca04660aa80ad20375653e3b8 | recipe A + ALPHA 0.80, centre0, fp16, int8 tensor, bmm, HOST_WRITE torch, FFNO dense, GPU_BAND True, D2_DIRECT_PRED True, SMALL_GRAPH True, SMALL_N 2, SMALL_GRAPH_LAZY True, BATCH_FP off | validate_strict SUBMITTABLE |
| track2_x27asnp_a080.zip | e01348168c2bc1be48e801245d23890f991f5fad40169dc8d719ba948080c55d | recipe A + ALPHA 0.80, centre0, fp16, int8, bmm, dense, R1 True, R1_ASYM True, NUMPY_OUT True, PAR_STREAMS True, graph CAPTURED, FALLBACK_FP 0 | validate_t2 SUBMITTABLE (plain and --adaptive); causal-check OK (176 steps, 0 diff) |
| track2_x27asnp_d1c30_a080.zip | 3e0d10d5e1478662efe79ca754ec610109d2eea7c7863ab90651d34983bc7cd6 | asnp + D1C_N0 30.0 | validate_t2 --adaptive SUBMITTABLE; causal-check OK; plain gate NOT SUBMITTABLE on "stateless" (expected: D1c changes P) |

Ambiguous knobs (chosen, see BUILD_RUNBOOK §5): BATCH_FP/GPU_FP/WARM_TAILS off on T1; FALLBACK_FP/GRAPH_TOL 0 on T2.

Local scores: the all-data packages on val_unseen (rel 0.0608 tke 0.3260 mvpe 0.0476, T2 sps 59.52) are CONTAMINATED.
Held-out twin reads of the same constants: `$S/notes/baseline_ref.md` §3 (asnp all-8 sps 55.594 local final 86.120;
+D1c 86.149; static 85.725). Latency numbers from these gates are invalid (RPDE_FORCE=1, shared nodes).
Build log: `build.log` here.
