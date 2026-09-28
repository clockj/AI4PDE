# EC final Track 2 packages (09-27, built on Ascend A100 from wt_c / wt_c_p2 @ 55f0e88 + design C patch [+ c2.patch + b2.patch])
Common: members ffno_all_w40l8_sim_s1..s4 + arch_vit_all_s1 + arch_vit_all_cmb3sim_s3 (r38), band = fallback band_r38_a080_c0_fill.npz (RPDE_BAND),
recipe A knobs (FORCE BAND_FILL CENTRE_ALPHA=0 INT8 LINEAR=tensor AMP fp16 SPEC_PATH=bmm HOST_WRITE=torch FFNO_PATH=dense), ALPHA 0.80,
T2 asnp (R1 True, R1_ASYM True, NUMPY_OUT True), FAMW off everywhere. Every package: validate_t2 --adaptive SUBMITTABLE; stream_eval --causal-check
CAUSALITY OK (176 steps, max|diff| 0); load line graph CAPTURED members 6. Knob-off identity (both wt_c and merged wt_c_p2): pred/lower/upper bit-equal
to fallback track2_x27asnp_d1c30_a080 on val_unseen. X3 QA stress tests: all pass (notes/qa_t2.md).

| package | sha256 | knobs on top of asnp | local ms/step vs asnp+D1c30 (interleaved) |
|---|---|---|---|
| T2_CONS_track2_c27cons_d1c30vc50.zip | d88c5d214b3ce967a1fbc332dfab7cf4a70934ba38bee807634559fafb43ad5c | D1C_N0 30, VC_N0 50 | +0.04 |
| T2_AGGR_track2_c27_d1c10vc50.zip | bad3f25a161d45811617868529d50cf16dd72653fc47fd3385d800aa194d2dae | D1C_N0 10, VC_N0 50 | +0.04 |
| T2_C2_track2_p2_c2.zip | 8d64f487073a0978562a0821bcc2f94852741b6fedd5f28fa3a9935fba63c411 | D1C_N0 30, D1C_EB True, D1C_EB_N0 1.0, D1C_LEAD_N0 5.0, VC_N0 30 | +0.18 |
| T2_P2_track2_p2_c2b2.zip | 8ca1a1f2e1b0ed99cc806daeb52afa8bd13197f7ab610b5cf2dfe19d9d6d6901 | C2 + R1_TSM 3, R1_BLK_N0 20.0 (R1_BLK 4) | +0.40 |

Twin / OOD reads, predictions: notes/result_c.md.
| **T2_MRGP2_track2_MRGP2.zip (UPLOAD)** | 1aa2896f7f98c3c775cba926c96a9972ea81dc66a3003090b34aaaf1a68f54bf | K-sprint merge (ARENA_STEPS 1024, PIN_OUT 128, WARM_TAILS, BATCH_CAP 56, JOINT_GAIN, SIGPOW (0.2423, 0.2454) + SIGBLEND band md5 e8309d9f, SMALL_GRAPH off) + C2 + B2 | 3.35 ms (P2 3.44, K cons 3.10) |
| T2_MRGP2nj_track2_MRGP2nj.zip | 9134b0ffa384166db5d18938ba83f46d844790bb61af56fa2274af68554993b0 | same without JOINT_GAIN | 3.33 ms |

MRGP2 tree: $S/wt_mrgp2 (= trick0927/wt_mrg f159924 + c2.patch + b2.patch, print line hand-merged). Build job rp_f27c_mb 7568186. Identity: knobs-off == K merged cons zip and == T2_P2 (bit-equal). Gates green, X3 GREEN. RSS after load 2.93 GB.
