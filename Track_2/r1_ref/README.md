# R1 reference implementations (Ramesh, 2026-09-25) — VERIFICATION TOOLS, not the method
- `causal_band.py`: numpy, float64. `CausalBandScaler(ks, n0, smooth, snap)`; shipping config = `KS` (K=9 log grid 0.5*2**(j/4)), `snap=False`.
- `causal_band_t2.py`: torch, float64, the same interface in T2's target-normalised space.
The METHOD is the fp32 torch implementation inlined in `Track_2/submit/submission.py` (`R1 = True`, `R1_KS`, `R1_SNAP = False`,
`R1_FP32 = True`). Expect it to differ from these float64 references on ~1e-5 of pixels in the argmax (09-25 CPU: 9/344064, max dk 0.013,
band numerator -8e-8); an fp64 build of the submission (`R1_FP32 = False`) matches `causal_band.py` bit for bit. CUDA fp32 summation
order differs from CPU, so the flipped pixels can change there. Their `self_test()` reads a twin export and `sps9.py` from our
workspaces (paths at the top of `_load` / `self_test`); point them at your own export to run them.
