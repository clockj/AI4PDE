# Track 1 — archive

Packages, sweeps and job records from the first two months. Nothing here is imported by the live pipeline, a gate or
an instrument. Each item is kept because it records **how an experiment was run**; the verdicts are in
[`../../ROUNDS.md`](../../ROUNDS.md) and [`../../ROADMAP.md`](../../ROADMAP.md) Part 2.

**Paths.** These directories were moved here on 2026-09-27. `ROUNDS.md`, `../report/` and `../plan/` were written
before that and name them at their old places: `Track_1/submit/` is `submit_unet_0730/`, and `Track_1/submit_trick/`,
`Track_1/fno_deeponet/`, `Track_1/deeponet/`, `Track_1/diffusion/` are the directories of the same name here.
The shipped `submit_fno/submission.py` also says `../submit_trick/` in its docstring; it is left as uploaded.

**Runnable today?** Most scripts load the kit-v7 scorer by absolute path, and that kit was removed when v9 replaced it
(2026-08-05). They are records, not tools. The two that still run are marked.

| directory | what it asked | verdict | how it was run | runs today |
|---|---|---|---|---|
| `submit_unet_0730/` | The first package: a two-head U-Net (1.93 M), α 0.75, per-element band. | Board 83.108, then 83.019 (kit v7); 79.733 under kit v9 (R6). Its own in-sample gate read 87.956. Replaced by the FNO package on 08-05. | `build_zip.py`, `validate_submission.py`; `plan_submission_20260729.md` | no (v7) |
| `submit_trick/` | Does the top of the early board come from the windowing, not from a model? Window *k*'s target is window *k*+1's input. | **Yes: 93.500101.** Disclosed (`DISCLOSURE.md`); the organizers confirmed such scores are invalid. **Never ship it.** | `build_zip.py`, `validate_strict.py` | no (v7) |
| `fno_deeponet/` | Can an FNO or DeepONet with physics losses pass 92 locally? | The 0.90 M FNO beat the 50 M official one (85.94 vs 83.06) and velocity-gradient matching was the one physics loss that paid. FNO became the backbone (R7). | `run_phase*.sh`, `run_sweep.sh`; results in `*_results_*.txt`; `plan_20260802.md`, `report_20260802.md` | no (v7) |
| `deeponet/` | DeepONet as a baseline. | Plateau at 84.30 locally, MVPE 0.1345 against the FNO's 0.0768. Closed. | `deeponet_baseline.py`, `eval_deeponet.py` | `deeponet_baseline.py` trains; the evaluator is v7 |
| `diffusion/` | PDE-Refiner on the U-Net backbone. | 82.53 against 83.83 for the MSE control, at four times the compute. Closed. | `pde_refiner.py` | **yes** |
| `build_rounds/` | How the packages of board rounds R10 – R12 were built. | Superseded by `../submit_fno/build_round13.sh`. | `build_round10.sh` … `finalize_round11.sh`, from `Track_1/` | no (paths) |
| `unet_r9_r12/` | The training queue of local rounds 9 – 12 on the RTX 4060 and the first A100 session: the arm lists (`queue*.txt`, `done.txt`), the queue driver (`runq.sh`), the sim and context arms (`run_*.sh`), the first trainer (`train.py`) and 51 training logs. | Every arm has its verdict in `ROUNDS.md` rounds 8 – 13. The logs exist nowhere else. | `runq.sh` | no (paths) |
