# BUILD RUNBOOK — build + gate a package on Ascend (X1, 09-27)

Status: PROVEN on Ascend 09-27 16:14–16:19 EDT (job 7566572, rp_f27x_build: 2 builds, 4 gates, 2 causal checks, 6 stream reads in 290 s). Proven job script:
`/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/jobs/rp_f27x_build.sh` — **copy it, edit the member lists and knobs, submit.**

## 0. Rules that bite
1. **Run the build ONLY in your own worktree** (`$S/wt_<dir>`), never in the main repo. `build_round13.sh` does `cd` to the
   repo root of the script's own tree, `sed`s constants into `Track_1/submit_fno/submission.py`, `Track_2/submit/submission.py`,
   `Track_1/submit_fno/_fno.py` (DIRECT_PRED) and leaves them there. Restore afterwards:
   `git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py`
   Do not edit those files, and do not run two builds in the same worktree at the same time.
2. **Always pass `RPDE_SUB=$S/sub_candidates`** (absolute). Default is `../sub_candidates` relative to the worktree
   (= `$S/sub_candidates` for `$S/wt_*`, so it's fine either way, but be explicit). Zips land as
   `$RPDE_SUB/track1_<TAG>.zip` and `$RPDE_SUB/track2_<TAG>.zip`. **Use a unique TAG** — a same-tag build overwrites.
3. **`RPDE_FORCE=1` is mandatory today** — sprint `rp_f27*` jobs are running, and the script refuses otherwise.
   Consequence: the `time`/ms numbers printed by validate_strict / validate_t2 are **NOT valid latency measurements**.
4. Untracked inputs your worktree needs (symlink from main, read-only):
   `Track_1/unet/<ckpt>.pth` for every member/twin, and `Track_1/val_{seen,unseen}.npz` (fit_band / validate_strict / ens_v9 need them):
   ```bash
   M0=/users/PAS1187/qwu/workspace/realpde/main; W=$S/wt_<dir>
   for v in val_seen val_unseen; do ln -sf $M0/Track_1/$v.npz $W/Track_1/$v.npz; done
   for n in ffno_all_w40l8_sim_s1 ffno_all_w40l8_sim_s2 ffno_all_w40l8_sim_s3 ffno_all_w40l8_sim_s4 arch_vit_all_s1 \
            arch_vit_all_cmb3sim_s3 p2_ffno_l8_sim_s1 p2_ffno_l8_sim_s2 p2_ffno_l8_sim_s3 arch_vit_d384 arch_vit_cmb3sim; do
     ln -sf $M0/Track_1/unet/$n.pth $W/Track_1/unet/$n.pth; done
   ```
   New checkpoints: pass absolute paths in `RPDE_A` / `RPDE_M` (e.g. `$S/ckpt/<dir>/x.pth`) — the script only checks `[ -f ]`.
5. Hard-coded paths: scoring kits default to `/users/PAS1187/qwu/RealPDE/...` (read-only, exist here); `validate_t2` extracts
   to `/tmp/rp_t2_validate` (node-local) — pass `--extract /tmp/<unique>` when running it yourself. Nothing bulk lands in $HOME.
6. **A new architecture must load in `Track_1/submit_fno/_fno.py`** (`load_any`, and `fuse()` if you want one invocation);
   the zip is self-contained. Any `ens_v9`/`fit_band` read that works for your ckpt uses the same `_fno.load_any`.

## 1. Knob sets (reconstructed; see §5 for ambiguities)
Common (recipe A, REPRODUCE.md): 
```
RPDE_FORCE=1 RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"'
RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense            + ALPHA argument 0.80
```
T1 "27lz": `RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True`
(GPU_BAND=True is the file default = the D2 path).
T2 "asnp": `RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True` (PAR_STREAMS=True dual stream and the
CUDA graph are file defaults). "asnp + D1c": add `RPDE_SHIP_D1C_N0=30.0`.
One build produces BOTH tracks; T1 knobs only touch T1 files and T2 knobs only T2 files (except DIRECT_PRED, which is in the
shared `_fno.py`; harmless for T2).

## 2. Copy-paste: build + gate (inside an sbatch on Ascend; see the job script for the header)
```bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; W=$S/wt_<dir>; cd $W
export HDF5_USE_FILE_LOCKING=FALSE
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
U=Track_1/unet
A="<4-6 shipped all-data members, space-separated, e.g. $U/ffno_all_w40l8_sim_s1.pth ... $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth>"
M="<held-out twins of the SAME recipes, e.g. $U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth>"
COMMON=(RPDE_FORCE=1 RPDE_SUB=$S/sub_candidates RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor
        'RPDE_SHIP_AMP="fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)        # + RPDE_SHIP_D1C_N0=30.0 for asnp+D1c
TAG=<unique_tag>
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$TAG \
    bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee $S/logs/build_$TAG.log
cp Track_2/submit/band8.npz $S/pkg/<dir>/band_$TAG.npz          # keep the band (reuse with RPDE_BAND=...)
git checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py
# T2 gate for an ADAPTIVE package (R1 / D1c / SIGEMA): the build's own validate_t2 runs WITHOUT --adaptive. Observed:
# asym R1 alone still PASSES stateless (R1 moves only the band?) -> SUBMITTABLE; with D1c it prints
# "[FAIL] stateless ... NOT SUBMITTABLE" -- EXPECTED (D1c changes P by design). The verdict for ANY adaptive package is these two:
$PY Track_2/validate_t2.py --adaptive --zip $S/sub_candidates/track2_$TAG.zip --extract /tmp/v_$TAG
$PY Track_2/stream_eval.py --submission /tmp/v_$TAG --causal-check            # must print "CAUSALITY OK"
sha256sum $S/sub_candidates/track{1,2}_$TAG.zip
```
* **Reuse a band** (members unchanged, only constants changed; skips the ~fit): add `RPDE_BAND=<path/band.npz>`.
  Refuses together with `RPDE_ERR_*`. **Changed members ⇒ never reuse; refit** (leave `RPDE_BAND` unset).
* Only one track needed? The script always builds both; ignore the other zip.
* Submit: `DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)`
  `sbatch ${DESK:+--exclude="$DESK"} --chdir=$W --output=$S/logs/rp_f27<dir>_build_%j.log <script>`
  (header: `-A PAS1186 -M ascend -p nextgen --gpus-per-node=1 -c 12 -t 00:40:00`).

## 3. Reading the output
* T1: `validate_strict.py` ends with a line `SUBMITTABLE` (else a list of failed checks). `grep -n "SUBMITTABLE" <log>`.
  It also prints the rel/tke/mvpe/sps of the ALL-DATA package on val_unseen — **contaminated (members saw it), not evidence.**
* T2: build-internal `validate_t2` → `NOT SUBMITTABLE: stateless ...` for adaptive packages (expected, ignore);
  the `--adaptive` re-run must print `SUBMITTABLE`; `stream_eval --causal-check` must print `CAUSALITY OK`.
* Band fit prints `sps board proxy (val_unseen AoA 15-20 ...)` — contaminated when A are all-data (it says "NOT honest").
* `grep -H "^ALPHA = \|^AMP = \|^R1"...` lines in the log show the baked constants — check them.

## 4. Stage wall times (A100 nextgen, job rp_f27x_build, 6 members, int8)
| stage | wall |
|---|---|
| queue wait (nextgen, 09-27 16:14) | ~20 s |
| band fit (fit_band, refit) | 33 s |
| T1 package + validate_strict | 16 s |
| T2 package + validate_t2 (in-build) | 25 s |
| **whole build_round13.sh, band refit** | **77 s** |
| whole build_round13.sh, `RPDE_BAND` reused | 35 s |
| validate_t2 --adaptive | 17 s |
| stream_eval --causal-check (+ full stream score) | 22 s |
| stream_eval one read (8 traj) | 15–17 s |
So a new package is built and gated in < 5 min of GPU time; the 25-min budget is dominated by queue + your edits.

## 5. Ambiguities (knob values the docs do not pin down)
* T1 27lz `BATCH_FP` / `GPU_FP` / `WARM_TAILS`: the 09-26 "D" package carried a batch fingerprint (board mvpe moved), 27lz
  is described as "same compute path as D + lazy N=1 graph". The fingerprint value is not in the repo. **Chosen: off**
  (file defaults `(0.0, 0.0)`, False, False). Outputs at N>=100 are identical either way.
* `SMALL_N`: commit d30c0b2 says "SMALL_N 2 = batch-1 graph for N=1 only"; report says 27lz graphs at N=1. **Chosen 2.**
* T2 `FALLBACK_FP` / `GRAPH_TOL`: not stated for asnp. **Chosen: file defaults (0.0).**
* D1c value: `D1C_N0 = 30` (report 4.6 "n/(n+30)").
* The other machine's zips (0d459a82 / db3dbe42) are NOT byte-reproducible here (band refit, zip timestamps); ours are
  functional equivalents.

## 6. Results of the proof run (fallbacks: `$S/pkg/fallback/README.md`)
* track1_x27_a080 (27lz): validate_strict SUBMITTABLE; `[rpde-t1] load: ... gpu_band True direct_pred True alpha 0.8 centre_alpha 0.0
  small_graph lazy members 6`.
* track2_x27_a080 (asnp): in-build validate_t2 SUBMITTABLE; --adaptive SUBMITTABLE; CAUSALITY OK (176 steps, max|diff| 0).
* track2_x27_d1c_a080 (asnp+D1c30): in-build NOT SUBMITTABLE (stateless, expected); --adaptive SUBMITTABLE; CAUSALITY OK.
* Check the `[rpde-t2] load:` line: it prints `r1 / r1_asym / d1c_n0 / sigema / alpha / graph CAPTURED / numpy_out / members`.
  `graph CAPTURED` must appear; `members N` must equal your member count.

## 7. Traps hit / to know
1. `REFUSING: other realpde Slurm jobs are RUNNING` is printed even with RPDE_FORCE=1 — it continues; not an error.
2. The ALL-DATA package's validate scores (rel 0.0608 / sps ~59) are contaminated — never quote as evidence.
3. T1 `time` (1.47 ms/window) and T2 ms/step here are NOT latency evidence (FORCE, shared nodes).
4. The in-build T2 gate uses a fixed `/tmp/rp_t2_validate` extract dir — two builds on one node at once would collide;
   run your own --adaptive gate with a unique `--extract`.
5. A second build in the same worktree reuses `Track_2/submit/band8.npz`: always `cp` the band out right after the build.
6. Unique `RPDE_TAG` per build, or the zip in `$S/sub_candidates` is overwritten silently (the script also leaves
   `Track_1/submit_fno/submission8.zip` and `Track_2/submission_t2.zip` in the worktree — ignore those).
7. The `sed` knobs are literal Python: strings need quotes (`RPDE_SHIP_AMP='"fp16"'`), booleans `True`/`False`, floats `30.0`.
8. To turn a baked constant OFF in an already-built package for a quick read, unzip it to /tmp and `sed` the extracted
   `submission.py` (that is how the twin stream reads in §baseline were made) — never ship a hand-edited zip.

## 8. sha256
`sha256sum $S/sub_candidates/track{1,2}_<TAG>.zip` (first 8 hex chars are what the notes quote).

## 9. Paired held-out read of a candidate (before building)
`$S/jobs/rp_f27x_candref.sh` — one job, ~4 min: ens_v9 (AoA 15–20 + full, f 1.39/1.29), band fit with the shipped
construction, twin T2 package with the asnp constants, stream reads static / asym R1 / asym R1 + D1c. Usage and the baseline
numbers: `$S/notes/baseline_ref.md` §4. It does NOT sed your worktree, but it reads `Track_2/submit/*` there, so do not run it
while a build_round13.sh runs in the same worktree.
