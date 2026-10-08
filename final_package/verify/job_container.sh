#!/bin/bash
#SBATCH -J rp_fin_cont
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH --exclude=a0108
#SBATCH --output=/fs/scratch/PAS1187/qwu/tmp/realpde/logs/rp_fin_cont_%j.log
# The training package as the organizers run it (FINAL_SUBMISSION.md of the Final phase kit): the image built from
# the branch's Dockerfile (as a SIF), /data mounted read-only in their layout (train_real without 7575_0, train_sim,
# mean_std_real.pt), /output writable, 1 GPU, 16 cores, 128 GB, no network, environment cleared.
#   B  two networks trained alone, 6 epochs: the solo epoch time on this card
#   A  solution.train() at 6 epochs with RPDE_JOBS trainings at once per GPU: the whole pipeline, the epoch times
#   C  solution.get_ttt_model('/output') through the organizers' evaluation loop (eval_output.py)
#   D  inside the container, the band and the archive from the eleven checkpoints of record, against the upload
# RPDE_JOBS and CODE (bind the worktree over /workspace, or use the SIF's copy) are taken from the environment.
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final1007
SIF=$S/img/t2_test.sif
BR=/users/PAS1187/qwu/workspace/realpde/code-finalsub
MAIN=/users/PAS1187/qwu/workspace/realpde/main
KIT=/users/PAS1187/qwu/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6
JOBS=${RPDE_JOBS:-4}
TAG=${TAG:-asc}
W=$S/run_${TAG}_j$JOBS; rm -rf "$W"; mkdir -p "$W"/{outB,outA,ident} || exit 1
for i in $(seq 1 60); do [ -f "$SIF" ] && [ -f "$S/orgdata/.done" ] && grep -q "SIF rc 0" "$S/logs/sif.log" && break; sleep 60; done
[ -f "$SIF" ] || { echo "no SIF"; exit 1; }
echo "node $(hostname)  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  cores $(nproc)  jobs $JOBS"
echo "package: $(git -C "$BR" rev-parse --short HEAD) $(git -C "$BR" status --short | wc -l) uncommitted"
ls "$S/orgdata/train_real" | wc -l; ls "$S/orgdata/train_sim" | wc -l
BIND="-B $S/orgdata:/data:ro"
[ "${CODE:-bind}" = bind ] && BIND="$BIND -B $BR:/workspace:ro"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
EX="apptainer exec --nv --containall --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} --pwd /workspace $BIND"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$W/gpu.csv" 2>&1 &
NVPID=$!

echo "=== B: solo, 6 epochs, member_ffno_s1 then member_vit_c ===  $(date)"
$EX -B "$W/outB:/output" --env RPDE_SIMCACHE=/output/simcache "$SIF" python - <<'PE' 2>&1 | tee "$W/B.log" | grep -a "start\|done\|data:\|rc \|Error\|error"
import os, solution
solution.RECIPES = {k: solution.RECIPES[k] for k in ("member_ffno_s1", "member_vit_c")}
env = {**os.environ, "RPDE_REAL": "/data/train_real", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
       "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONDONTWRITEBYTECODE": "1"}
os.makedirs("/output/ck", exist_ok=True)
solution.run(["python", "-c", "from simreal import load_sim_mapped; load_sim_mapped('/data/train_sim', '/data/train_real', exclude=None, real_exclude=[])"],
             solution.UNET, "/output/simmap.log", env)
solution.say("map done")
solution.train_all("python", "/data/train_real", "/data/train_sim", "/output/ck", env, 6, 1, 16, ["0"])
PE
grep -a "epoch\|saved\|params" "$W"/outB/ck/*.log | sed 's#.*/ck/##'
date

echo "=== A: solution.train('/data', '/output', 'cuda') at 6 epochs, $JOBS at a time ===  $(date)"
TV=""; [ -x /usr/bin/time ] && TV="/usr/bin/time -v"
$TV $EX -B "$W/outA:/output" --env RPDE_EPOCHS=6 --env RPDE_JOBS=$JOBS "$SIF" \
    python -c "import solution; solution.train('/data', '/output', 'cuda')" > "$W/A.log" 2>&1
echo "A rc $?"
grep -a "^\[train\|Maximum resident\|Elapsed" "$W/A.log"
for f in "$W"/outA/work/ckpt/*.log; do echo "$(basename "$f"): $(grep -a 'epoch   0\|epoch   5\|saved' "$f" | sed 's/  */ /g' | tr '\n' '|')"; done
ls -la "$W/outA"; du -sh "$W/outA/work"
date

echo "=== C: get_ttt_model('/output') through the organizers' loop ===  $(date)"
$EX -B "$W/outA:/output:ro" -B "$KIT:/kit:ro" -B "$S/evaldata:/evaldata:ro" -B "$MAIN/final_package/verify:/verify:ro" "$SIF" \
    python /verify/eval_output.py --kit /kit --data /evaldata --output /output 2>&1 | tee "$W/C.log" | tail -6
date

echo "=== D: band and archive from the checkpoints of record, inside the container, against the upload ===  $(date)"
U=$MAIN/Track_1/unet
$EX -B "$W/ident:/output" -B "$U:/ref/ckpt:ro" -B "$S/ref:/ref/rec:ro" "$SIF" bash -c '
  set -u; cd /workspace
  export RPDE_REAL=/data/train_real TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1
  C=/ref/ckpt
  A="$C/ffno_all_w40l8_sim_s1.pth $C/ffno_all_w40l8_sim_s2.pth $C/ffno_all_w40l8_sim_s3.pth $C/ffno_all_w40l8_sim_s4.pth $C/arch_vit_all_s1.pth $C/arch_vit_all_cmb3sim_s3.pth"
  M="$C/p2_ffno_l8_sim_s1.pth $C/p2_ffno_l8_sim_s2.pth $C/p2_ffno_l8_sim_s3.pth $C/arch_vit_d384.pth $C/arch_vit_cmb3sim.pth"
  O=/output; mkdir -p $O/val $O/band $O/stage
  python band/make_val.py --data /data/train_real --out $O/val > $O/make_val.log 2>&1; echo "make_val rc $?"
  python band/fit_band.py --members $A --twins $M --val $O/val --out $O/band/static.npz > $O/fit_band.log 2>&1; echo "fit_band rc $?"
  python band/dump_twins.py --twins $M --val $O/val --out $O/band/dump.npz > $O/dump.log 2>&1; echo "dump_twins rc $?"
  python band/sigma_band.py --dump $O/band/dump.npz --band $O/band/static.npz --out $O/band > $O/sigma.log 2>&1; echo "sigma_band rc $?"
  grep -a "q (u, v)\|SIGPOW" $O/sigma.log
  python - <<PE
import numpy as np, json
a = np.load("$O/band/band_sigblend_lam0.5.npz"); b = np.load("/ref/rec/k27.npz")
for k in b.files:
    x, y = a[k], b[k]
    print(f"  band[{k}]: " + ("BIT-IDENTICAL" if x.shape == y.shape and np.array_equal(x, y) else f"differs, max |d| {np.abs(x - y).max():.3g}"))
print("  SIGPOW refit", json.load(open("$O/band/band_sigblend_lam0.5.json")).get("T2_sigpow"), "record", json.load(open("/ref/rec/k27.json")).get("T2_sigpow"))
PE
  python build/build.py --members $A --band /ref/rec/k27.npz --sigblend /ref/rec/k27.json --stage $O/stage --out $O/track2.zip > $O/build.log 2>&1; echo "build rc $?"
  python - <<PE
import zipfile, hashlib
h = lambda z: {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
a = h(zipfile.ZipFile("$O/track2.zip")); b = h(zipfile.ZipFile("/ref/rec/upload.zip"))
bad = [n for n in b if a.get(n) != b[n]] + [n for n in a if n not in b]
print("IDENTITY container build vs upload:", "ALL %d FILES BYTE-IDENTICAL" % len(b) if not bad else "DIFFER %s" % bad)
PE
' 2>&1 | tee "$W/D.log"
date
kill $NVPID 2>/dev/null
python3 - "$W/gpu.csv" <<'PE'
import sys, csv
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 3 and r[1].strip().split()[0].isdigit()]
mem = [int(r[1].strip().split()[0]) for r in rows]; util = [int(r[2].strip().split()[0]) for r in rows]
print(f"GPU over the job: peak memory {max(mem) if mem else 0} MiB, utilisation mean {sum(util)/max(1,len(util)):.0f}% max {max(util) if util else 0}%")
PE
echo "JOB_CONTAINER DONE"
