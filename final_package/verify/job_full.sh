#!/bin/bash
#SBATCH -J rp_fin_full
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH --exclude=a0108
#SBATCH --output=/fs/scratch/PAS1187/qwu/tmp/realpde/logs/rp_fin_full_%j.log
# The full run the organizers will do, from the final image: solution.train('/data', '/output', 'cuda') at 400 epochs
# (two GPUs, 16 cores, 128 GB, /data in their layout without 7575_0, environment cleared), then the organizers'
# evaluation step on /output (eval_output.py) and the scores of the built archive against the archive of record.
#   SIF=<image.sif> TAG=<name> sbatch job_full.sh
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final1007
SIF=${SIF:-$S/img/t2_final.sif}
TAG=${TAG:-full}
MAIN=/users/PAS1187/qwu/workspace/realpde/main
KIT=/users/PAS1187/qwu/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6
W=$S/run_$TAG; mkdir -p "$W/out" || exit 1
for i in $(seq 1 90); do [ -f "$SIF" ] && grep -q "SIF_FINAL rc 0" "$S/logs/sif_final.log" 2>/dev/null && break; sleep 30; done
[ -f "$SIF" ] || { echo "no SIF $SIF"; exit 1; }
echo "node $(hostname)  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')  cores $(nproc)  sif $SIF"
sha256sum "$SIF" | cut -c1-16
EX="apptainer exec --nv --containall --cleanenv --pwd /workspace -B $S/orgdata:/data:ro"
nvidia-smi --query-gpu=index,timestamp,memory.used,utilization.gpu --format=csv -l 60 > "$W/gpu.csv" 2>&1 &
NVPID=$!
echo "=== TRAIN  $(date)"
TV=""; [ -x /usr/bin/time ] && TV="/usr/bin/time -v"
$TV $EX -B "$W/out:/output" "$SIF" python -c "import solution; solution.train('/data', '/output', 'cuda')" > "$W/train.log" 2>&1
echo "TRAIN rc $?  $(date)"
grep -a "^\[train\|Maximum resident\|Elapsed" "$W/train.log"
for f in "$W"/out/work/ckpt/*.log; do echo "$(basename "$f"): $(grep -a 'epoch   0\|epoch 399\|saved' "$f" | sed 's/  */ /g' | tr '\n' '|' | cut -c1-300)"; done
ls -la "$W/out"
echo "=== EVAL  $(date)"
$EX -B "$W/out:/output:ro" -B "$KIT:/kit:ro" -B "$S/evaldata:/evaldata:ro" -B "$MAIN/final_package/verify:/verify:ro" "$SIF" \
    python /verify/eval_output.py --kit /kit --data /evaldata --output /output 2>&1 | tee "$W/eval.log" | tail -5
echo "=== the archive of record on the same stream (reference)"
mkdir -p "$W/ref"; ( cd "$W/ref" && unzip -o -q "$S/ref/upload.zip" )
$EX -B "$W/ref:/output:ro" -B "$KIT:/kit:ro" -B "$S/evaldata:/evaldata:ro" -B "$MAIN/final_package/verify:/verify:ro" "$SIF" \
    python /verify/eval_output.py --kit /kit --data /evaldata --output /output 2>&1 | tee "$W/eval_ref.log" | tail -4
kill $NVPID 2>/dev/null
python3 - "$W/gpu.csv" <<'PE'
import sys, csv
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 4 and r[2].strip().split()[0].isdigit()]
mem = [int(r[2].strip().split()[0]) for r in rows]; util = [int(r[3].strip().split()[0]) for r in rows]
print(f"GPU over the job: peak memory {max(mem) if mem else 0} MiB per card, utilisation mean {sum(util)/max(1,len(util)):.0f}%")
PE
echo "JOB_FULL DONE  $(date)"
