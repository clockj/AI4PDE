#!/bin/bash
# Local A100 step time of two built Track-2 packages, interleaved in ONE job (arm T1's price tag, not board latency evidence).
#
#   inside a GPU job:  bash ood_timing.sh PKG_A PKG_B OUT.json [ROUNDS]
#
# ROUNDS (default 4) x (A then B): each is stream_eval.py --traj <4 trajectories of VAL_TRAJ> --repeat 3 in its own process
# (one model object per process); the per-round figure is the median over its 3 reps of the mean step time. Reported: the
# median over rounds of each, and of the paired difference B - A with its min / max over rounds.
set -uo pipefail
A=${1:?PKG_A}; B=${2:?PKG_B}; OUT=${3:?OUT}; N=${4:-4}
PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
WT=${OOD_WT:?OOD_WT (a source tree)}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
TRAJ="3750_0 8850_20 16500_5 21600_15"
one() { $PY "$WT/Track_2/stream_eval.py" --submission "$1" --traj $TRAJ --repeat 3 2>&1 | tr -d '\000' \
          | awk '/rep [0-9]+: mean\/step/ {print $4}' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
ra=(); rb=()
for r in $(seq 1 "$N"); do
  ta=$(one "$A"); tb=$(one "$B")
  echo "round $r: A $ta ms  B $tb ms"
  ra+=("$ta"); rb+=("$tb")
done
$PY - "$OUT" "$A" "$B" "${ra[*]}" "${rb[*]}" "$(hostname)" "${SLURM_JOB_ID:-}" <<'EOF'
import json, sys, numpy as np
out, A, B, ra, rb, host, job = sys.argv[1:8]
a = np.array([float(x) for x in ra.split()]); b = np.array([float(x) for x in rb.split()]); d = b - a
r = dict(pkg_a=A, pkg_b=B, rounds_a=a.tolist(), rounds_b=b.tolist(), median_a=float(np.median(a)), median_b=float(np.median(b)),
         delta_median=float(np.median(d)), delta_min=float(d.min()), delta_max=float(d.max()), host=host, job=job)
json.dump(r, open(out, "w"), indent=1)
print(f"A {r['median_a']:.3f} ms  B {r['median_b']:.3f} ms  B - A {r['delta_median']:+.3f} ms [{r['delta_min']:+.3f}, {r['delta_max']:+.3f}]")
EOF
echo RP_DP_OOD_END
