#!/bin/bash
#SBATCH -J rp_k272_candref
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
# Adapted from final0927/jobs/rp_f27x_candref.sh: paired REAL-code-path read of JOINT_GAIN on the held-out twin package
# (F-sprint's t2twin_asnp.zip = 5 twins, shipped band, asnp constants), knob off vs on, static / asym R1 / asym R1 + D1c.
set -uo pipefail
W=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2xr; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
O=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/pkg/candref_k272; mkdir -p $O; T=/tmp/rp_k272_cand_$SLURM_JOB_ID
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
$PY - <<'PYEOF'
import sys, os; sys.argv=['x']; sys.path.insert(0, 'Track_1/verify'); import joint_gain as j
T = "/tmp/rp_k272_cand_" + os.environ["SLURM_JOB_ID"]
off = {"JOINT_GAIN": "False"}; on = {"JOINT_GAIN": "True"}
st = {"R1": "False", "R1_ASYM": "False"}; d1 = {"D1C_N0": "30.0"}
V = {}
for jn, jv in (("off", off), ("on", on)):
    V[f"static_{jn}"] = {"sub": {**jv, **st}, "fno": {"JOINT_GAIN": "False"}}
    V[f"asym_{jn}"] = {"sub": dict(jv), "fno": {"JOINT_GAIN": "False"}}
    V[f"asymd1c_{jn}"] = {"sub": {**jv, **d1}, "fno": {"JOINT_GAIN": "False"}}
j.stage("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/t2twin_asnp.zip", T, 2, V)
PYEOF
for v in static_off static_on asym_off asym_on asymd1c_off asymd1c_on; do
  echo "[STAGE $(date +%T)] stream twin $v"
  $PY Track_2/stream_eval.py --submission $T/t2_$v --save $O/$v.npz 2>&1 | grep -E "load:|errors|scores|final|saved"
done
$PY /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/jobs/rp_k272_table.py $O | tee $O/table.txt
rm -rf $T; echo "RP_K272_CANDREF_END $(date +%T)"
