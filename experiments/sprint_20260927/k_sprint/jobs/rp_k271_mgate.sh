#!/bin/bash
#SBATCH -J rp_k271_mgate
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
W=$K/wt_mrg; cd $W; D=/tmp/e1mg_$SLURM_JOB_ID; mkdir -p $D
E=$K/e1/mgate_$SLURM_JOB_ID.jsonl
unzip -q $S/pkg/c_final/T2_CONS_track2_c27cons_d1c30vc50.zip -d $D/ref_cons
unzip -q $K/sub_candidates/track2_k27fin_d1c_a080.zip -d $D/ref_fin
for t in ${TAGS:-k27mrg_eqA k27mrg_eqB k27mrg_cons_a080 k27mrg_aggr_a080}; do unzip -q $K/sub_candidates/track2_$t.zip -d $D/$t; done
for v in ref_cons ${EQ:-k27mrg_eqA} ref_fin k27mrg_eqB; do [ -d $D/$v ] || continue
  echo "== eq $v"; $PY $K/e1/t2_edge.py $D/$v $v 336 $E 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-400; done
for t in ${GT:-k27mrg_cons_a080 k27mrg_aggr_a080}; do
  echo "== gates $t"
  $PY Track_2/validate_t2.py --adaptive --zip $K/sub_candidates/track2_$t.zip --extract $D/v_$t 2>&1 | grep -a "load:\|FAIL\|SUBMITTABLE\|errors\|scores" | cut -c1-80,200-560
  $PY Track_2/stream_eval.py --submission $D/v_$t --causal-check 2>&1 | grep -a "CAUSAL" | cut -c1-120
  $PY $K/e1/t2_edge.py $D/$t $t 1100 $E 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-500
done
echo RP_K271_MGATE_END
