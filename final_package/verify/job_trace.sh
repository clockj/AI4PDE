#!/bin/bash
#SBATCH -J rp_fin_trace
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 02:30:00
# Step 1 of the hand-over check, on the WHOLE tree of the commit that rebuilt the uploads byte for byte (fdac00b) plus
# the hand-over patch and entry point, inside sandbox.sh (none of the author's paths visible):
#   ident  the eleven checkpoints of record + the committed band -> both archives, compared file by file with the uploads
#   smoke  run_track2.sh from the released data at 2 epochs (every stage, band refit, build, gates), under strace:
#          the files it opens are the allowlist of the curated package (make_package.sh)
set -uo pipefail
MAIN=/users/PAS1187/qwu/workspace/realpde/main
FP=$MAIN/final_package
REF=${RPDE_HANDOVER_REF:-fdac00b}
W=/fs/scratch/PAS1187/qwu/tmp/realpde/final1005/trace
rm -rf "$W"; mkdir -p "$W"/{ident,smoke}/{pkg,work} || exit 1
for x in ident smoke; do
  git -C "$MAIN" archive "$REF" | tar -x -C "$W/$x/pkg" || exit 1
  ( cd "$W/$x/pkg" && for p in "$FP"/patches/*.patch; do patch -p1 < "$p" || exit 1; done ) || exit 1
  cp -p "$FP"/overlay/run_track2.sh "$W/$x/pkg/" || exit 1
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== IDENT: checkpoints of record + committed band, against the uploaded archives ==="
bash "$FP/verify/sandbox.sh" "$W/ident/pkg" "$W/ident/work" \
    refckpt=$MAIN/Track_1/unet ref=$MAIN/../sub_candidates/final -- bash -c '
  export RPDE=$ORG/data RPDE_KIT=$ORG/kit PY=$ORG/env/bin/python
  RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=$ORG/refckpt RPDE_REPRO_LOG=$ORG/work/log RPDE_REPRO_PKG=$ORG/work/pkg \
  RPDE_REF_T1=$ORG/ref/track1_k27fin_a080.zip RPDE_REF_T2=$ORG/ref/T2_MRGP2_track2_MRGP2.zip \
  bash reproduce.sh final' 2>&1 | tr -d '\000' | tee "$W/ident.log" | grep -a "SKIP\|TRAIN\|IDENTITY\|gates:\|CAUSAL\|BUILD_FINAL\|FAILED\|REFUSING\|Error\|sha\|\.zip"
echo "IDENT rc ${PIPESTATUS[0]}"

echo "=== SMOKE: run_track2.sh from the released data, 2 epochs, traced ==="
date
bash "$FP/verify/sandbox.sh" "$W/smoke/pkg" "$W/smoke/work" -- bash -c '
  RPDE=$ORG/data RPDE_KIT=$ORG/kit WORK=$ORG/work RPDE_REPRO_EPOCHS=2 \
  strace -f -qq -y --seccomp-bpf -e trace=openat,execve -o $ORG/work/trace.log bash run_track2.sh' 2>&1 | tr -d '\000' \
  | tee "$W/smoke.log" | grep -a "python \|data:\|SMOKE\|TRAIN\|SKIP\|refit\|gates:\|CAUSAL\|BUILD_FINAL\|FAILED\|REFUSING\|RUN_TRACK2\|TRACK 2\|run_track2\|\.zip\|Error"
echo "SMOKE rc ${PIPESTATUS[0]}"
date
# Every file of the package that was opened and existed in the commit: the allowlist.
ORGP=$(grep -ao '/tmp/rp_org_[0-9]*/pkg/' "$W/smoke/work/trace.log" | head -1)
grep -a -v ' = -1 ' "$W/smoke/work/trace.log" | grep -ao "<${ORGP}[^>]*>" | sed "s|^<${ORGP}||; s|>$||" | sort -u > "$W/opened_all.txt"
git -C "$MAIN" ls-tree -r --name-only "$REF" | sort > "$W/tree.txt"
comm -12 "$W/opened_all.txt" "$W/tree.txt" > "$W/opened.txt"
echo "opened: $(wc -l < "$W/opened_all.txt") paths under the package, $(wc -l < "$W/opened.txt") of them files of $REF"
gzip -f "$W/smoke/work/trace.log"
echo RP_FIN_TRACE_END
