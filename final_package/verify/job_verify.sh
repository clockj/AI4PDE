#!/bin/bash
#SBATCH -J rp_fin_verify
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 02:30:00
# Step 2 of the hand-over check: the CURATED package exactly as it would be sent (make_package.sh), extracted from its
# archives into fresh directories and run inside sandbox.sh, where none of the author's directories exist.
#   smoke   from the .tar.gz: run_track2.sh from the released data at 2 epochs -- every stage, band refit, build, gates
#   ident   from the .zip: the eleven checkpoints of record + the band of the submission -> both archives, compared file
#           by file with the uploads (the band and its json are copied in for this check only; the package has none)
#   no7575  from the .zip: two training stages on a copy of the release WITHOUT train_real/7575_0.h5
set -uo pipefail
MAIN=/users/PAS1187/qwu/workspace/realpde/main
FP=$MAIN/final_package
W=/fs/scratch/PAS1187/qwu/tmp/realpde/final1005/verify
N=realpde_track2_training_package
rm -rf "$W"; mkdir -p "$W"/{dist,smoke,ident,no7575}/work || exit 1
DRAFT=1 bash "$FP/make_package.sh" "$W/dist" || exit 1
tar -xzf "$W/dist/$N.tar.gz" -C "$W/smoke" || exit 1
for x in ident no7575; do ( cd "$W/$x" && unzip -q "$W/dist/$N.zip" ) || exit 1; done
diff -r "$W/smoke/$N" "$W/ident/$N" && echo "tar.gz and zip extract to the same tree"
( cd "$W/ident/$N" && sha256sum --quiet -c MANIFEST.sha256 ) && echo "MANIFEST.sha256 OK ($(wc -l < "$W/ident/$N/MANIFEST.sha256") files)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
KEEP="python \|data:\|SMOKE\|TRAIN\|SKIP\|refit\|gates:\|CAUSAL\|IDENTITY\|BUILD_FINAL\|FAILED\|REFUSING\|RUN_TRACK2\|TRACK 2\|run_track2\|STAGES DONE\|\.zip\|\.pth\|Error"

echo "=== SMOKE: the package from its .tar.gz, run_track2.sh from the released data, 2 epochs ==="
date
bash "$FP/verify/sandbox.sh" "$W/smoke/$N" "$W/smoke/work" -- bash -c '
  RPDE=$ORG/data RPDE_KIT=$ORG/kit WORK=$ORG/work RPDE_REPRO_EPOCHS=2 bash run_track2.sh' 2>&1 | tr -d '\000' \
  | tee "$W/smoke.log" | grep -a "$KEEP"
echo "SMOKE rc ${PIPESTATUS[0]}"
date

echo "=== IDENT: the package from its .zip + the checkpoints and the band of record, against the uploaded archives ==="
for f in Track_1/bands/k27_sigblend_lam0.5.npz Track_1/sigblend/band_k0_sigblend_lam0.5.json; do
  mkdir -p "$W/ident/$N/$(dirname $f)" && git -C "$MAIN" show fdac00b:$f > "$W/ident/$N/$f" || exit 1
done
bash "$FP/verify/sandbox.sh" "$W/ident/$N" "$W/ident/work" \
    refckpt=$MAIN/Track_1/unet ref=$MAIN/../sub_candidates/final -- bash -c '
  ln -s ../../Track_1/submit_fno/_fno.py Track_2/submit/_fno.py
  export RPDE=$ORG/data RPDE_KIT=$ORG/kit PY=$ORG/env/bin/python
  RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=$ORG/refckpt RPDE_REPRO_LOG=$ORG/work/log RPDE_REPRO_PKG=$ORG/work/pkg \
  RPDE_REF_T1=$ORG/ref/track1_k27fin_a080.zip RPDE_REF_T2=$ORG/ref/T2_MRGP2_track2_MRGP2.zip \
  bash reproduce.sh final' 2>&1 | tr -d '\000' | tee "$W/ident.log" | grep -a "$KEEP"
echo "IDENT rc ${PIPESTATUS[0]}"

echo "=== NO7575: two training stages, 1 epoch, on the release without train_real/7575_0.h5 ==="
bash "$FP/verify/sandbox.sh" "$W/no7575/$N" "$W/no7575/work" -- bash -c '
  D=$ORG/work/data81; mkdir -p $D/train_real/train_real
  for f in $ORG/data/train_real/train_real/*.h5; do [ "$(basename $f)" = 7575_0.h5 ] || ln -s $f $D/train_real/train_real/; done
  ln -s $ORG/data/train_sim $D/train_sim
  for st in twins_ffno members_ffno; do
    RPDE=$D RPDE_KIT=$ORG/kit WORK=$ORG/work RPDE_REPRO_EPOCHS=1 RPDE_REPRO_SEEDS=1 RPDE_REPRO_STAGE=$st bash run_track2.sh || exit 1
  done; ls -la $ORG/work/ckpt/*.pth' 2>&1 | tr -d '\000' | tee "$W/no7575.log" | grep -a "$KEEP"
echo "NO7575 rc ${PIPESTATUS[0]}"
date
echo RP_FIN_VERIFY_END
