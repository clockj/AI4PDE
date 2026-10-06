#!/bin/bash
#SBATCH -J rp_fin_vbranch
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 02:00:00
# The reorganised package (branch final-submission, worktree ../code-finalsub), exactly as `git archive` of its HEAD
# would be sent, extracted into fresh directories and run inside sandbox.sh (none of the author's paths visible):
#   smoke   run_track2.sh from the released data at 2 epochs: every stage, band, archive, gates
#   ident   build/build.py with the checkpoints and the band of record, against the uploaded archive (9/9 files)
#   no7575  two training stages at 1 epoch on a copy of the release without train_real/7575_0.h5
set -uo pipefail
MAIN=/users/PAS1187/qwu/workspace/realpde/main
FP=$MAIN/final_package
BR=/users/PAS1187/qwu/workspace/realpde/code-finalsub
W=/fs/scratch/PAS1187/qwu/tmp/realpde/final1005/vbranch
rm -rf "$W"; mkdir -p "$W"/{smoke,ident,no7575}/{pkg,work} || exit 1
git -C "$BR" archive HEAD -o "$W/package.tar" || exit 1
for x in smoke ident no7575; do tar -xf "$W/package.tar" -C "$W/$x/pkg" || exit 1; done
echo "package: $(git -C "$BR" rev-parse --short HEAD), $(tar -tf "$W/package.tar" | grep -vc '/$') files"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
KEEP="python \|data:\|EPOCHS\|TRAIN\|SKIP\|q (u, v)\|SIGPOW\|wrote\|FAIL\|scores\|ms/step\|SUBMITTABLE\|CAUSALITY\|IDENTITY\|RUN_TRACK2\|TRACK 2\|run_track2\|Error\|rc "

echo "=== SMOKE: run_track2.sh from the released data, 2 epochs ==="
date
bash "$FP/verify/sandbox.sh" "$W/smoke/pkg" "$W/smoke/work" -- bash -c '
  RPDE=$ORG/data RPDE_KIT=$ORG/kit WORK=$ORG/work EPOCHS=2 bash run_track2.sh' 2>&1 | tr -d '\000' \
  | tee "$W/smoke.log" | grep -a "$KEEP"
echo "SMOKE rc ${PIPESTATUS[0]}"
date

echo "=== IDENT: build.py with the checkpoints and the band of record, against the upload ==="
git -C "$MAIN" show fdac00b:Track_1/bands/k27_sigblend_lam0.5.npz > "$W/ident/work/k27.npz"
git -C "$MAIN" show fdac00b:Track_1/sigblend/band_k0_sigblend_lam0.5.json > "$W/ident/work/k27.json"
bash "$FP/verify/sandbox.sh" "$W/ident/pkg" "$W/ident/work" \
    refckpt=$MAIN/Track_1/unet ref=$MAIN/../sub_candidates/final -- bash -c '
  export RPDE=$ORG/data RPDE_KIT=$ORG/kit PY=$ORG/env/bin/python
  U=$ORG/refckpt
  $PY build/build.py --members $U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth \
      $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth \
      --band $ORG/work/k27.npz --sigblend $ORG/work/k27.json --stage $ORG/work/stage --out $ORG/work/track2.zip || exit 1
  $PY - $ORG/work/track2.zip $ORG/ref/T2_MRGP2_track2_MRGP2.zip <<PE
import hashlib, sys, zipfile
a, b = (zipfile.ZipFile(p) for p in sys.argv[1:3])
h = lambda z: {n: hashlib.md5(z.read(n)).hexdigest() for n in sorted(z.namelist())}
ha, hb = h(a), h(b)
bad = [n for n in sorted(set(ha) | set(hb)) if ha.get(n) != hb.get(n)]
print("IDENTITY vs uploaded T2_MRGP2_track2_MRGP2.zip:", "ALL %d FILES BYTE-IDENTICAL" % len(ha) if not bad else "DIFFER %s" % bad)
sys.exit(1 if bad else 0)
PE
' 2>&1 | tr -d '\000' | tee "$W/ident.log" | grep -a "$KEEP"
echo "IDENT rc ${PIPESTATUS[0]}"

echo "=== NO7575: two training stages, 1 epoch, on the release without train_real/7575_0.h5 ==="
bash "$FP/verify/sandbox.sh" "$W/no7575/pkg" "$W/no7575/work" -- bash -c '
  D=$ORG/work/data81; mkdir -p $D/train_real/train_real
  for f in $ORG/data/train_real/train_real/*.h5; do [ "$(basename $f)" = 7575_0.h5 ] || ln -s $f $D/train_real/train_real/; done
  ln -s $ORG/data/train_sim $D/train_sim
  RPDE=$D RPDE_KIT=$ORG/kit WORK=$ORG/work EPOCHS=1 bash run_track2.sh twins_ffno 1 || exit 1
  RPDE=$D RPDE_KIT=$ORG/kit WORK=$ORG/work EPOCHS=1 bash run_track2.sh members_ffno 1 || exit 1
  ls -la $ORG/work/ckpt/*.pth' 2>&1 | tr -d '\000' | tee "$W/no7575.log" | grep -a "$KEEP\|\.pth"
echo "NO7575 rc ${PIPESTATUS[0]}"
date
echo RP_FIN_VBRANCH_END
