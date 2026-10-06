#!/bin/bash
# Is the reorganised training code (branch final-submission) the same program as fdac00b's train2.py? On the CPU, where
# the arithmetic is deterministic, the three recipe families are trained for 2 epochs on a handful of windows
# (--stride 2000: one window per trajectory) with the same seed by both codes, and the state_dicts must be bit-identical.
# GPU kernels are not deterministic run to run (LayerNorm / attention backward), so the GPU is the wrong place for this
# test: job_equiv.sh measured old-vs-new there at 4e-7 (F-FNO) and 1e-2 (ViT) after 2 epochs, and the old code against
# itself at the same levels (jobs 55454404, 55454453).
#
#   bash cpu_equiv.sh <scratch dir> [<CFD-map cache dir>]      # ~15 min on 8 cores; needs the release under $RPDE
set -uo pipefail
MAIN=/users/PAS1187/qwu/workspace/realpde/main
BR=/users/PAS1187/qwu/workspace/realpde/code-finalsub
W=${1:?scratch dir}; CACHE=${2:-$W/cache}
export PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python} RPDE=${RPDE:-/users/PAS1187/qwu/RealPDE}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
REAL=$RPDE/train_real/train_real; SIM=$RPDE/train_sim/train_sim
rm -rf "$W/old" "$W/new" "$W/ck"; mkdir -p "$W/old" "$W/new" "$W/ck" "$CACHE" || exit 1
git -C "$MAIN" archive fdac00b Track_1/unet | tar -x -C "$W/old"
git -C "$BR" archive HEAD unet | tar -x -C "$W/new"
sed -i 's/^    dev = "cuda"$/    dev = "cpu"/' "$W/old/Track_1/unet/train2.py" "$W/new/unet/train.py"
grep -q 'dev = "cpu"' "$W/old/Track_1/unet/train2.py" && grep -q 'dev = "cpu"' "$W/new/unet/train.py" || { echo "patch failed"; exit 1; }
LOSS="--loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0"
OPT="--stride 2000 --batch 32 --lr 2e-4 --epochs 2 --wd 1e-5"
O_FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head $LOSS $OPT"
O_VITC="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head $LOSS $OPT"
O_VITP="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 $OPT"
O_SIMF="--sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 2000"
COMMON="--epochs 2 --batch 32 --stride 2000 --lr 2e-4 --wd 1e-5"
N_FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 $COMMON"
N_VITC="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 $COMMON"
N_VITP="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --lam-f 1.0 --lam-cov 0.0 --lam-grad 1.0 --lam-tgrad 0.0 --lam-v 0.3 --lam-t 0.15 $COMMON"
N_SIMF="--sim $SIM --lam-sim 0.3 --sim-stride 2000"
old() { ( cd "$W/old/Track_1/unet" && RPDE_SIMCACHE=$CACHE $PY train2.py --data "$REAL" "${@:2}" --out "$W/ck/old_$1.pth" ) > "$W/ck/old_$1.log" 2>&1; echo "  old $1 rc $?"; }
new() { ( cd "$W/new/unet" && RPDE_SIMCACHE=$CACHE $PY train.py --data "$REAL" "${@:2}" --out "$W/ck/new_$1.pth" ) > "$W/ck/new_$1.log" 2>&1; echo "  new $1 rc $?"; }
old twin_ffno_s1 $O_FFNO $O_SIMF --sim-exclude-held --seed 1;   new twin_ffno_s1 $N_FFNO $N_SIMF --sim-exclude-held --seed 1
old twin_vit_p $O_VITP --seed 1;                                new twin_vit_p $N_VITP --seed 1
old member_vit_c $O_VITC --seed 3 --all-data $O_SIMF;           new member_vit_c $N_VITC --seed 3 --all-data $N_SIMF
$PY - "$W/ck" <<'PE'
import sys, torch
W = sys.argv[1]; bad = 0
for n in ("twin_ffno_s1", "twin_vit_p", "member_vit_c"):
    a = torch.load(f"{W}/old_{n}.pth", map_location="cpu"); b = torch.load(f"{W}/new_{n}.pth", map_location="cpu")
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    eq = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    meta = all(a[k] == b[k] for k in ("width", "var_head", "arch", "modes_y", "modes_x", "n_layers", "ctx"))
    print(f"  CPU {n:14s}: weights {'BIT-IDENTICAL' if eq else 'DIFFER'}   checkpoint fields {'same' if meta else 'DIFFER'}")
    bad += (not eq) or (not meta)
print("CPU_EQUIV " + ("PASS" if not bad else "FAIL"))
PE
