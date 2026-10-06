#!/bin/bash
#SBATCH -J rp_fin_equiv
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 02:00:00
# Is the reorganised package (branch final-submission, ../code-finalsub) the same method as fdac00b? On one GPU:
#   train   three recipes (F-FNO twin with CFD, ViT twin without, ViT member with CFD) at 2 epochs, same seed, old code
#           and new code: the state_dicts must be bit-identical, and so must the CFD map caches
#   band    from the eleven checkpoints of record: old fit_band vs new fit_band (bit-identical static band, and equal to
#           the committed r38_a080_c0_fill.npz); old dump + make_k0_band vs new dump + sigma_band (q, log kappa, npz)
#   build   new build.py with the checkpoints and the band of record, against the uploaded archive: 9/9 files
#   gates   new validate.py and causal_check.py on that archive
set -uo pipefail
MAIN=/users/PAS1187/qwu/workspace/realpde/main
NEW=/users/PAS1187/qwu/workspace/realpde/code-finalsub
W=/fs/scratch/PAS1187/qwu/tmp/realpde/final1005/equiv
export PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
export RPDE=/users/PAS1187/qwu/RealPDE
export RPDE_KIT=/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export PYTHONDONTWRITEBYTECODE=1
REAL=$RPDE/train_real/train_real; SIM=$RPDE/train_sim/train_sim
rm -rf "$W"; mkdir -p "$W"/{old,new,ck_old,ck_new,cache_old,cache_new,val,band_old,band_new,pkg} || exit 1
git -C "$MAIN" archive fdac00b | tar -x -C "$W/old" || exit 1
( cd "$NEW" && git archive HEAD 2>/dev/null | tar -x -C "$W/new" ) || cp -r "$NEW"/. "$W/new/"
[ -f "$W/new/run_track2.sh" ] || cp -r "$NEW"/. "$W/new/"
cp -p "$MAIN"/Track_1/val_seen.npz "$MAIN"/Track_1/val_unseen.npz "$W/val/" || exit 1
cp -p "$W"/val/*.npz "$W/old/Track_1/"
nvidia-smi --query-gpu=name --format=csv,noheader
RC=0
ok() { if [ "$1" = 0 ]; then echo "PASS  $2"; else echo "FAIL  $2"; RC=1; fi; }

echo "=== TRAIN: old vs new, 2 epochs, same seed ==="
LOSS="--loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0"
OPT="--stride 5 --batch 32 --lr 2e-4 --epochs 2 --wd 1e-5"
O_FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head $LOSS $OPT"
O_VITC="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head $LOSS $OPT"
O_VITP="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 $OPT"
O_SIMF="--sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5"
COMMON="--epochs 2 --batch 32 --stride 5 --lr 2e-4 --wd 1e-5"
N_FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 $COMMON"
N_VITC="--arch vit --width 384 --n-layers 8 --dropout 0.3 --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 $COMMON"
N_VITP="--arch vit --width 384 --n-layers 8 --dropout 0.3 --lam-f 1.0 --lam-cov 0.0 --lam-grad 1.0 --lam-tgrad 0.0 --lam-v 0.3 --lam-t 0.15 $COMMON"
N_SIMF="--sim $SIM --lam-sim 0.3 --sim-stride 5"
old_train() { ( cd "$W/old/Track_1/unet" && RPDE_SIMCACHE=$W/cache_old $PY train2.py --data "$REAL" "${@:2}" --out "$W/ck_old/$1.pth" ) > "$W/ck_old/$1.log" 2>&1; echo "  old $1 rc $?  $(tail -1 "$W/ck_old/$1.log" | cut -c1-80)"; }
new_train() { ( cd "$W/new/unet" && RPDE_SIMCACHE=$W/cache_new $PY train.py --data "$REAL" "${@:2}" --out "$W/ck_new/$1.pth" ) > "$W/ck_new/$1.log" 2>&1; echo "  new $1 rc $?  $(tail -1 "$W/ck_new/$1.log" | cut -c1-80)"; }
old_train twin_ffno_s1 $O_FFNO $O_SIMF --sim-exclude-held --seed 1
new_train twin_ffno_s1 $N_FFNO $N_SIMF --sim-exclude-held --seed 1
old_train twin_vit_p $O_VITP --seed 1
new_train twin_vit_p $N_VITP --seed 1
old_train member_vit_c $O_VITC --seed 3 --all-data $O_SIMF
new_train member_vit_c $N_VITC --seed 3 --all-data $N_SIMF
$PY - "$W" <<'PE'
import sys, os, glob, numpy as np, torch
W = sys.argv[1]; bad = 0
for n in ("twin_ffno_s1", "twin_vit_p", "member_vit_c"):
    a = torch.load(f"{W}/ck_old/{n}.pth", map_location="cpu"); b = torch.load(f"{W}/ck_new/{n}.pth", map_location="cpu")
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    same_keys = list(sa) == list(sb)
    eq = same_keys and all(torch.equal(sa[k], sb[k]) for k in sa)
    worst = 0.0 if eq else max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if k in sb)
    top = {k for k in ("width", "var_head", "arch", "modes_y", "modes_x", "n_layers", "mean", "std", "ctx")}
    meta = all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in top)
    print(f"  {n:14s} keys {'same' if same_keys else 'DIFFER'}  weights {'BIT-IDENTICAL' if eq else f'DIFFER (max {worst:.3e})'}  meta {'same' if meta else 'DIFFER'}")
    bad += (not eq) or (not meta)
for f in sorted(glob.glob(f"{W}/cache_old/*.npy")):
    g = os.path.join(f"{W}/cache_new", os.path.basename(f))
    same = os.path.exists(g) and np.array_equal(np.load(f, mmap_mode="r"), np.load(g, mmap_mode="r"))
    print(f"  cache {os.path.basename(f)}: {'BIT-IDENTICAL' if same else 'DIFFER'}"); bad += not same
sys.exit(1 if bad else 0)
PE
ok $? "training code: old and new give the same weights and the same CFD map"

echo "=== BAND: old vs new, from the checkpoints of record ==="
U=$MAIN/Track_1/unet
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
( cd "$W/old" && $PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.80 --band-p 0.45 --fill-unscored --centre-alpha 0 --out "$W/band_old/static.npz" ) > "$W/band_old/fit.log" 2>&1; echo "  old fit_band rc $?"
( cd "$W/new" && $PY band/fit_band.py --members $A --twins $M --val "$W/val" --out "$W/band_new/static.npz" ) > "$W/band_new/fit.log" 2>&1; echo "  new fit_band rc $?"
( cd "$W/old" && $PY Track_1/sigblend/dump_twins.py "$W/band_old/dump.npz" --twins $M ) > "$W/band_old/dump.log" 2>&1; echo "  old dump rc $?"
( cd "$W/new" && $PY band/dump_twins.py --twins $M --val "$W/val" --out "$W/band_new/dump.npz" ) > "$W/band_new/dump.log" 2>&1; echo "  new dump rc $?"
( cd "$W/old/Track_1/sigblend" && B_DUMP=$W/band_old/dump.npz K0_BAND=$W/band_old/static.npz K0_INFL=1.29 K0_NB=16 PYTHONPATH=.. $PY make_k0_band.py "$W/band_old/k0" ) > "$W/band_old/k0.log" 2>&1; echo "  old make_k0_band rc $?"
( cd "$W/new" && $PY band/sigma_band.py --dump "$W/band_new/dump.npz" --band "$W/band_new/static.npz" --out "$W/band_new/k0" ) > "$W/band_new/k0.log" 2>&1; echo "  new sigma_band rc $?"
git -C "$MAIN" show fdac00b:Track_1/bands/r38_a080_c0_fill.npz > "$W/band_old/committed_static.npz"
$PY - "$W" <<'PE'
import sys, json, numpy as np
W = sys.argv[1]; bad = 0
def cmp(name, fa, fb, keys=None):
    global bad
    a, b = np.load(fa), np.load(fb)
    ks = keys or sorted(set(a.files) | set(b.files))
    for k in ks:
        same = k in a.files and k in b.files and np.array_equal(a[k], b[k])
        if not same:
            worst = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) if (k in a.files and k in b.files and a[k].shape == b[k].shape) else float("nan")
            print(f"  {name} [{k}]: DIFFER (max |diff| {worst:.3e})"); bad += 1
        else:
            print(f"  {name} [{k}]: BIT-IDENTICAL")
cmp("static old vs new", f"{W}/band_old/static.npz", f"{W}/band_new/static.npz")
cmp("static new vs committed r38_a080_c0_fill", f"{W}/band_new/static.npz", f"{W}/band_old/committed_static.npz", ["base", "sigma_ref", "p"])
cmp("dump old vs new", f"{W}/band_old/dump.npz", f"{W}/band_new/dump.npz", ["F", "S"])
jo = json.load(open(f"{W}/band_old/k0/band_k0_sigblend_lam0.5.json")); jn = json.load(open(f"{W}/band_new/k0/band_sigblend_lam0.5.json"))
print(f"  q old {jo['q']}  new {jn['q']}  equal {jo['q'] == jn['q']}  T2_sigpow equal {jo['T2_sigpow'] == jn['T2_sigpow']}")
bad += jo["q"] != jn["q"]
cmp("sigblend npz old vs new", f"{W}/band_old/k0/band_k0_sigblend_lam0.5.npz", f"{W}/band_new/k0/band_sigblend_lam0.5.npz")
sys.exit(1 if bad else 0)
PE
ok $? "band code: old and new give the same static band (= the committed one), dump and sigma power law"

echo "=== BUILD: new build.py with the checkpoints and band of record, against the upload ==="
git -C "$MAIN" show fdac00b:Track_1/bands/k27_sigblend_lam0.5.npz > "$W/pkg/k27.npz"
git -C "$MAIN" show fdac00b:Track_1/sigblend/band_k0_sigblend_lam0.5.json > "$W/pkg/k27.json"
( cd "$W/new" && $PY build/build.py --members $A --band "$W/pkg/k27.npz" --sigblend "$W/pkg/k27.json" --stage "$W/pkg/stage" --out "$W/pkg/track2.zip" ) 2>&1 | tail -14
$PY - "$W/pkg/track2.zip" "$MAIN/../sub_candidates/final/T2_MRGP2_track2_MRGP2.zip" <<'PE'
import hashlib, sys, zipfile
a, b = (zipfile.ZipFile(p) for p in sys.argv[1:3])
h = lambda z: {n: hashlib.md5(z.read(n)).hexdigest() for n in sorted(z.namelist())}
ha, hb = h(a), h(b)
bad = [n for n in sorted(set(ha) | set(hb)) if ha.get(n) != hb.get(n)]
print("IDENTITY new build vs uploaded T2_MRGP2_track2_MRGP2.zip:", "ALL %d FILES BYTE-IDENTICAL" % len(ha) if not bad else "DIFFER %s" % bad)
sys.exit(1 if bad else 0)
PE
ok $? "build: the archive of record, file by file"

echo "=== GATES: new validate.py and causal_check.py on that archive ==="
( cd "$W/new" && $PY build/validate.py --zip "$W/pkg/track2.zip" --extract "$W/pkg/validate" ) 2>&1 | tee "$W/pkg/validate.log" | grep -a "FAIL\|scores\|ms/step\|SUBMITTABLE"
grep -aq "^SUBMITTABLE" "$W/pkg/validate.log"; ok $? "validate.py: SUBMITTABLE"
( cd "$W/new" && $PY build/causal_check.py --submission "$W/pkg/validate" ) 2>&1 | tee "$W/pkg/causal.log" | tail -2
grep -aq "CAUSALITY OK" "$W/pkg/causal.log"; ok $? "causal_check.py: CAUSALITY OK"

[ "$RC" = 0 ] && echo "EQUIV ALL PASS" || echo "EQUIV FAILED"
echo RP_FIN_EQUIV_END
