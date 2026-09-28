#!/bin/bash
# OOD stream instrument, one ensemble end to end (GPU node; submit with ood_submit.sh).
#
#   bash ood_run.sh NAME REGIME LAYOUT            # LAYOUT = mirror | six | vitc2 | standin | twins | list
#   OOD_MEMBERS="a.pth b.pth ..." bash ood_run.sh NAME REGIME list
#
# Stages (OOD_STAGES, default all, in this order):
#   band      fit the regime band on the ensemble's own errors on its fit split (VAL_TRAJ minus the regime):
#             static fit_band.py (alpha 0.80, centre 0, p 0.45, inflate 1.39, fill-unscored, smooth 3, AoA 15-20 part of
#             the split) -> dump_twins.py on the whole split -> make_k0_band.py (lambda 0.5, inflate 1.29, 16 bins).
#             OOD_BAND=record skips the fit and uses the band of record (k27 + its json); OOD_BAND=<npz> OOD_SIGBLEND=<json>
#             uses a given pair.
#   build     Track_1/submit_fno/build_final.sh in the scratch worktree with RPDE_A = RPDE_M = the ensemble and the band
#             above; every other constant as build_final.sh ships it. Must end BUILD_FINAL OK.
#   variants  extract track2_<tag>.zip to pkg/record and derive the knob variants (ood_variants.py).
#   stream    ood_stream.py per variant (one process each; --causal-check on control), json per variant.
#   report    ood_report.py -> report.md, summary.json.
# Everything lands in $OOD_ROOT/runs/NAME. Nothing is written under $HOME.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${1:?NAME}; REGIME=${2:?REGIME}; LAYOUT=${3:?LAYOUT}
ROOT=${OOD_ROOT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood}
REPO=/users/PAS1187/qwu/workspace/realpde/main
COMMIT=${OOD_COMMIT:-$(git -C "$ROOT/wt_ood" rev-parse HEAD)}     # the campaign's pinned commit (wt_ood's HEAD)
PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
STAGES=${OOD_STAGES:-band build variants stream report}
VARIANTS=${OOD_VARIANTS:-frozen base control prev_input statsmask famw}
# OOD_CSET=1 adds the one-at-a-time stream-constant variants (ood_variants.py CSET); OOD_REBLANK="m1 m2 m3" streams
# OOD_REBLANK_VARIANTS (default control statsmask prev_input) once more per mask, json <variant>__<mask>.json.
[ "${OOD_CSET:-0}" = 1 ] && VARIANTS="$VARIANTS $($PY "$HERE/ood_variants.py" --list-cset)"
for sp in ${OOD_VARIANT_SPECS:-}; do VARIANTS="$VARIANTS ${sp%%:*}"; done      # ood_variants.py reads the specs from the env
REBLANK=${OOD_REBLANK:-}
RB_VARIANTS=${OOD_REBLANK_VARIANTS:-control statsmask prev_input}
RUN=$ROOT/runs/$NAME
# Every run gets its OWN source tree ($RUN/src, `git archive` of the pinned commit): the build seds tracked files and the
# band fit imports _fno.py, so two runs must never share a tree (flock on scratch does not serialise across nodes).
WT=$RUN/src
if [ ! -f "$WT/.ood_src_$COMMIT" ]; then
  rm -rf "$WT"; mkdir -p "$WT"
  git -C "$REPO" archive "$COMMIT" | tar -x -C "$WT" || exit 1
  for v in val_seen val_unseen; do ln -s "$REPO/Track_1/$v.npz" "$WT/Track_1/$v.npz"; done
  touch "$WT/.ood_src_$COMMIT"
fi
export OOD_WT=$WT PY TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$RUN"/{band,build,pkg,json,log}
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
say() { echo "[ood_run $(date +%H:%M:%S)] $*"; }
cd "$WT" || exit 1
SEDDED="Track_1/submit_fno/submission.py Track_1/submit_fno/_fno.py Track_2/submit/submission.py"
pristine() { for f in $SEDDED; do git -C "$REPO" show "$COMMIT:$f" | cmp -s - "$WT/$f" || echo "$f"; done; }
say "run $NAME regime $REGIME layout $LAYOUT  src $WT @ ${COMMIT:0:7}  node $(hostname)  job ${SLURM_JOB_ID:-none}"
[ -z "$(pristine)" ] || { say "REFUSING: the source tree differs from $COMMIT in: $(pristine)"; exit 1; }

# members: derived and config-checked (refuses a member that saw an evaluation or fit trajectory, or all-data)
if [ "$LAYOUT" = list ]; then
  MEMBERS=$($PY "$HERE/ood_lib.py" members --regime "$REGIME" --members ${OOD_MEMBERS:?OOD_MEMBERS} 2> "$RUN/members.txt") \
    || { cat "$RUN/members.txt"; exit 1; }
else
  MEMBERS=$($PY "$HERE/ood_lib.py" members --regime "$REGIME" --layout "$LAYOUT" 2> "$RUN/members.txt") \
    || { cat "$RUN/members.txt"; exit 1; }
fi
cat "$RUN/members.txt"; echo "$MEMBERS" > "$RUN/members.list"

# band
# i6fit (arm I6): the band is fitted post hoc on the pooled held-out dumps of OTHER regimes (OOD_I6_SOURCES = "dump[:group] ...",
# OOD_I6_TARGET_BAND = the band whose sigma_ref this ensemble uses), with fit_band.py patched (patches/fit_band_from_dump.patch,
# applied to THIS run's source tree only) and make_k0_band.py unchanged; then built and streamed like any other band.
if has i6fit; then
  if [ ! -f "$WT/.ood_patched_i6" ]; then
    patch -p1 -d "$WT" < "$HERE/../patches/fit_band_from_dump.patch" > "$RUN/log/i6_patch.log" || { cat "$RUN/log/i6_patch.log"; exit 1; }
    touch "$WT/.ood_patched_i6"
  fi
  say "i6fit: pooled band from ${OOD_I6_SOURCES:?OOD_I6_SOURCES}"
  $PY "$HERE/ood_i6.py" fit --src "$WT" --target-band "${OOD_I6_TARGET_BAND:?OOD_I6_TARGET_BAND}" --out "$RUN/band/i6" \
      $OOD_I6_SOURCES > "$RUN/log/i6fit.log" 2>&1 || { tail -30 "$RUN/log/i6fit.log"; say "FAILED: i6fit"; exit 1; }
  grep -a "pooled\|band   \|lam 0.5\|FINAL fit" "$RUN/log/i6fit.log" | cut -c1-300
  OOD_BAND=$RUN/band/i6/k0/band_k0_sigblend_lam0.5.npz; OOD_SIGBLEND=$RUN/band/i6/k0/band_k0_sigblend_lam0.5.json
fi
case "${OOD_BAND:-fit}" in
  record) BAND=$WT/Track_1/bands/k27_sigblend_lam0.5.npz; SB=$WT/Track_1/sigblend/band_k0_sigblend_lam0.5.json ;;
  fit)    BAND=$RUN/band/k0/band_k0_sigblend_lam0.5.npz; SB=$RUN/band/k0/band_k0_sigblend_lam0.5.json ;;
  *)      BAND=$OOD_BAND; SB=${OOD_SIGBLEND:?OOD_SIGBLEND with OOD_BAND} ;;
esac
if has band && [ "${OOD_BAND:-fit}" = fit ]; then
  say "band: fit split"
  eval "$($PY "$HERE/ood_lib.py" split --regime "$REGIME" --style "${OOD_BANDSTYLE:-heldout}" --write "$RUN/band/fit_split.npz" 2> "$RUN/log/split.log")" \
    || { cat "$RUN/log/split.log"; exit 1; }
  cat "$RUN/log/split.log"
  say "band: static fit (style ${OOD_BANDSTYLE:-heldout}: --fit-on $FIT_ON --fit-aoa $FIT_AOA)"
  $PY Track_1/submit_fno/fit_band.py --ckpt $MEMBERS --err-ckpt $MEMBERS --alpha 0.80 --band-p 0.45 --fill-unscored \
      --centre-alpha 0 --fit-on $FIT_ON --fit-aoa $FIT_AOA --out "$RUN/band/band_static.npz" > "$RUN/log/fit_band.log" 2>&1 \
    || { tail -20 "$RUN/log/fit_band.log"; say "FAILED: static band"; exit 1; }
  grep -a "fit-on\|band  \|fill-unscored\|wrote" "$RUN/log/fit_band.log"
  say "band: twin dump on the fit split"
  $PY Track_1/sigblend/dump_twins.py "$RUN/band/dump_fit.npz" --val "$RUN/band/fit_split.npz" --twins $MEMBERS \
      > "$RUN/log/dump_twins.log" 2>&1 || { tail -20 "$RUN/log/dump_twins.log"; say "FAILED: dump"; exit 1; }
  say "band: SIGBLEND (lambda 0.5)"
  ( cd Track_1/sigblend && B_DUMP="$RUN/band/dump_fit.npz" K0_BAND="$RUN/band/band_static.npz" K0_INFL=1.29 K0_NB=16 \
      PYTHONPATH=.. $PY make_k0_band.py "$RUN/band/k0" ) > "$RUN/log/make_k0_band.log" 2>&1 \
    || { tail -20 "$RUN/log/make_k0_band.log"; say "FAILED: SIGBLEND"; exit 1; }
  grep -a "lam 0.5\|FINAL fit\|lam0.5.npz" "$RUN/log/make_k0_band.log" | cut -c1-400
  rm -f "$RUN/band/dump_fit.npz"                  # 0.5 GB; re-created by the stage if needed
fi
[ -f "$BAND" ] && [ -f "$SB" ] || { say "no band: $BAND / $SB"; exit 1; }
say "band $BAND  md5 $(md5sum < "$BAND" | cut -c1-12)  sigblend $SB"

TAG=ood_$NAME
if has build; then
  say "build (build_final.sh, RPDE_A = RPDE_M = ensemble)"
  RPDE_FORCE=1 RPDE_TAG=$TAG RPDE_A="$MEMBERS" RPDE_M="$MEMBERS" RPDE_BAND="$BAND" RPDE_SIGBLEND="$SB" \
      bash Track_1/submit_fno/build_final.sh "$RUN/build" > "$RUN/log/build.log" 2>&1
  tr -d '\000' < "$RUN/log/build.log" | grep -a "^band \|gates:\|CAUSAL\|SUBMITTABLE\|BUILD_FINAL\|extracted\|load:" | cut -c1-400
  grep -aq "BUILD_FINAL OK" "$RUN/log/build.log" || { say "FAILED: build (see $RUN/log/build.log)"; exit 1; }
  [ -z "$(pristine)" ] || { say "FAILED: the build left edits in: $(pristine)"; exit 1; }
  # the Track-2 gate's load line must carry the shipped stream knobs and this band's SIGPOW
  grep -a "\[rpde-t2\] load" "$RUN/log/build.log" | tail -1 | grep -q "r1 True r1_asym True r1_tsm 3 r1_blk_n0 20.0 r1_blk 4 d1c_n0 30.0 d1c_eb True d1c_eb_n0 1.0 d1c_lead_n0 5.0 vc_n0 30.0 fw_n0 0.0 sigpow (" \
    || { say "FAILED: the Track-2 gate did not load the shipped constants"; exit 1; }
fi
if has variants; then
  rm -rf "$RUN/pkg"; mkdir -p "$RUN/pkg/record"
  $PY -m zipfile -e "$RUN/build/track2_$TAG.zip" "$RUN/pkg/record" || exit 1
  $PY "$HERE/ood_variants.py" "$RUN/pkg" $VARIANTS || exit 1
fi
if has stream; then
  for v in $VARIANTS; do
    [ "${OOD_SKIP_EXISTING:-0}" = 1 ] && [ -s "$RUN/json/$v.json" ] && { say "stream $v: json exists, skipped"; continue; }
    say "stream $v"
    CC=""; [ "$v" = control ] && CC="--causal-check"
    $PY "$HERE/ood_stream.py" --pkg "$RUN/pkg/$v" --regime "$REGIME" --out "$RUN/json/$v.json" $CC 2>&1 \
      | tr -d '\000' | tee "$RUN/log/stream_$v.log" | grep -av "^\s*$" | cut -c1-400
    [ "${PIPESTATUS[0]}" = 0 ] || { say "FAILED: stream $v"; exit 1; }
  done
  for m in $REBLANK; do
    for v in $RB_VARIANTS; do
      [ "${OOD_SKIP_EXISTING:-0}" = 1 ] && [ -s "$RUN/json/${v}__$m.json" ] && continue
      say "stream $v re-blanked $m"
      $PY "$HERE/ood_stream.py" --pkg "$RUN/pkg/$v" --regime "$REGIME" --reblank "$m" --out "$RUN/json/${v}__$m.json" 2>&1 \
        | tr -d '\000' | tee "$RUN/log/stream_${v}__$m.log" | grep -a "^  \|WARN\|Error\|FAIL" | cut -c1-300
      [ "${PIPESTATUS[0]}" = 0 ] || { say "FAILED: stream $v $m"; exit 1; }
    done
  done
fi
# i6dump (arm I6): what a band fit needs on this regime's HELD-OUT windows (ood_i6dump.py -> $RUN/i6/heldout.npz).
# OOD_I6_IDCHECK=1 also dumps the regime's own static fit split and checks that the patched --from-dump path of fit_band.py
# reproduces the static band this run fitted by running the models.
if has i6dump; then
  say "i6dump"
  $PY "$HERE/ood_i6dump.py" --regime "$REGIME" --members $MEMBERS --band "$BAND" --out "$RUN/i6/heldout.npz" 2>&1 \
    | tee "$RUN/log/i6dump.log" | grep -a "ood_i6dump\|Error" ; [ "${PIPESTATUS[0]}" = 0 ] || { say "FAILED: i6dump"; exit 1; }
  if [ "${OOD_I6_IDCHECK:-0}" = 1 ] && [ -f "$RUN/band/band_static.npz" ]; then
    CK=$RUN/i6/idcheck; mkdir -p "$CK"; rm -rf "$CK/src"; mkdir -p "$CK/src"
    git -C "$REPO" archive "$COMMIT" Track_1 | tar -x -C "$CK/src" && patch -p1 -d "$CK/src" < "$HERE/../patches/fit_band_from_dump.patch" > /dev/null
    $PY "$HERE/ood_i6dump.py" --regime "$REGIME" --members $MEMBERS --band "$RUN/band/band_static.npz" --windows static-fit \
        --out "$CK/staticfit.npz" && \
    $PY - "$CK/staticfit.npz" "$CK/fit.npz" <<'PYEOF'
import sys, numpy as np
d = np.load(sys.argv[1])
R = np.clip(d["S"][0] / d["sref"][..., :2], 0.3, 3.0).astype(np.float32)
np.savez(sys.argv[2], P=d["P"], C=d["F"][0], Y=d["y"], R=R)
PYEOF
    $PY "$CK/src/Track_1/submit_fno/fit_band.py" --from-dump "$CK/fit.npz" --sigma-ref-from "$RUN/band/band_static.npz" \
        --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --out "$CK/band_static_fromdump.npz" > "$CK/fit.log" 2>&1
    $PY -c "
import numpy as np, sys
a, b = np.load('$RUN/band/band_static.npz'), np.load('$CK/band_static_fromdump.npz')
d = np.abs(a['base'] - b['base']); r = d / np.maximum(np.abs(a['base']), 1e-12)
print('[i6 idcheck] from-dump vs model-run static band: bit-identical', bool(np.array_equal(a['base'], b['base'])),
      ' max |diff| %.3e  max rel %.3e  elements differing %d of %d' % (d.max(), r.max(), int((d > 0).sum()), d.size))
" | tee "$CK/idcheck.txt"
    rm -rf "$CK/src" "$CK/fit.npz"
  fi
fi
# i7 (arm I7 screen): stream frozen and control once more with --dump, plain and under each OOD_I7_MASKS, then ood_i7.py.
# The dumps (float32, ~1.3 MB per window and variant) stay in $RUN/dump unless OOD_I7_KEEP=0.
if has i7; then
  for m in "" ${OOD_I7_MASKS-m1 m2 m3}; do
    for v in frozen control; do
      suf=${m:+__$m}
      say "i7 dump $v$suf"
      $PY "$HERE/ood_stream.py" --pkg "$RUN/pkg/$v" --regime "$REGIME" ${m:+--reblank $m} --dump "$RUN/dump/$v$suf" \
          --out "$RUN/json_i7/$v$suf.json" 2>&1 | tr -d '\000' | tee "$RUN/log/i7_$v$suf.log" | grep -a "^  \|dumped\|Error\|FAIL" | cut -c1-300
      [ "${PIPESTATUS[0]}" = 0 ] || { say "FAILED: i7 dump $v$suf"; exit 1; }
    done
  done
  $PY "$HERE/ood_i7.py" "$RUN" > "$RUN/log/i7.log" 2>&1 || { tail -30 "$RUN/log/i7.log"; say "FAILED: ood_i7.py"; exit 1; }
  say "i7 report $RUN/i7.md"
  [ "${OOD_I7_KEEP:-1}" = 0 ] && rm -rf "$RUN/dump"
fi
# idread: the in-distribution read of this regime ensemble (VAL_TRAJ minus the regime; the regime band is fitted on these
# trajectories, so band-dependent numbers are in sample) -> $RUN/idread/{json,report.md}
if has idread; then
  for v in ${OOD_IDREAD_VARIANTS:-frozen base control}; do
    say "idread $v"
    $PY "$HERE/ood_stream.py" --pkg "$RUN/pkg/$v" --regime "${REGIME}ID" --out "$RUN/idread/json/$v.json" 2>&1 \
      | tr -d '\000' | tee "$RUN/log/idread_$v.log" | grep -a "^  \|Error\|FAIL" | cut -c1-300
    [ "${PIPESTATUS[0]}" = 0 ] || { say "FAILED: idread $v"; exit 1; }
  done
  $PY "$HERE/ood_report.py" "$RUN/idread" > "$RUN/log/idread_report.log" 2>&1 || { tail -20 "$RUN/log/idread_report.log"; exit 1; }
fi
if has report; then
  $PY "$HERE/ood_report.py" "$RUN" > "$RUN/log/report.log" 2>&1 || { cat "$RUN/log/report.log"; exit 1; }
  say "report $RUN/report.md"
fi
say "done"
echo RP_DP_OOD_END
