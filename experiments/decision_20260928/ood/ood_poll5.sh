#!/bin/bash
# Chunk 5: submit each wave-3 read the moment its members (and sentinels) exist, with at most MAXJ of this instrument's
# jobs in the queue at once. Every 5 minutes, until everything is submitted or DEADLINE (epoch seconds) passes.
#   T4 at A20 / A10     t4/ckpt/<R>/<r>_ffnosim_resim_s1-4.pth + t4/logs/rp_dp_t4_<r>_s<k>.done        -> runs a20_t4, a10_t4
#   T4 twins (ID)       t4id/ckpt/ID/p2_ffno_l8_sim_resim_s1-3.pth + t4id/logs/rp_dp_t4id_s<k>.done    -> id_t4_ship
#   T4v (RE)            single-member screen (ood_single.py t4v) CLEARS                               -> re_t4v
#   T4b (RE)            single-member screen (ood_single.py t4b) CLEARS                               -> re_t4b
# A screen that is complete (all json present) and does not clear is recorded and not read further.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928
ST=$D/ood/wave5; mkdir -p "$ST"
MAXJ=${MAXJ:-4}; DEADLINE=${DEADLINE:-$(( $(date +%s) + 6 * 3600 ))}
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
nq() { squeue -M ascend -u "$USER" -h -o '%j' | grep -c '^rp_dp_ood_'; }
allf() { for f in "$@"; do case "$f" in *.done) [ -e "$f" ] || return 1;; *) [ -s "$f" ] || return 1;; esac; done; return 0; }   # sentinels are empty files
go() {   # go NAME REGIME LAYOUT TIME [VAR=value ...]
  local name=$1 reg=$2 lay=$3 t=$4; shift 4
  [ -f "$ST/$name.submitted" ] && return 0
  [ "$(nq)" -lt "$MAXJ" ] || { echo "[$(date +%H:%M)] $name ready, waiting for a slot"; return 0; }
  out=$(env -u OOD_BAND -u OOD_BANDSTYLE -u OOD_SKIP_EXISTING -u OOD_VARIANT_SPECS -u OOD_MEMBERS OOD_CSET=0 OOD_REBLANK= \
        OOD_VARIANTS="frozen base control" OOD_IDREAD_VARIANTS="frozen base control" \
        OOD_STAGES="band build variants stream idread report" "$@" bash "$HERE/ood_submit.sh" "$name" "$reg" "$lay" "$t" 2>&1)
  echo "[$(date +%H:%M)] $out"
  awk '/^SUBMITTED/ {print $2}' <<< "$out" > "$ST/$name.submitted"
}
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for r in a20 a10; do
    R=${r^^}
    allf $(for k in 1 2 3 4; do echo "$D/t4/ckpt/$R/${r}_ffnosim_resim_s$k.pth $D/t4/logs/rp_dp_t4_${r}_s$k.done"; done) \
      && go "${r}_t4" "$R" t4 00:50:00
  done
  allf $(for k in 1 2 3; do echo "$D/t4id/ckpt/ID/p2_ffno_l8_sim_resim_s$k.pth $D/t4id/logs/rp_dp_t4id_s$k.done"; done) \
    && go id_t4_ship ID twins_t4 00:40:00 OOD_BANDSTYLE=ship OOD_STAGES="band build variants stream report"
  for arm in t4v t4b; do
    [ -f "$ST/$arm.screen" ] && grep -q "CLEARS\|does not clear" "$ST/$arm.screen" && ! grep -q "missing" "$ST/$arm.screen" \
      || $PY "$HERE/ood_single.py" $arm > "$ST/$arm.screen" 2>&1
    if grep -q "CLEARS" "$ST/$arm.screen"; then
      if [ "$arm" = t4v ]; then
        allf $D/t4vc/ckpt/RE/re_vitc_resim_s3.pth $D/t4vp/ckpt/RE/re_vitp_resim_s1.pth && go re_t4v RE t4v 01:10:00
      else
        allf $(for k in 1 2 3 4; do echo "$D/t4b/ckpt/RE/re_ffnosim_resim50_s$k.pth"; done) && go re_t4b RE t4b 01:10:00
      fi
    fi
  done
  done_all=1
  for n in a20_t4 a10_t4 id_t4_ship; do [ -f "$ST/$n.submitted" ] || done_all=0; done
  for arm in t4v t4b; do
    if grep -q "CLEARS" "$ST/$arm.screen" 2>/dev/null; then [ -f "$ST/re_$arm.submitted" ] || done_all=0
    elif ! { grep -q "does not clear" "$ST/$arm.screen" 2>/dev/null && ! grep -q missing "$ST/$arm.screen"; }; then done_all=0; fi
  done
  [ "$done_all" = 1 ] && { echo "[$(date +%H:%M)] everything submitted or screened out"; exit 0; }
  sleep 300
done
echo "[$(date +%H:%M)] poller reached its deadline"
