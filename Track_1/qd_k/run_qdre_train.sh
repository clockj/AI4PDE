#!/usr/bin/env bash
# Bram / errhead_s1 -- does a head trained on the model's OWN per-frame error carry information
# the shipped sigma head cannot? ONE job, TWO checkpoints, same seed, paired:
#
#   ARM      the F-FNO held-out twin recipe (REPRODUCE.md 1.5, seed 1) PLUS --err-head
#   CONTROL  the identical command WITHOUT --err-head
#
# --var-head regresses onto std_t(y), the TARGET window's per-pixel temporal std: one amplitude
# envelope shared by all 20 forecast frames, a statistic of the data that never sees what this
# model got wrong. --err-head (new, this request's code) adds a SECOND head emitting (B, 20*C,
# H, W) -- the field's own shape, frame axis included -- trained on |pred - y| with pred
# detached. It SHARES the trunk's gradient (ROUNDS.md:377-380: a detached variance head reaches
# corr 0.679, worse than copying the input window's std at 0.756, against 0.907 shared -- and an
# error head is worse off still, since detached it must predict this model's own failures from
# features the field task had no reason to encode). So the field DOES move, the accuracy
# subscores DO move, and the CONTROL MUST BE TRAINED. check_pair.py no longer asserts a bitwise
# field; it reports the difference, and the reading prices accuracy and band separately.
#
# Smoke:  EPOCHS=1 bash run.sh
set -uo pipefail
shopt -s nullglob

# `pitzer submit` has NO --env and runs `bash /work/run.sh` bare: nothing from the manager's
# shell reaches this script, so every ${VAR:-default} below resolves to its DEFAULT in the job,
# silently (pitzer-manager, msg 4ff578ad -- they caught exactly this in this file before
# submitting it). The defaults here are therefore THE VALUES THIS DIRECTORY IS MEANT TO RUN.
# The smoke lives in its own directory, ../errhead_smoke, with its own hard-coded values --
# it is not this directory with a switch flipped.
EPOCHS=400                         # HARD-CODED. The smoke is ../errhead_smoke, not a switch.
PY=${RPDE_PY:-/users/PAS1187/qwu/envs/mural_osc/bin/python}
RPDE=${RPDE_DATA:-/users/PAS1187/qwu/RealPDE}
REAL=$RPDE/train_real/train_real
SIM=$RPDE/train_sim/train_sim
CKD=$RPDE/checkpoints_r34          # read-only, never written; MANIFEST.md + SHA256SUMS live there
W=${RPDE_QDRE_WORK:-/work}                            # this request directory, as mounted in the job
CODE=$W/code
OUT=${RPDE_QDRE_OUT:-/out}
CK=/tmp/ck
VALD=/tmp/val                      # make_val.py writes relative to cwd, and /work may be read-only

fail() { echo "FAIL: $*" >&2; exit 1; }

# Assign in the CURRENT shell, never through $( ): a helper that calls `fail` inside command
# substitution only kills the subshell, the script sails on, and the FAIL message itself lands
# in the variable as a legal-looking path. printf -v makes the exit an actual exit.
need() {                           # need VAR filename -- locate a shipped checkpoint under $CKD
  local __v=$1 __n=$2 __f
  __f=$(find "$CKD" -name "$__n" -print -quit 2>/dev/null)
  [ -n "$__f" ] || fail "missing checkpoint: $__n (searched $CKD)"
  printf -v "$__v" '%s' "$__f"
}

mkdir -p "$OUT/logs" "$CK" "$VALD" || fail "mkdir"

# mural_osc is torch 2.7 / numpy 2 and torch>=2.6 defaults torch.load(weights_only=True), which
# our numpy-carrying checkpoints fail. Our own files are trusted; restore the pre-2.6 default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HDF5_USE_FILE_LOCKING=FALSE

# h5py is missing from mural_osc: unpack the wheel that ships in this directory onto PYTHONPATH.
WHEELS=("$W"/wheels/*.whl)
[ "${#WHEELS[@]}" -ge 1 ] || fail "no h5py wheel in $W/wheels (nullglob: the array is empty)"
mkdir -p /tmp/pylib || fail "mkdir /tmp/pylib"
for whl in "${WHEELS[@]}"; do
  "$PY" -m zipfile -e "$whl" /tmp/pylib || fail "could not unpack $whl"
done
export PYTHONPATH=/tmp/pylib${PYTHONPATH:+:$PYTHONPATH}

echo "== env $(date)"; hostname; nvidia-smi -L
# Record the shell, not just the python. `set -u` treats an empty array expansion differently
# across bash versions, and this script expands possibly-empty arrays; bash 5.2 is safe locally
# but I cannot verify the node's version from the VM, so print it instead of assuming it.
echo "bash $BASH_VERSION"
"$PY" -c "import torch,h5py,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'h5py',h5py.__version__,'np',numpy.__version__)" || fail "env"
for d in "$REAL" "$SIM" "$CKD" "$CODE/Track_1/unet"; do [ -d "$d" ] || fail "missing $d"; done
[ -f "$W/check_pair.py" ] || fail "missing $W/check_pair.py"
echo "patch of record:"; sha256sum "$W"/errhead.patch "$CODE"/Track_1/unet/{train2.py,models_alt.py}

# sim-map cache: the repo default is /fs/scratch (invisible here); simreal.py reads RPDE_SIMCACHE.
export RPDE_SIMCACHE=/tmp/simcache
mkdir -p "$RPDE_SIMCACHE" && [ -w "$RPDE_SIMCACHE" ] || fail "simcache not writable"
"$PY" -c "import os,sys; sys.path.insert(0,'$CODE/Track_1/unet'); import simreal; assert simreal.CACHE==os.environ['RPDE_SIMCACHE'], 'override did not take: '+simreal.CACHE; print('sim cache', simreal.CACHE)" || fail "simcache"

# The fixed real input slice the ARM/CONTROL field fingerprint is read on (first 8 windows).
echo "== build val_unseen/val_seen from raw $(date)"
( cd "$VALD" && "$PY" "$CODE/Track_1/make_val.py" --data "$REAL" ) || fail "make_val"
[ -s "$VALD/val_unseen.npz" ] || fail "val_unseen.npz not written"

# ---- the recipe -----------------------------------------------------------------------------
# REPRODUCE.md 1.5, the F-FNO held-out twin (p2_ffno_l8_sim_s{1,2,3}.pth), copied flag for flag:
# --all-data dropped (so --exclude = model.VAL_TRAJ) and --sim-exclude-held added. Twins, not the
# all-data members, because the band's error field is fitted on the twins. Seed 1 for both arms.
SEED=${SEED:-1}                    # repo copy: seed from env. For pitzer submit write it LITERAL (SEED=2): no env reaches the job
# ---- Fen / qdre (Ilse 08:02Z, #Track-1-W-2): the ONLY other change vs Devi's qd_r_s1 ----------
# Held-out set = Track_1/re_split.py's three Re groups (val_relo, val_rehi, val_rein; 26 traj),
# taken by CALLING re_split.excluded() -- never a hand-written list -- so training sees 56 traj.
# --sim-exclude-held stays (Devi's recipe): sim at the held-out conditions is dropped too.
EXCL_STR=$("$PY" -c "import sys; sys.path.insert(0, '$CODE/Track_1'); import re_split; print(' '.join(re_split.excluded('$REAL')))") \
    || fail "re_split.excluded() failed"
read -r -a EXCL <<< "$EXCL_STR"
[ "${#EXCL[@]}" -eq 26 ] || fail "re_split.excluded() returned ${#EXCL[@]} trajectories, expected 26"
echo "held out (${#EXCL[@]}): ${EXCL[*]}"
RECIPE=(--data "$REAL" --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32
        --dropout 0.3 --var-head
        --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3
        --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0
        --stride 5 --batch 32 --lr 2e-4 --epochs "$EPOCHS" --wd 1e-5 --seed "$SEED"
        --sim "$SIM" --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5
        --sim-exclude-held --exclude "${EXCL[@]}")
# pinball_r: three seeds, each a paired (arm, control) run, in THREE SIBLING DIRECTORIES
# (pinball_r_s1/2/3) because `pitzer submit` passes NO environment: any ${VAR:-default} here
# would silently resolve to its default inside the job (presubmit.sh flags exactly that). So
# these are LITERALS, and the only thing that differs between the sibling directories is SEED.
TAU=0.85                           # LITERAL. Centre of the live bracket: Corin tau 0.83-0.90 (from achieved coverage), Jonas measuring the score-optimal quantile (pointing 0.82-0.85). Chosen BEFORE any submission.
QDT=0.1                            # LITERAL. qd_r (Jonas spec): temperature of the smoothed inside-indicator.
ARM=$CK/qdre_arm_s${SEED}.pth
CTL=$CK/pinball_control_s${SEED}.pth

train() {                          # train LABEL OUTFILE [extra flags...]
  local label=$1 out=$2; shift 2
  echo "== train $label $(date)"
  ( cd "$CODE/Track_1/unet" && "$PY" train2.py "${RECIPE[@]}" "$@" --out "$out" ) \
      > "$OUT/logs/train_$label.log" 2>&1
  local rc=$?
  grep -m2 -E "^(sim|params|git):" "$OUT/logs/train_$label.log" || true
  tail -3 "$OUT/logs/train_$label.log"
  [ $rc -eq 0 ] || fail "train $label rc=$rc (see logs/train_$label.log)"
  [ -s "$out" ] || fail "train $label wrote no checkpoint at $out"
}

# ONE job trains both -- that is the point of the request, and it is what ARMS defaults to.
# But a V100 job is capped at 360 minutes (pitzer skill) and this twin costs 4:24 per arm
# (REPRODUCE.md 7, `pz_p2simS1..S3`, measured on V100-32G), so the pair only fits inside one
# slot below about 250 epochs. At the full 400 the run has to be split, and the split costs the
# experiment NOTHING: the two arms are separate processes at the same --seed either way, so the
# pairing is a property of the seed and the detach, not of the time slot. Second half: fetch the
# first half's checkpoint into this directory's `pair/` (5 MB) and submit with ARMS=control.
# DEFAULT IS ONE ARM, deliberately. Both arms at 400 epochs is ~528 min against a 360-min cap,
# so "arm control" could never have been a correct default for a submitted job -- it would run
# out of wall clock in the second arm. Half a pair exits with PINBALL_HALF_OK, not the OK line.
ARMS="arm"                         # HARD-CODED to the FIRST half. The CONTROL IS REQUIRED
                                   # AGAIN: the head now shares the trunk gradient, so the field
                                   # moves and cannot be inferred from the arm. For the second
                                   # submission I
                                   # edit this line to "control" and say so in the message -- a
                                   # visible one-word diff beats an invisible submit-line flag.
PAIR=$W/pair                       # HARD-CODED (pitzer-manager msg 33a3eee1): the job mounts
                                   # this request directory at /work, so this is the only value
                                   # that can ever be right inside a job. No switch.
# Print the settings actually in force, AFTER they are all assigned. This echo lived in the
# env block until 2026-09-23, where it referenced $ARMS 52 lines before the assignment --
# under `set -u` that is a hard failure, and it killed the first smoke in 2 seconds.
echo "EFFECTIVE SETTINGS: EPOCHS=$EPOCHS  ARMS='$ARMS'  PAIR=$PAIR  SEED=$SEED  QDT=$QDT  (TAU unused by qd arms)"

for a in $ARMS; do
  case $a in
    arm)     train "arm_s${SEED}"     "$ARM" --err-head --err-loss qd --qd-temp "$QDT" ;;
    control) train "control_s${SEED}" "$CTL" ;;
    *)       fail "unknown arm '$a' (want 'arm', 'control', or both)" ;;
  esac
done

# Bring in the other half if this job trained only one and the pair was carried in. Copy, never
# symlink, and never let a missing file leave a plausible path in the variable: the guards below
# test the file itself, so an absent half is an absent half rather than an unreadable name.
for half in "qdre_arm_s${SEED}.pth:$ARM" "pinball_control_s${SEED}.pth:$CTL"; do
  src=$PAIR/${half%%:*}; dst=${half#*:}
  if [ ! -s "$dst" ] && [ -s "$src" ]; then
    cp "$src" "$dst" || fail "could not stage $src"
    echo "staged the other half from $src"
  fi
done
HAVE=()
[ -s "$ARM" ] && HAVE+=("$ARM")
[ -s "$CTL" ] && HAVE+=("$CTL")
[ "${#HAVE[@]}" -ge 1 ] || fail "no checkpoint to check (ARMS='$ARMS')"

# ---- THE ONE FLAG THAT WOULD SILENTLY BREAK THE PAIRING (Lior, msg a7d21599) -----------------
# _sam_ascent (train2.py:130-147) scales its perturbation by the GLOBAL parameter-space gradient
# norm. The err head's gradients are non-zero, so with --sam-rho > 0 that norm differs between
# ARM and CONTROL, the perturbation applied to the SHARED trunk points somewhere else, and the
# two diverge from step 1 -- detach or no detach. The recipe does not pass --sam-rho and the
# default is 0.0, but a default is not an invariant: assert it from what the checkpoints actually
# recorded, which is the only copy that can testify about the run that happened.
"$PY" - "${HAVE[@]}" <<'EOF' || fail "sam_rho invariant"
import sys, torch
for p in sys.argv[1:]:
    cfg = torch.load(p, map_location="cpu", weights_only=False).get("config", {})
    v = cfg.get("sam_rho", None)
    if v is None:
        sys.exit(f"FAIL: {p} records no sam_rho -- cannot verify the pairing invariant")
    if float(v) != 0.0:
        sys.exit(f"FAIL: {p} has sam_rho={v}; ARM and CONTROL cannot be bitwise paired")
    print(f"sam_rho=0.0 confirmed in {p}")
EOF

# ---- DID THE NEW LOSS ACTUALLY REACH THE HEAD?  (Lior, msg 720f5299 -- Kiran's VERDICT_MOVED)
# Every check above says the RECIPE is right. None of them says the head was WRITTEN TO. The head
# inits to zero weight, bias=log(0.3); if it is still EXACTLY zero after training, the pinball loss
# never reached it and that is a different failure from "the head learned nothing useful". A
# 1-epoch head that has moved a little is the healthy case (Bram's smoke3: sd ratio 0.0079); exact
# zero is a dead connection. Only asserted on blobs that carry the head.
"$PY" - "${HAVE[@]}" <<'EOF' || fail "head_moved invariant"
import sys, torch
for p in sys.argv[1:]:
    b = torch.load(p, map_location="cpu", weights_only=False)
    ks = [k for k in b["model_state_dict"] if k.startswith("head_logerr")]
    if not ks:
        print(f"{p}: control (no head_logerr) -- nothing to check")
        continue
    moved = 0.0
    for k in ks:
        t = b["model_state_dict"][k]
        if t.is_floating_point():
            moved = max(moved, float(t.abs().max()))
    if moved == 0.0:
        sys.exit(f"FAIL: {p} head_logerr is EXACTLY zero -- the pinball loss never reached the "
                 f"head (Kiran's zero-init deadlock shape), this is not 'training had no effect'")
    print(f"{p}: head_logerr moved, max|w| = {moved:.3e}")
EOF

# ---- qd_r: THE ARM MUST HAVE BEEN TRAINED UNDER THE QD LOSS (runs for HALF jobs too) --------
# The half-job path exits before the pair invariants below, so the one property this request
# exists to test -- which loss trained the head -- is asserted here, on every blob that has a head.
"$PY" - "${HAVE[@]}" <<'EOF' || fail "qd invariant"
import sys, torch
for p in sys.argv[1:]:
    b = torch.load(p, map_location="cpu", weights_only=False)
    if not b.get("err_head"):
        print(f"{p}: control (no err head) -- qd invariant not applicable"); continue
    if b.get("err_loss") != "qd":
        sys.exit(f"FAIL: {p} records err_loss={b.get('err_loss')!r}, expected 'qd'")
    if b.get("qd_temp") is None:
        sys.exit(f"FAIL: {p} records no qd_temp")
    if b.get("pinball_tau") is not None:
        sys.exit(f"FAIL: {p} records pinball_tau={b.get('pinball_tau')} on a qd arm")
    print(f"qd invariant OK: {p} err_loss=qd qd_temp={b.get('qd_temp')}")
EOF

# ---- keep, BEFORE the fingerprint ------------------------------------------------------------
# The checkpoints are the deliverable and they cost hours; copy them out before anything that is
# allowed to fail, so a failed fingerprint still hands back the weights that produced it.
cp "${HAVE[@]}" "$OUT/" || fail "could not copy the checkpoints to $OUT"
CKPTS=("$OUT"/*.pth)
[ "${#CKPTS[@]}" -eq "${#HAVE[@]}" ] || fail "copied ${#CKPTS[@]} checkpoints, expected ${#HAVE[@]}"
sha256sum "${CKPTS[@]}" | tee "$OUT/checkpoints.sha256"
for lg in "$OUT"/logs/train_*.log; do
  grep -E "^  epoch" "$lg" | tail -5 > "$OUT/epochs_$(basename "$lg" .log).txt"
done

# ---- THE COMPARISON REPORT (no longer a bitwise gate) ----------------------------------------
# REWRITTEN 2026-09-23. The err head now SHARES the trunk gradient, so the field DOES move and a
# "must be bitwise identical" assertion would fail this job by construction, every time. What
# replaces it is weaker but honest: report the differences, and hard-assert only the things that
# MUST still hold -- that the two runs differ in the intended flags and nothing else, and that
# the arm actually built the head. The accuracy shift is now a MEASURED QUANTITY the reading has
# to price, not a confound the script can rule out.
if [ ! -s "$ARM" ] || [ ! -s "$CTL" ]; then
  echo "== HALF JOB: ARMS='$ARMS' and no counterpart in $PAIR; the comparison needs both"
  echo "   fetch $(basename "${HAVE[0]}") into this request's pair/ and submit the other half"
  echo "=== out ==="; ls -la "$OUT"
  echo "== done $(date)"
  echo QD_HALF_OK
  exit 0
fi
need SHIP p2_ffno_l8_sim_s1.pth
echo "== comparison (shipped twin of record read in place: $SHIP) $(date)"
"$PY" "$W/check_pair.py" --arm "$ARM" --control "$CTL" --val "$VALD/val_unseen.npz" \
      --shipped "$SHIP" --unet "$CODE/Track_1/unet" 2>&1 | tee "$OUT/check_pair.txt"
echo "(check_pair exit ${PIPESTATUS[0]} -- NOT a failure condition any more: a non-zero field"
echo " difference is now EXPECTED and is the quantity the reading has to price.)"

# What must still be true, and is still worth failing on:
# The label can itself contain brackets -- smoke3 printed `FIELD_MAXDIFF[val_unseen[:8]] 0.07`,
# and my old pattern `\[[^]]*\]` stopped at the FIRST `]`, silently matching only 2 of the 3
# lines and failing the job on a count. Match the token at line start and take the last field;
# do not try to parse the label.
mapfile -t DIFFS < <(awk '/^(FIELD|WEIGHT)_MAXDIFF/ {print $NF}' "$OUT/check_pair.txt")
[ "${#DIFFS[@]}" -ge 3 ] || fail "expected >=3 MAXDIFF lines in check_pair.txt, got ${#DIFFS[@]}"
echo "field/weight differences reported: ${DIFFS[*]}"
# The criterion INVERTED when the head started sharing the trunk gradient, so assert the new one
# rather than leaving it written in REQUEST.md and unchecked -- a stated check that nothing
# executes is the failure mode this whole project has been correcting all night. After a real
# training step the ARM's field MUST differ from the CONTROL's: exactly 0 everywhere would mean
# the head's gradient never reached the trunk, i.e. the shared-gradient change silently did not
# take, and the run would be measuring the detached variant that ROUNDS.md:377 already refuted.
NONZERO=0
for d in "${DIFFS[@]}"; do case "$d" in 0|0.0|0.0e+00|0.000e+00) ;; *) NONZERO=1 ;; esac; done
[ "$NONZERO" -eq 1 ] || fail "every MAXDIFF is exactly 0 -- the err head is not reaching the trunk; \
this is the DETACHED variant, which ROUNDS.md:377-380 measures at corr 0.679, below the free \
baseline of 0.756. Do not read this run."
echo "field difference is non-zero, as the shared-gradient design requires"
# A GAP I CANNOT CLOSE WITH AN ASSERTION, so it is printed instead of silently passed. "Non-zero"
# also admits ~1e-7, which is what plain V100 run-to-run nondeterminism looks like (the repo
# enables no deterministic algorithms) -- that would pass the test above while still meaning the
# gradient never reached the trunk. A real shared-gradient difference should be orders larger. I
# will not invent a threshold I cannot defend, so: flag it and let a human judge.
BIG=0
for d in "${DIFFS[@]}"; do
  case "$d" in 0|0.0|0.0e+00|0.000e+00) ;; *)
    awk -v v="$d" 'BEGIN{exit !(v+0 > 1e-5)}' && BIG=1 ;;
  esac
done
[ "$BIG" -eq 1 ] || echo "WARNING: every non-zero difference is below 1e-5, which is the size of \
V100 run-to-run nondeterminism. This PASSES the check above but is NOT evidence that the head \
reached the trunk. Read check_pair.txt before trusting this pair."
"$PY" - "$ARM" "$CTL" <<'EOF' || fail "config invariant"
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ca, cc = a.get("config", {}), c.get("config", {})
if not a.get("err_head"):  sys.exit("FAIL: the ARM blob does not record err_head=True")
if c.get("err_head"):      sys.exit("FAIL: the CONTROL blob records err_head=True")
# qd_r: the ARM must record err_loss="qd"; the CONTROL (pinball_r_sN_ctl) records no err head.
if a.get("err_loss") != "qd":
    sys.exit(f"FAIL: the ARM blob records err_loss={a.get('err_loss')!r}, not 'qd'")
if c.get("err_head"):
    sys.exit("FAIL: the CONTROL blob has an err head -- it must be a plain headless run")
print(f"qd invariant OK: ARM err_loss=qd qd_temp={a.get('qd_temp')}, CONTROL headless")
if ca.get("err_detach") or cc.get("err_detach"):
    sys.exit("FAIL: err_detach is set; this directory is the SHARED-gradient arm (ROUNDS.md:377)")
allowed = {"err_head", "lam_e", "err_detach", "pinball_tau", "err_loss", "qd_temp", "out", "train2_argv", "git"}
diff = {k for k in set(ca) | set(cc) if ca.get(k) != cc.get(k)} - allowed
if diff: sys.exit(f"FAIL: ARM and CONTROL differ in unintended config keys: {sorted(diff)}")
ks = [k for k in a["model_state_dict"] if k.startswith("head_logerr")]
if not ks: sys.exit("FAIL: the ARM has no head_logerr.* tensors")
print(f"config invariant OK: {len(ks)} head_logerr tensors on the ARM, "
      f"none on the CONTROL, and no unintended config difference")
EOF

echo "=== out ==="; ls -la "$OUT"
echo "== done $(date)"
echo QD_JOB_OK
