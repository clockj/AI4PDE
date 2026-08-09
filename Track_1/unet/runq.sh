#!/bin/bash
# Driver that re-reads `queue.txt` before every run, so the queue can be reordered, extended or
# trimmed while it is running -- editing a live bash script is not safe, editing a data file is.
#
# queue.txt: one arm per line, "<out.pth>  <extra train2.py flags>". Blank lines and # ignored.
# A line is moved to done.txt as soon as it starts, so a restart never repeats an arm.
set -u
PY=/home/clockj/miniconda3/envs/realpde/bin/python
cd "$(dirname "$0")"
# The SHIPPED recipe, read back off fno_grad.pth's stored config -- not train2.py's argparse
# defaults, which differ on three flags that all matter:
#   --lam-v      default 1.0  vs  0.3   (log-std MSE weight)
#   --lam-t      default 0.5  vs  0.15  (TKE loss weight)
#   --alpha-train default 1.0 vs  0.0   (whether the FIELD loss sees the sigma-rescaled field)
# Every member of the shipped 8 uses 0.3 / 0.15 / 0.0. Taking the defaults instead triples the
# variance-loss weight and couples the field loss to an untrained sigma; it collapses the model
# (rel-L2 0.19 against 0.112). Change one thing per arm, and this line is the one thing that
# must never be the thing that changed.
BASE="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4 --stride 5 --epochs 150 \
      --lam-grad 1.0 --loss mse --var-head --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0"

while :; do
  line=$(grep -vE '^\s*(#|$)' queue.txt | head -1)
  [ -z "$line" ] && break
  # Not `cut -d' ' -f2-`: cut echoes the WHOLE line when the delimiter is absent, so an arm
  # with no extra flags passed its own filename to train2.py as a positional argument.
  out=${line%% *}
  args=${line#"$out"}
  grep -vxF "$line" queue.txt > queue.tmp && mv queue.tmp queue.txt
  echo "$line" >> done.txt
  echo "### $out START $(date +%T)"
  $PY train2.py $BASE $args --out "$out" > "log_${out%.pth}.txt" 2>&1
  echo "### $out DONE rc=$? $(date +%T)"
done
echo "ALL DONE $(date +%T)"
