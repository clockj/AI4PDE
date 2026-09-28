#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c
cd $REPO/Track_2/stream_c
$PY dump_twins.py --split unseen --out /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/twins_unseen.npz --ckpt /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/p2_ffno_l8_sim_s1.pth /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/p2_ffno_l8_sim_s2.pth /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/p2_ffno_l8_sim_s3.pth /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/arch_vit_d384.pth /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/arch_vit_cmb3sim.pth
$PY dump_twins.py --split all --fp16 --out /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/twins_all.npz --ckpt /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/p2_ffno_l8_sim_s1.pth /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/arch_vit_d384.pth
echo DONE_DUMP
