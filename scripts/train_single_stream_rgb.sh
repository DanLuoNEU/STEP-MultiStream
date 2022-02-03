#!/bin/bash
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
cd ../
name="20220127-STEP-rgb-kinetics400_KRImixed-no_context-3cls_tcmv-T3i1-fps10-max1-i3d-two_branch"
# classification pretrain_path
pretrain_path="/data/Dan/CLASP_paper/exp_cache/20220126-Cls-rgb-kinetics400_KRImixed-no_context-3cls_tcmv-T3i1-fps10-max1-i3d-two_branch/checkpoint_best.pth"
resume_path="Auto"
input_type="rgb"
num_classes=3
T=1 # 3
fps=10 # PVD(ori 15, demo 10) |  KRI 30

data_root="/data/CLASP-DATA/CLASP2-STEP/data"
save_root="/data/Dan/CLASP_paper/exp_cache/"
kinetics_pretrain="pretrained/i3d_kinetics.pth+pretrained/i3d_flow_kinetics.pth"
base_net="i3d"
det_net="two_branch"
max_iter=3   # index starts from 1
iterative_mode="temporal"
anchor_mode="1"
temporal_mode="predict"
pool_mode="align"
pool_size=7
# training schedule
num_workers=4
max_epochs=10 # 10
batch_size=4 # 8
optimizer="adam"
base_lr=7.5e-5
det_lr0=1.5e-4
det_lr=7.5e-4
save_step=250 # 11465
print_step=20 # 500
scheduler="cosine"
milestones="-1"
warmup_iters=300 # 1000

# losses
dropout=0.3
fc_dim=256
lambda_reg=5
lambda_neighbor=1
cls_thresh="0.2,0.35,0.5"
reg_thresh="0.2,0.35,0.5"
max_pos_num=5
neg_ratio=2
NUM_SAMPLE=-1
topk=300
evaluate_topk=300

# data augmentation / normalization
scale_norm=2    # for i3d
do_flip="True"
do_crop="True"
do_photometric="True"
do_erase="True"
freeze_affine="True" # do not update BN
freeze_stats="True" # do not update BN statistics


CUDA_VISIBLE_DEVICES=0,3 /home/dan/anaconda3/envs/py36pt110/bin/python train_single_stream.py --data_root $data_root --save_root $save_root \
    --kinetics_pretrain $kinetics_pretrain --pretrain_path $pretrain_path --resume_path $resume_path \
    --name $name --input_type $input_type --num_classes $num_classes \
    --base_net $base_net --det_net $det_net --max_iter $max_iter --T $T --fps $fps\
    --iterative_mode $iterative_mode --anchor_mode $anchor_mode --temporal_mode $temporal_mode \
    --pool_mode $pool_mode --pool_size $pool_size --save_step $save_step --topk $topk --evaluate_topk $evaluate_topk \
    --num_workers $num_workers --max_epochs $max_epochs --batch_size $batch_size --print_step $print_step \
    --optimizer $optimizer --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 --milestones $milestones \
    --scale_norm $scale_norm --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \
    --fc_dim $fc_dim --dropout $dropout --NUM_SAMPLE $NUM_SAMPLE --scheduler $scheduler --warmup_iters $warmup_iters \
    --cls_thresh $cls_thresh --reg_thresh $reg_thresh --max_pos_num $max_pos_num --neg_ratio $neg_ratio \
    --freeze_affine $freeze_affine --freeze_stats $freeze_stats --lambda_reg $lambda_reg --lambda_neighbor $lambda_neighbor 
    # --model_ft
