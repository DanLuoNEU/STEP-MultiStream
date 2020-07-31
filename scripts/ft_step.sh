#!/bin/bash

# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


cd ../

data_root="/data/truppr/ava/"
save_root="/data/truppr/ava/cache/"
# pretrain_path="/data/truppr/ava/cache/Cls-max1-i3d-two_branch/checkpoint_best.pth"
pretrain_path="/data/truppr/ava/cache/20200408/Cls-max1-i3d-two_branch/checkpoint_best.pth"

name="STEP"
base_net="i3d"
det_net="two_branch"
# resume_path="Auto"
resume_path="Best"

T=3
max_iter=3 #3    # index starts from 1
iterative_mode="temporal"
anchor_mode="1"
temporal_mode="predict"
pool_mode="align"
pool_size=7

# training schedule
num_workers=16
max_epochs=50 # 10 # 14
batch_size=4 # 8
optimizer="adam"
base_lr=7.5e-5
det_lr0=1.5e-4
det_lr=7.5e-4
save_step=250 # 11465
print_step=20 # 500
scheduler="cosine"
milestones="-1"
warmup_iters=50 # 1000

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
freeze_affine="True"
freeze_stats="True"


/home/dan/anaconda3/envs/py36pt110/bin/python ft.py --data_root $data_root --save_root $save_root --num_classes 3 \
    --name $name --pretrain_path $pretrain_path --resume_path $resume_path \
    --base_net $base_net --det_net $det_net --max_iter $max_iter --T $T \
    --iterative_mode $iterative_mode --anchor_mode $anchor_mode --anchor_mode $anchor_mode --temporal_mode $temporal_mode \
    --pool_mode $pool_mode --pool_size $pool_size --save_step $save_step --topk $topk --evaluate_topk $evaluate_topk \
    --num_workers $num_workers --max_epochs $max_epochs --batch_size $batch_size --print_step $print_step \
    --optimizer $optimizer --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 --milestones $milestones \
    --scale_norm $scale_norm --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \
    --fc_dim $fc_dim --dropout $dropout --NUM_SAMPLE $NUM_SAMPLE --scheduler $scheduler --warmup_iters $warmup_iters \
    --cls_thresh $cls_thresh --reg_thresh $reg_thresh --max_pos_num $max_pos_num --neg_ratio $neg_ratio \
    --freeze_affine $freeze_affine --freeze_stats $freeze_stats --lambda_reg $lambda_reg --lambda_neighbor $lambda_neighbor --inflate 
