#!/bin/bash

# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


cd ../

data_root="/data/CLASP-DATA/CLASP2-STEP/data"
save_root="/data/Dan/ava_v2_1/cache/"
# classification pretrain_path
resnet_pretrain="pretrained/resnet50.pth"
kinetics_pretrain="pretrained/mvf_R50_8x8.pth"
pretrain_path="/data/Dan/ava_v2_1/cache/Cls-mvf-224x224-pool14-lr5e-4-max1/checkpoint_best.pth"

name="STEP-mvf-test"
resume_path="Auto"
num_classes=3

T=1
max_iter=3   # index starts from 1
iterative_mode="temporal"
anchor_mode="1"
temporal_mode="predict"
pool_mode="align"
pool_size=14

# training schedule
num_workers=4
max_epochs=10 # 15
batch_size=1 # 8
optimizer="adam"
base_lr=1e-4 #7.5e-5
det_lr0=1e-3 #1.5e-4
det_lr=1e-3 #7.5e-4
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
scale_norm=2  
do_flip="True"
do_crop="True"
do_photometric="True"
do_erase="True"
freeze_affine="True" # do not update BN
freeze_stats="True" # do not update BN statistics


/home/dan/anaconda3/envs/py36pt110/bin/python train_ft.py --data_root $data_root --save_root $save_root \
    --resnet_pretrain $resnet_pretrain --kinetics_pretrain $kinetics_pretrain --pretrain_path $pretrain_path \
    --name $name --resume_path $resume_path --num_classes $num_classes --T $T --max_iter $max_iter \
    --iterative_mode $iterative_mode --anchor_mode $anchor_mode --temporal_mode $temporal_mode \
    --pool_mode $pool_mode --pool_size $pool_size \
    --num_workers $num_workers --max_epochs $max_epochs --batch_size $batch_size \
    --optimizer $optimizer --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 \
    --save_step $save_step --print_step $print_step --scheduler $scheduler --milestones $milestones --warmup_iters $warmup_iters \
    --dropout $dropout --fc_dim $fc_dim --lambda_reg $lambda_reg --lambda_neighbor $lambda_neighbor \
    --cls_thresh $cls_thresh --reg_thresh $reg_thresh --max_pos_num $max_pos_num --neg_ratio $neg_ratio \
    --NUM_SAMPLE $NUM_SAMPLE --topk $topk --evaluate_topk $evaluate_topk \
    --scale_norm $scale_norm --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \    
    --freeze_affine $freeze_affine --freeze_stats $freeze_stats
