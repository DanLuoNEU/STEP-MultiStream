#!/bin/bash

# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

cd ../

data_root="/data/truppr/ava/"
save_root="/data/Dan/ava_v2_1/cache"
kinetics_pretrain="pretrained/i3d_kinetics.pth|pretrained/i3d_flow_kinetics.pth"

name="Cls_2s"
base_net="i3d"
det_net="two_branch"
resume_path="Auto"
## No --fp16 for this, this model is pretrained with fp32
# resume_path="pretrained/ava_cls.pth"

input_type="2s"
T=9
max_iter=1    # index starts from 1
iterative_mode="spatial"
pool_mode="align"
pool_size=7

# training schedule
num_workers=8
max_epochs=15 #10 #14
batch_size=4
optimizer="adam"
base_lr=5e-5
det_lr0=1e-4
det_lr=5e-5 #5e-5
# save_step=22930
save_step=251
print_step=20
scheduler="cosine"
milestones="-1"
warmup_iters=50

# losses
dropout=0.3
fc_dim=256

# data augmentation / normalization
scale_norm=0    # for i3d_of
do_flip="True"
do_crop="True"
do_photometric="False" # No need for optical flow
do_erase="True"
freeze_affine="True"
freeze_stats="True"


python train_cls_2s.py --data_root $data_root --save_root $save_root \
    --name $name --resume_path $resume_path --kinetics_pretrain $kinetics_pretrain \
    --base_net $base_net --det_net $det_net --max_iter $max_iter --T $T \
    --iterative_mode $iterative_mode --input_type $input_type\
    --pool_mode $pool_mode --pool_size $pool_size --save_step $save_step \
    --num_workers $num_workers --max_epochs $max_epochs --batch_size $batch_size --print_step $print_step \
    --optimizer $optimizer --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 --milestones $milestones \
    --scale_norm $scale_norm --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \
    --fc_dim $fc_dim --dropout $dropout  --scheduler $scheduler --warmup_iters $warmup_iters \
    --freeze_affine $freeze_affine --freeze_stats $freeze_stats --num_classes 3
