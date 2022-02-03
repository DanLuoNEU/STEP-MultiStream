#!/bin/bash

# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

cd ../

name="20220127-Cls-flow-kinetics400_KRImixed-no_context-3cls_tcmv-T3i1-fps10"
input_type="flow"
num_classes=3
fps=10
max_iter=1    # index starts from 1
T=3 # 9
resume_path="Auto"
# resume_path="/data/Dan/ava_v2_1/cache/Cls-kinetics400_KRImixed-no_context-2cls-T9i1-max1-i3d-two_branch/checkpoint_best.pth" # for 2cls
# resume_path="pretrained/ava_cls.pth"

data_root="/data/CLASP-DATA/CLASP2-STEP/data"
save_root="/data/Dan/CLASP_paper/exp_cache/"
kinetics_pretrain="pretrained/i3d_kinetics.pth+pretrained/i3d_flow_kinetics.pth"
base_net="i3d"
det_net="two_branch"
iterative_mode="spatial"
pool_mode="align"
pool_size=7
## No --fp16 for this, this model is pretrained with fp32

# training schedule
num_workers=8 # 16
max_epochs=10 # 14
batch_size=8
optimizer="adam"
base_lr=5e-5
det_lr0=1e-4
det_lr=5e-4
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
scale_norm=0    # [-1,1] for i3d, only matters in augmentation to get [-1,1] input images
                # handled in augmentations_multi_stream with dignity, 0(best) or 2 doesn't matter when .flo saved in [-1,1]
do_flip="True"
do_crop="True"
do_photometric="False" # No photometric augmentation for Flow stream
do_erase="True"
freeze_affine="True"
freeze_stats="True"


/home/dan/anaconda3/envs/py36pt110/bin/python train_cls_single_stream.py --name $name \
    --data_root $data_root --save_root $save_root \
    --resume_path $resume_path --kinetics_pretrain $kinetics_pretrain \
    --base_net $base_net --det_net $det_net \
    --max_iter $max_iter --T $T --num_classes $num_classes \
    --iterative_mode $iterative_mode --pool_mode $pool_mode --pool_size $pool_size \
    --optimizer $optimizer --batch_size $batch_size \
    --scheduler $scheduler --warmup_iters $warmup_iters --milestones $milestones \
    --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 \
    --fc_dim $fc_dim --dropout $dropout --freeze_affine $freeze_affine --freeze_stats $freeze_stats \
    --input_type $input_type --fps $fps --num_workers $num_workers --scale_norm $scale_norm \
    --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \
    --max_epochs $max_epochs --print_step $print_step --save_step $save_step
    # --model_ft
