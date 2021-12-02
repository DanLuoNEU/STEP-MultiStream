#!/bin/bash
cd ../

data_root="/data/CLASP-DATA/CLASP2-STEP/data"
save_root="/data/Dan/ava_v2_1/cache/"
resnet_pretrain="pretrained/resnet50.pth"
kinetics_pretrain="pretrained/mvf_R50_8x8.pth"

name="Cls-mvf-test"
resume_path="Auto"
num_classes=3

T=3
max_iter=1    # index starts from 1
iterative_mode="spatial"
pool_mode="align"
pool_size=14
# training schedule
num_workers=4 # 16
batch_size=4
optimizer="adam"
base_lr=1e-4 # 5e-5
det_lr0=1e-3 # 5e-4
det_lr=1e-3 # 5e-4
max_epochs=20 # 10
# save_step=22930
save_step=251
print_step=20
scheduler="cosine"
milestones="-1"
warmup_iters=100 # 50
# losses
dropout=0.3
fc_dim=256
# data augmentation / normalization
scale_norm=2
do_flip="True"
do_crop="True"
do_photometric="True"
do_erase="True"
freeze_affine="True"
freeze_stats="True"


/home/dan/anaconda3/envs/py36pt110/bin/python train_cls_mvf.py --data_root $data_root --save_root $save_root \
    --name $name --resume_path $resume_path \
    --resnet_pretrain $resnet_pretrain --kinetics_pretrain $kinetics_pretrain \
    --max_iter $max_iter --T $T --iterative_mode $iterative_mode \
    --pool_mode $pool_mode --pool_size $pool_size --save_step $save_step \
    --num_workers $num_workers --max_epochs $max_epochs --batch_size $batch_size --print_step $print_step \
    --optimizer $optimizer --base_lr $base_lr --det_lr $det_lr --det_lr0 $det_lr0 --milestones $milestones \
    --scale_norm $scale_norm --do_flip $do_flip --do_crop $do_crop --do_photometric $do_photometric --do_erase $do_erase \
    --fc_dim $fc_dim --dropout $dropout  --scheduler $scheduler --warmup_iters $warmup_iters \
    --freeze_affine $freeze_affine --freeze_stats $freeze_stats --num_classes $num_classes
