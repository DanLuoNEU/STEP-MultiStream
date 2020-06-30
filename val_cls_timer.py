"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
from collections import OrderedDict
import time
from datetime import datetime
#from tensorboardX import SummaryWriter
import glob

from config import parse_config
from models import BaseNet, ROINet, TwoBranchNet, ContextNet
from utils.utils import inference, train_select, AverageMeter, get_gpu_memory, select_proposals, Timer
from utils.tube_utils import flatten_tubes, valid_tubes
from utils.solver import WarmupCosineLR, WarmupStepLR, get_params
from data.ava_cls import AVADataset, detection_collate, WIDTH, HEIGHT
from data.augmentations import TubeAugmentation, BaseTransform
from utils.eval_utils import ava_evaluation
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap


args = parse_config()
args.max_iter = 1    # only 1 step for classification pretraining

try:
    import apex
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    print ('Warning: If you want to use fp16, please apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    args.fp16 = False
    pass

args.image_size = (WIDTH, HEIGHT)
label_dict = {}
if args.num_classes == 3:
    label_map = os.path.join(args.data_root, 'label/ava_finetune.pbtxt')
    categories, class_whitelist = read_labelmap(open(label_map, 'r'))
    classes = [(val['id'], val['name']) for val in categories]
    id2class = {c[0]: c[1] for c in classes}    # gt class id (1~3) --> class name
    for i, c in enumerate(sorted(list(class_whitelist))):
        label_dict[i] = c
elif args.num_classes == 60:
    label_map = os.path.join(args.data_root, 'label/ava_action_list_v2.1_for_activitynet_2018.pbtxt')
    categories, class_whitelist = read_labelmap(open(label_map, 'r'))
    classes = [(val['id'], val['name']) for val in categories]
    id2class = {c[0]: c[1] for c in classes}    # gt class id (1~80) --> class name
    for i, c in enumerate(sorted(list(class_whitelist))):
        label_dict[i] = c
else:
    for i in range(80):
        label_dict[i] = i+1

## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

#args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"  # specify which GPU(s) to be used
gpu_count = torch.cuda.device_count()
torch.backends.cudnn.benchmark=True

def main():

    args.exp_name = '{}-max{}-{}-{}'.format(args.name, args.max_iter, args.base_net, args.det_net)
    args.save_root = os.path.join(args.save_root, args.exp_name+'/')

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    log_name = args.save_root+"training-"+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')+".log"
    log_file = open(log_name, "w", 1)
    log_file.write(args.exp_name+'\n')

    ################ DataLoader setup #################

    print('Loading Dataset...')
    augmentation = TubeAugmentation(args.image_size, args.means, args.stds, do_flip=args.do_flip, do_crop=args.do_crop, do_photometric=args.do_photometric, scale=args.scale_norm, do_erase=args.do_erase)
    log_file.write("Data augmentation: "+ str(augmentation))

    train_dataset = AVADataset(args.data_root, 'train', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, augmentation, stride=1, num_classes=args.num_classes, foreground_only=True)
    val_dataset = AVADataset(args.data_root, 'val', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, BaseTransform(args.image_size, args.means, args.stds,args.scale_norm), stride=1, num_classes=args.num_classes, foreground_only=True)

    if args.milestones[0] == -1:
        args.milestones = [int(np.ceil(len(train_dataset) / args.batch_size) * args.max_epochs)]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 1, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    log_file.write("Training size: " + str(len(train_dataset)) + "\n")
    log_file.write("Validation size: " + str(len(val_dataset)) + "\n")
    print('Training STEP on ', train_dataset.name)

    ################ define models #################

    nets = OrderedDict()
    # backbone network
    nets['base_net'] = BaseNet(args)
    # ROI pooling
    nets['roi_net'] = ROINet(args.pool_mode, args.pool_size)

    # detection network
    ic_t = 0
    for i in range(args.max_iter):
        if args.det_net == "two_branch":
            nets['det_net%d' % i] = TwoBranchNet(args, cls_only=True)
            ic_t = nets['det_net%d' % i].global_cls.in_channels
        else:
            raise NotImplementedError
    if not args.no_context:
        # context branch
        nets['context_net'] = ContextNet(args)
    
    for key in nets:
        nets[key] = nets[key].cuda()

    ################ Training setup #################
    # DataParallel is used
    nets['base_net'] = torch.nn.DataParallel(nets['base_net'])
    if not args.no_context:
        nets['context_net'] = torch.nn.DataParallel(nets['context_net'])
    for i in range(args.max_iter):
        nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
        nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

    ############ Pretrain & Resume ###########

    # resume trained model if needed
    if args.resume_path is not None:
        if args.resume_path.lower() == "best":
            model_path = args.save_root+'/checkpoint_best.pth'
            if not os.path.isfile(model_path):
                model_path = None
        elif args.resume_path.lower() == "auto":
            # automatically get the latest model
            model_paths = glob.glob(os.path.join(args.save_root, 'checkpoint_*.pth'))
            best_path =  os.path.join(args.save_root, 'checkpoint_best.pth')
            if best_path in model_paths:
                model_paths.remove(best_path)
            if len(model_paths):
                iters = [int(val.split('_')[-1].split('.')[0]) for val in model_paths]
                model_path = model_paths[np.argmax(iters)]
            else:
                model_path = None
        else:
            model_path = args.resume_path
            if not os.path.isfile(model_path):
                raise ValueError("Resume model not found!", args.resume_path)

        if model_path is not None:
            print ("Resuming trained model from %s" % model_path)
            checkpoint = torch.load(model_path, map_location='cpu')

            nets['base_net'].load_state_dict(checkpoint['base_net'])
            if not args.no_context and 'context_net' in checkpoint:
                nets['context_net'].load_state_dict(checkpoint['context_net'])
            for i in range(args.max_iter):
                if args.num_classes != len(checkpoint['det_net%d' % i]['global_cls.bias']):
                    nets['det_net%d' % i].global_cls = nn.Conv3d(ic_t, 60, (1,1,1), bias=True)
                nets['det_net%d' % i].load_state_dict(checkpoint['det_net%d' % i])
            
            # Finetune 'ava_cls.pth' provided by STEP
            if args.num_classes != len(checkpoint['det_net%d' % i]['global_cls.bias']):
                print(f" >>>>>> Finetuning from {len(checkpoint['det_net%d' % i]['global_cls.bias'])} to {args.num_classes} <<<<<< ")
                for p in nets['base_net'].parameters(): p.requires_grad = False
                if not args.no_context:
                    for p in nets['context_net'].parameters(): p.requires_grad = False
                for i in range(args.max_iter):
                    for p in nets['det_net%d' % i].parameters(): p.requires_grad = False
                    nets['det_net%d' % i].global_cls = nn.Conv3d(ic_t, args.num_classes, (1,1,1), bias=True)
                    nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
                    nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

            if args.num_classes == len(checkpoint['det_net%d' % i]['global_cls.bias']):
                args.start_iteration = checkpoint['iteration']
                if checkpoint['iteration'] % int(np.ceil(len(train_dataset)/args.batch_size)) == 0:
                    args.start_epochs = checkpoint['epochs']
                else:
                    args.start_epochs = checkpoint['epochs'] - 1

            del checkpoint
            torch.cuda.empty_cache()

    ################ Optimizer and Scheduler setup #################

    params = get_params(nets, args)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.det_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.det_lr)
        # optimizer = optim.Adam(filter(lambda x : x.requires_grad, nets['det_net0'].parameters()),lr=args.det_lr, weight_decay=args.weight_decay) # Tried, but didnt work
    else:
        raise NotImplementedError

    if args.scheduler == "cosine":
        scheduler = WarmupCosineLR(optimizer, args.milestones, args.min_ratio, args.cycle_decay, args.warmup_iters)
    else:
        scheduler = WarmupStepLR(optimizer, args.milestones, args.warmup_iters)

    # Initialize AMP if needed
    if args.fp16:
        models, optimizer = amp.initialize([net for _,net in nets.items()], optimizer, opt_level="O1")
        for i, key in enumerate(nets):
            nets[key] = models[i]

    ######################################################

    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')

    for i in range(args.max_iter):
        log_file.write(str(nets['det_net%d' % i])+'\n\n')
    
    params_num = [sum(p.numel() for p in nets['base_net'].parameters())]
    if not args.no_context:
        params_num.append(sum(p.numel() for p in nets['context_net'].parameters()))
    for i in range(args.max_iter):
        params_num.append(sum(p.numel() for p in nets['det_net%d' % i].parameters()))
    print("Number of parameters: "+str(params_num)+'\n\n')

    # Start training
    train(args, nets, optimizer, scheduler, train_dataloader, val_dataloader, log_file)


def train(args, nets, optimizer, scheduler, train_dataloader, val_dataloader, log_file):

    for _, net in nets.items():
        net.train()

    # loss counters
    batch_time = AverageMeter(200)
    losses = AverageMeter(200)

#    writer = SummaryWriter(args.save_root+"summary"+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    ################ Training loop #################

    best_mAP = 0.    # best validation mAP so far

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    epochs = args.start_epochs
    iteration = args.start_iteration
    epoch_size = int(np.ceil(len(train_dataloader.dataset) / args.batch_size))

    while epochs < 10:
        ##### Validation at the end of each epoch #####

        torch.cuda.synchronize()
        tvs = time.perf_counter()
    
        for _, net in nets.items():
            net.eval() # switch net to evaluation mode
        print('Validating at ', iteration)
        all_metrics = validate(args, val_dataloader, nets, iteration, iou_thresh=args.iou_thresh)
    
        prt_str = ''
        for i in range(args.max_iter):
            prt_str += 'Iter '+str(i+1)+': MEANAP =>'+str(all_metrics[i]['PascalBoxes_Precision/mAP@0.5IOU'])+'\n'
        print(prt_str)
        log_file.write(prt_str)
    
        log_file.write("Best MEANAP so far => {}\n".format(best_mAP))
        for i in class_whitelist:
            log_file.write("({}) {}: {}\n".format(i,id2class[i], 
                all_metrics[-1]["PascalBoxes_PerformanceByCategory/AP@0.5IOU/{}".format(id2class[i])]))    
              
        for _, net in nets.items():
            net.train() # switch net to training mode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prt_str2 = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
        print(prt_str2)
        log_file.write(prt_str2)

        epochs += 1


    log_file.close()


def validate(args, val_dataloader, nets, iteration=0, iou_thresh=0.5):
    """
    Test the model on validation set
    """

    # write results to files for evaluation
    output_files = []
    fouts = []
    for i in range(args.max_iter):
        output_file = args.save_root+'val_result-'+str(iteration)+'-iter'+str(i+1)+'.csv'
        output_files.append(output_file)
        f = open(output_file, 'w')
        fouts.append(f)

    gt_file = args.save_root+'val_gt.csv'
    fout = open(gt_file, 'w')

    base_timer = Timer("base", logger=None)
    context_timer = Timer("context", logger=None)
    twobranch_timer = Timer("2b", logger=None)
    rest_timer = Timer("rest", logger=None)
    with torch.no_grad():    # for evaluation
        for num, (images, targets, tubes, infos) in enumerate(val_dataloader):
            print(images.shape, len(tubes))
            for b in range(len(infos)):
                for n in range(len(infos[b]['boxes'])):
                    mid = int(len(infos[b]['boxes'][n])/2)
                    box = infos[b]['boxes'][n][mid]
                    labels = infos[b]['labels'][n][mid]
                    for label in labels:
                        fout.write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6}\n'.format(
                                    infos[b]['video_name'],
                                    infos[b]['fid'],
                                    box[0], box[1], box[2], box[3],
                                    label))

            _, _, channels, height, width = images.size()
            images = images.cuda()

            # get conv features
            base_timer.start()
            conv_feat = nets['base_net'](images)
            base_timer.stop()
            context_feat = None
            if not args.no_context:
                context_timer.start()
                context_feat = nets['context_net'](conv_feat)
                context_timer.stop()

            ############## Inference ##############

            T_start = 0
            T_length = args.T

            selected_tubes = []
            rest_timer.start()
            # use target boxes as tubes
            for b in range(len(targets)):
                cur_tubes = tubes[b]
                selected_tubes.append(cur_tubes)

            # flatten list of tubes
            flat_tubes, tubes_nums = flatten_tubes(selected_tubes, batch_idx=True)    # add batch_idx for ROI pooling
            flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat)    # gpu:0 for ROI pooling

            # ROI Pooling
            pooled_feat = nets['roi_net'](conv_feat[:, T_start:T_start+T_length].contiguous(), flat_tubes)
            _,C,W,H = pooled_feat.size()
            pooled_feat = pooled_feat.view(-1, T_length, pooled_feat.size(1), W, H)

            temp_context_feat = None
            if not args.no_context:
                temp_context_feat = torch.zeros((pooled_feat.size(0),context_feat.size(1),T_length,1,1)).to(context_feat)
                for p in range(pooled_feat.size(0)):
                    temp_context_feat[p] = context_feat[int(flat_tubes[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()
            rest_timer.stop()

            twobranch_timer.start()
            global_prob, _,_,_, _,_,_ = nets['det_net0'](pooled_feat, context_feat=temp_context_feat, tubes=None, targets=None)
            twobranch_timer.stop()
            #################### Evaluation #################
            rest_timer.start()
            # loop for each batch
            tubes_count = 0
            for b in range(len(tubes_nums)):
                info = infos[b]
                seq_start = tubes_count
                tubes_count = tubes_count + tubes_nums[b]
    
                cur_pred_prob = global_prob[seq_start:seq_start+tubes_nums[b]]
                cur_pred_tubes = flat_tubes[seq_start:seq_start+tubes_nums[b]][:,int(flat_tubes.shape[1]/2),1:]

                # loop for each foreground class
                for cl_ind in range(args.num_classes):
                    scores = cur_pred_prob[:, cl_ind].squeeze().view(-1)
                    c_mask = scores.gt(args.conf_thresh) # greater than minmum threshold

                    scores = scores[c_mask]
                    if len(scores) == 0:
                        continue
                    boxes = cur_pred_tubes.clone()
                    try:
                        l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    except:
                        pdb.set_trace()
                    boxes = boxes[l_mask].view(-1, 4)
                    scores = scores.cpu().numpy()
                    boxes = boxes.cpu().numpy()
                    boxes[:, ::2] /= width
                    boxes[:, 1::2] /= height

                    # write to files
                    for n in range(boxes.shape[0]):
                        fouts[i].write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6},{7:.4}\n'.format(
                                                info['video_name'],
                                                info['fid'],
                                                boxes[n,0],boxes[n,1],boxes[n,2],boxes[n,3],
                                                label_dict[cl_ind],
                                                scores[n]))
            rest_timer.stop()

    print(f"base avg time: {(Timer.timers['base']/len(val_dataloader)):0.4f} seconds")
    print(f"context avg time: {(Timer.timers['context']/len(val_dataloader)):0.4f} seconds")
    print(f"2b avg time: {(Timer.timers['2b']/len(val_dataloader)):0.4f} seconds")
    print(f"rest avg time: {(Timer.timers['rest']/len(val_dataloader)):0.4f} seconds")

    fout.close()

    all_metrics = []
    for i in range(args.max_iter):
        fouts[i].close()

        metrics = ava_evaluation(os.path.join(args.data_root, 'label/'), output_files[i], gt_file)
        all_metrics.append(metrics)
    
    return all_metrics


if __name__ == '__main__':
    main()

