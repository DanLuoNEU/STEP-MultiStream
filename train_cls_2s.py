"""
04/30/2020, Dan
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
import glob
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
#from tensorboardX import SummaryWriter

from config import parse_config
from models.networks_2S import BaseNet, ROINet 
from models.two_branch_2S import TwoBranchNet, ContextNet
from utils.utils import inference, train_select, AverageMeter, get_gpu_memory, select_proposals
from utils.tube_utils import flatten_tubes, valid_tubes
from utils.solver import WarmupCosineLR, WarmupStepLR, get_params
from data.ava_cls_2S import AVADataset, detection_collate, WIDTH, HEIGHT
from data.augmentations_2S import TubeAugmentation, BaseTransform
from utils.eval_utils import ava_evaluation
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap


args = parse_config()
args.max_iter = 1    # only 1 step for classification pretraining
args.pret_rgb="/data/Dan/ava_v2_1/cache/Cls-max1-i3d-two_branch/checkpoint_best.pth"
args.pret_of ="/data/Dan/ava_v2_1/cache/Cls_of-max1-i3d-two_branch/checkpoint_best.pth"

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
if args.num_classes != 80:
    if args.num_classes == 3:
        label_map = os.path.join(args.data_root, 'label/ava_finetune.pbtxt')
    elif args.num_classes == 60:
        label_map = os.path.join(args.data_root, 'label/ava_action_list_v2.1_for_activitynet_2018.pbtxt')
    categories, class_whitelist = read_labelmap(open(label_map, 'r'))
    classes = [(val['id'], val['name']) for val in categories]
    id2class = {c[0]: c[1] for c in classes}    # gt class id (1~x) --> class name
    for i, c in enumerate(sorted(list(class_whitelist))):
        label_dict[i] = c
else:
    for i in range(80): # 80 classes
        label_dict[i] = i+1

## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

#args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # specify which GPU(s) to be used
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
    augmentation = TubeAugmentation(args.image_size, scale=args.scale_norm, input_type=args.input_type,
                                    do_flip=args.do_flip, do_crop=args.do_crop, do_photometric=args.do_photometric, do_erase=args.do_erase)
    log_file.write("Data augmentation: "+ str(augmentation))
    train_dataset = AVADataset(args.data_root, 'train', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, augmentation, stride=1, num_classes=args.num_classes, foreground_only=True)
    val_dataset = AVADataset(args.data_root, 'val', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, BaseTransform(size=args.image_size, scale=args.scale_norm,input_type='2s'), stride=1, num_classes=args.num_classes, foreground_only=True)

    if args.milestones[0] == -1:
        args.milestones = [int(np.ceil(len(train_dataset) / args.batch_size) * args.max_epochs)]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    log_file.write("Training size: " + str(len(train_dataset)) + "\n")
    log_file.write("Validation size: " + str(len(val_dataset)) + "\n")
    print('Training STEP on ', train_dataset.name)

    ################ define models #################

    nets_rgb = OrderedDict()
    nets_of = OrderedDict()
    # backbone network
    nets_rgb['base_net'] = BaseNet(args,'rgb')
    nets_of['base_net'] = BaseNet(args,'flow')
    # ROI pooling
    nets_rgb['roi_net'] = ROINet(args.pool_mode, args.pool_size)
    nets_of['roi_net'] = ROINet(args.pool_mode, args.pool_size)

    # detection network
    for i in range(args.max_iter):
        if args.det_net == "two_branch":
            nets_rgb['det_net%d' % i] = TwoBranchNet(args, cls_only=True, input_type='rgb')
            nets_of['det_net%d' % i] = TwoBranchNet(args, cls_only=True, input_type='flow')
        else:
            raise NotImplementedError
    if not args.no_context:
        # context branch
        nets_rgb['context_net'] = ContextNet(args,'rgb')
        nets_of['context_net'] = ContextNet(args,'flow')
    
    for key in nets_rgb:
        nets_rgb[key] = nets_rgb[key].cuda()
    for key in nets_of:
        nets_of[key] = nets_of[key].cuda()

    ################ Training setup #################
    ################ Optimizer and Scheduler setup #################
    # Two ways to solve the conflict
    ## 1. Two optimizers
    ## 2. Two nets and concatenate params
    # params = get_params(nets, args) # bug: name is not base_net etc.
    params_rgb = get_params(nets_rgb,args)
    params_of = get_params(nets_of,args)
    params = params_rgb + params_of
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.det_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.det_lr)
    else:
        raise NotImplementedError

    if args.scheduler == "cosine":
        scheduler = WarmupCosineLR(optimizer, args.milestones, args.min_ratio, args.cycle_decay, args.warmup_iters)
    else:
        scheduler = WarmupStepLR(optimizer, args.milestones, args.warmup_iters)

    # Initialize AMP if needed
    if args.fp16:
        models, optimizer = amp.initialize([net for _,net in nets_rgb.items()], optimizer, opt_level="O1")
        for i, key in enumerate(nets_rgb):
            nets_rgb[key] = models[i]
        models, optimizer = amp.initialize([net for _,net in nets_of.items()], optimizer, opt_level="O1")
        for i, key in enumerate(nets_of):
            nets_of[key] = models[i]

    # DataParallel is used
    nets_rgb['base_net'] = torch.nn.DataParallel(nets_rgb['base_net'])
    nets_of['base_net'] = torch.nn.DataParallel(nets_of['base_net'])
    if not args.no_context:
        nets_rgb['context_net'] = torch.nn.DataParallel(nets_rgb['context_net'])
        nets_of['context_net'] = torch.nn.DataParallel(nets_of['context_net'])
    for i in range(args.max_iter):
        nets_rgb['det_net%d' % i].to('cuda:%d' % ((i+1)/gpu_count))
        nets_rgb['det_net%d' % i].set_device('cuda:%d' % ((i+1)/gpu_count))
        nets_of['det_net%d' % i].to('cuda:%d' % ((i+1)/gpu_count))
        nets_of['det_net%d' % i].set_device('cuda:%d' % ((i+1)/gpu_count))

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

            nets_rgb['base_net'].load_state_dict(checkpoint['base_net_rgb'])
            nets_of['base_net'].load_state_dict(checkpoint['base_net_of'])
            if not args.no_context and 'context_net' in checkpoint:
                nets_rgb['context_net'].load_state_dict(checkpoint['context_net_rgb'])
                nets_of['context_net'].load_state_dict(checkpoint['context_net_of'])
            for i in range(args.max_iter):
                nets_rgb['det_net%d' % i].load_state_dict(checkpoint['det_net_rgb%d' % i])
                nets_of['det_net%d' % i].load_state_dict(checkpoint['det_net_of%d' % i])
            
            # # Finetune
            # if args.num_classes != len(checkpoint['det_net%d' % i]['global_cls.bias']):
            #     print(f" >>>>>> Finetuning from {len(checkpoint['det_net%d' % i]['global_cls.bias'])} to {args.num_classes} <<<<<< ")
            #     for p in nets['base_net'].parameters(): p.requires_grad = False
            #     if not args.no_context:
            #         for p in nets['context_net'].parameters(): p.requires_grad = False
            #     for i in range(args.max_iter):
            #         for p in nets['det_net%d' % i].parameters(): p.requires_grad = False
            #         nets['det_net%d' % i].global_cls = nn.Conv3d(ic_t, args.num_classes, (1,1,1), bias=True)
            #         nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
            #         nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

            # if 'optimizer' in checkpoint:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            # if 'scheduler' in checkpoint:
            #     scheduler.load_state_dict(checkpoint['scheduler'])

            args.start_iteration = checkpoint['iteration']
            if checkpoint['iteration'] % int(np.ceil(len(train_dataset)/args.batch_size)) == 0:
                args.start_epochs = checkpoint['epochs']
            else:
                args.start_epochs = checkpoint['epochs'] - 1

            del checkpoint
            torch.cuda.empty_cache()
        else:
            # Reading model from the pretrained ones
            print ("Retrieving trained model from %s" % args.pret_rgb)
            checkpoint_rgb = torch.load(args.pret_rgb, map_location='cpu')
            print ("Retrieving trained model from %s" % args.pret_of)
            checkpoint_of = torch.load(args.pret_of, map_location='cpu')

            nets_rgb['base_net'].load_state_dict(checkpoint_rgb['base_net'])
            nets_of['base_net'].load_state_dict(checkpoint_of['base_net'])
            if not args.no_context and 'context_net' in checkpoint_rgb:
                nets_rgb['context_net'].load_state_dict(checkpoint_rgb['context_net'])
            if not args.no_context and 'context_net' in checkpoint_of:
                nets_of['context_net'].load_state_dict(checkpoint_of['context_net'])
            for i in range(args.max_iter):
                nets_rgb['det_net%d' % i].load_state_dict(checkpoint_rgb['det_net%d' % i])
                nets_of['det_net%d' % i].load_state_dict(checkpoint_of['det_net%d' % i])
            
            del checkpoint_rgb
            del checkpoint_of
            torch.cuda.empty_cache()
    ######################################################

    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')

    for i in range(args.max_iter):
        log_file.write(str(nets_rgb['det_net%d' % i])+'\n\n')
        log_file.write(str(nets_of['det_net%d' % i])+'\n\n')
    
    params_num = [sum(p.numel() for p in nets_rgb['base_net'].parameters() if p.requires_grad)]
    if not args.no_context:
        params_num.append(sum(p.numel() for p in nets_rgb['context_net'].parameters() if p.requires_grad))
    for i in range(args.max_iter):
        params_num.append(sum(p.numel() for p in nets_rgb['det_net%d' % i].parameters() if p.requires_grad))
    params_num.append(sum(p.numel() for p in nets_of['base_net'].parameters() if p.requires_grad))
    if not args.no_context:
        params_num.append(sum(p.numel() for p in nets_of['context_net'].parameters() if p.requires_grad))
    for i in range(args.max_iter):
        params_num.append(sum(p.numel() for p in nets_of['det_net%d' % i].parameters() if p.requires_grad))
    log_file.write("Number of parameters: "+str(params_num)+'\n\n')

    # Start training
    train(args, nets_rgb, nets_of, optimizer, scheduler, train_dataloader, val_dataloader, log_file)


def train(args, nets_rgb, nets_of, optimizer, scheduler, train_dataloader, val_dataloader, log_file):

    for _, net in nets_rgb.items():
        net.train()
    for _, net in nets_of.items():
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

    while epochs < args.max_epochs:
        for data_id, (images_c, targets, tubes, infos) in enumerate(train_dataloader):
            images = images_c[:,:,:3,:,:].cuda()
            flows = images_c[:,:,3:,:,:].cuda()
            # print(images, flows)
            # adjust learning rate
            scheduler.step()
            lr = optimizer.param_groups[-1]['lr']

            # get conv features
            conv_feat_rgb = nets_rgb['base_net'](images)
            context_feat_rgb = None
            if not args.no_context:
                context_feat_rgb = nets_rgb['context_net'](conv_feat_rgb)
            conv_feat_of = nets_of['base_net'](flows)
            context_feat_of = None
            if not args.no_context:
                context_feat_of = nets_of['context_net'](conv_feat_of)

            ########### Forward pass for the two-branch header ############

            optimizer.zero_grad()

            # adaptively get the start chunk
            max_chunks = 1
            T_start = 0
            T_length = args.T

            selected_tubes = []
            target_tubes = []
            # use target boxes as tubes
            for b in range(len(targets)):

                cur_tubes = tubes[b]
                selected_pos, selected_neg, ious = select_proposals(
                            targets[b][:,int(max_chunks/2)].reshape(targets[b].shape[0],1,-1), 
                            cur_tubes[:, int(cur_tubes.shape[1]/2)].reshape(cur_tubes.shape[0],1,-1),
                            None,
                            0.75, 5, 'uniform', 3)
                cur_selected_tubes = np.zeros((len(selected_pos)+len(selected_neg), cur_tubes.shape[1], 4), dtype=np.float32)
                cur_target_tubes = np.zeros((len(selected_pos)+len(selected_neg), 1, 6+args.num_classes), dtype=np.float32)    # only one frame for loss
                row = 0
                for ii,jj in selected_pos:
                    cur_selected_tubes[row] = cur_tubes[jj]
                    cur_target_tubes[row,:,:4] = targets[b][ii,int(max_chunks/2),:4]
                    cur_target_tubes[row,:,6:] = targets[b][ii,int(max_chunks/2),4:]
                    cur_target_tubes[row,:,4] = 1     # flag for classification
                    row += 1

                for ii,jj in selected_neg:
                    cur_selected_tubes[row] = cur_tubes[jj]
                    cur_target_tubes[row,:,4] = 1     # flag for classification
                    row += 1

                cur_target_tubes = np.concatenate([cur_target_tubes,
                                                   cur_target_tubes,
                                                   cur_target_tubes], axis=1)

                selected_tubes.append(cur_selected_tubes)
                target_tubes.append(cur_target_tubes)

            # flatten list of tubes
            flat_targets, tubes_nums = flatten_tubes(target_tubes, batch_idx=False)
            flat_tubes, _ = flatten_tubes(selected_tubes, batch_idx=True)    # add batch_idx for ROI pooling
            flat_targets = torch.FloatTensor(flat_targets).to(conv_feat_rgb)
            flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat_rgb)    # gpu:0 for ROI pooling
            flat_targets_of = flat_targets.to(conv_feat_of)
            flat_tubes_of = flat_tubes.to(conv_feat_of)    # gpu:0 for ROI pooling

            # ROI Pooling
            pooled_feat_rgb = nets_rgb['roi_net'](conv_feat_rgb[:, T_start:T_start+T_length].contiguous(), flat_tubes)
            pooled_feat_of = nets_of['roi_net'](conv_feat_of[:, T_start:T_start+T_length].contiguous(), flat_tubes_of)
            _,C,W,H = pooled_feat_rgb.size()
            pooled_feat_rgb = pooled_feat_rgb.view(-1, T_length, pooled_feat_rgb.size(1), W, H)
            pooled_feat_of = pooled_feat_of.view(-1, T_length, pooled_feat_of.size(1), W, H)

            temp_context_feat_rgb = None
            temp_context_feat_of = None
            if not args.no_context:
                temp_context_feat_rgb = torch.zeros((pooled_feat_rgb.size(0),context_feat_rgb.size(1),T_length,1,1)).to(context_feat_rgb)
                temp_context_feat_of = torch.zeros((pooled_feat_of.size(0),context_feat_of.size(1),T_length,1,1)).to(context_feat_of)
                for p in range(pooled_feat_rgb.size(0)):
                    temp_context_feat_rgb[p] = context_feat_rgb[int(flat_tubes[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()
                    temp_context_feat_of[p] = context_feat_of[int(flat_tubes_of[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()

            _,_,_,_, cur_loss_global_cls_rgb, _, _ = nets_rgb['det_net0'](pooled_feat_rgb, context_feat=temp_context_feat_rgb, tubes=flat_tubes, targets=flat_targets)
            _,_,_,_, cur_loss_global_cls_of, _, _ = nets_of['det_net0'](pooled_feat_of, context_feat=temp_context_feat_of, tubes=flat_tubes, targets=flat_targets_of)
            # Fuse the loss
            cur_loss_global_cls = 0.6*cur_loss_global_cls_rgb+0.4*cur_loss_global_cls_of
            cur_loss_global_cls = cur_loss_global_cls.mean()

            ########### Gradient updates ############

            losses.update(cur_loss_global_cls.item())

            if args.fp16:
                with amp.scale_loss(cur_loss_global_cls, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                cur_loss_global_cls.backward()

            # if data_id < 3:
            #     print('\nglobal weight: ')
            #     print(nets['det_net0'].global_cls.weight)
            #     print('\nglobal grad: ')
            #     print(nets['det_net0'].global_cls.weight.grad)
            #     print(cur_loss_global_cls.item())
            optimizer.step()


            ############### Print logs and save models ############

            iteration += 1

            if iteration % args.print_step == 0 and iteration>0:

                gpu_memory = get_gpu_memory()

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time.update(t1 - t0)

                print_line = 'Epoch {}/{}({}) Iteration {:06d} lr {:.2e} ' \
                             'loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f}) GPU usage: {}'.format(
                                epochs+1, args.max_epochs, epoch_size, iteration, lr,
                                losses.val, losses.avg, batch_time.val, batch_time.avg, gpu_memory)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                log_file.write(print_line+'\n')
                print(print_line)

            if (iteration % args.save_step == 0) and iteration>0:
                print('Saving state, iter:', iteration)
                save_name = args.save_root+'checkpoint_'+str(iteration) + '.pth'
                save_dict = {
                    'epochs': epochs+1,
                    'iteration': iteration,
                    'base_net_rgb': nets_rgb['base_net'].state_dict(),
                    'context_net_rgb': nets_rgb['context_net'].state_dict(),
                    'base_net_of': nets_of['base_net'].state_dict(),
                    'context_net_of': nets_of['context_net'].state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_mAP': best_mAP,
                    'cfg': args}
                for i in range(args.max_iter):
                    save_dict['det_net_rgb%d' % i] = nets_rgb['det_net%d' % i].state_dict()
                    save_dict['det_net_of%d' % i] = nets_of['det_net%d' % i].state_dict()
                torch.save(save_dict, save_name)

                # only keep the latest model
                if os.path.isfile(args.save_root+'checkpoint_'+str(iteration-args.save_step) + '.pth'):
                    os.remove(args.save_root+'checkpoint_'+str(iteration-args.save_step) + '.pth')
                    print (args.save_root+'checkpoint_'+str(iteration-args.save_step) + '.pth  removed!')
            # break
            # For consistency when resuming from the middle of an epoch
            if iteration % epoch_size == 0 and iteration > 0:
                break

        
        ##### Validation at the end of each epoch #####

        if True:
            torch.cuda.synchronize()
            tvs = time.perf_counter()
    
            for _, net in nets_rgb.items():
                net.eval() # switch net to evaluation mode
            for _, net in nets_of.items():
                net.eval() # switch net to evaluation mode
            print('Validating at ', iteration)
            all_metrics = validate(args, val_dataloader, nets_rgb,nets_of, iteration, iou_thresh=args.iou_thresh)
    
            prt_str = ''
            for i in range(args.max_iter):
                prt_str += 'Iter '+str(i+1)+': MEANAP =>'+str(all_metrics[i]['PascalBoxes_Precision/mAP@0.5IOU'])+'\n'
            print(prt_str)
            log_file.write(prt_str)
    
            log_file.write("Best MEANAP so far => {}\n".format(best_mAP))
            for i in class_whitelist:
                log_file.write("({}) {}: {}\n".format(i,id2class[i], 
                    all_metrics[-1]["PascalBoxes_PerformanceByCategory/AP@0.5IOU/{}".format(id2class[i])]))
    
    
    #        writer.add_scalar('mAP', all_metrics[-1]['PascalBoxes_Precision/mAP@0.5IOU'], iteration)
    #        for key, ap in all_metrics[-1].items():
    #            writer.add_scalar(key, ap, iteration)
    
            if all_metrics[-1]['PascalBoxes_Precision/mAP@0.5IOU'] > best_mAP:
                best_mAP = all_metrics[-1]['PascalBoxes_Precision/mAP@0.5IOU']
                print('Saving current best model, iter:', iteration)
                save_name = args.save_root+'checkpoint_best.pth'
                save_dict = {
                    'epochs': epochs+1,
                    'iteration': iteration,
                    'base_net_rgb': nets_rgb['base_net'].state_dict(),
                    'context_net_rgb': nets_rgb['context_net'].state_dict(),
                    'base_net_of': nets_of['base_net'].state_dict(),
                    'context_net_of': nets_of['context_net'].state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_mAP': best_mAP,
                    'cfg': args}
                for i in range(args.max_iter):
                    save_dict['det_net_rgb%d' % i] = nets_rgb['det_net%d' % i].state_dict()
                    save_dict['det_net_of%d' % i] = nets_of['det_net%d' % i].state_dict()
                torch.save(save_dict, save_name)
    
            for _, net in nets_rgb.items():
                net.train() # switch net to training mode
            for _, net in nets_of.items():
                net.train() # switch net to training mode
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prt_str2 = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
            print(prt_str2)
            log_file.write(prt_str2)

        epochs += 1


    log_file.close()
#    writer.close()


def validate(args, val_dataloader, nets_rgb, nets_of, iteration=0, iou_thresh=0.5):
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

    with torch.no_grad():    # for evaluation
        for num, (images_c, targets, tubes, infos) in enumerate(val_dataloader):

            if (num+1) % 100 == 0:
                print ("%d / %d" % (num+1, len(val_dataloader.dataset)/args.batch_size))

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

            _, _, channels, height, width = images_c.size()
            images = images_c[:,:,:3,:,:].cuda()
            flows = images_c[:,:,3:,:,:].cuda()

            # get conv features
            conv_feat_rgb = nets_rgb['base_net'](images)
            context_feat_rgb = None
            if not args.no_context:
                context_feat_rgb = nets_rgb['context_net'](conv_feat_rgb)
            conv_feat_of = nets_of['base_net'](flows)
            context_feat_of = None
            if not args.no_context:
                context_feat_of = nets_of['context_net'](conv_feat_of)

            ############## Inference ##############

            T_start = 0
            T_length = args.T

            selected_tubes = []
            # use target boxes as tubes
            for b in range(len(targets)):
                cur_tubes = tubes[b]
                selected_tubes.append(cur_tubes)

            # flatten list of tubes
            flat_tubes, tubes_nums = flatten_tubes(selected_tubes, batch_idx=True)    # add batch_idx for ROI pooling
            flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat_rgb)    # gpu:0 for ROI pooling
            flat_tubes_of = flat_tubes.to(conv_feat_of)    # gpu:0 for ROI pooling

            # ROI Pooling            
            pooled_feat_rgb = nets_rgb['roi_net'](conv_feat_rgb[:, T_start:T_start+T_length].contiguous(), flat_tubes)
            pooled_feat_of = nets_of['roi_net'](conv_feat_of[:, T_start:T_start+T_length].contiguous(), flat_tubes)
            _,C,W,H = pooled_feat_rgb.size()
            pooled_feat_rgb = pooled_feat_rgb.view(-1, T_length, pooled_feat_rgb.size(1), W, H)
            pooled_feat_of = pooled_feat_of.view(-1, T_length, pooled_feat_of.size(1), W, H)


            temp_context_feat_rgb = None
            temp_context_feat_of = None
            if not args.no_context:
                temp_context_feat_rgb = torch.zeros((pooled_feat_rgb.size(0),context_feat_rgb.size(1),T_length,1,1)).to(context_feat_rgb)
                temp_context_feat_of = torch.zeros((pooled_feat_of.size(0),context_feat_of.size(1),T_length,1,1)).to(context_feat_of)
                for p in range(pooled_feat_rgb.size(0)):
                    temp_context_feat_rgb[p] = context_feat_rgb[int(flat_tubes[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()
                    temp_context_feat_of[p] = context_feat_of[int(flat_tubes_of[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()
            global_prob_rgb, _,_,_, _,_,_ = nets_rgb['det_net0'](pooled_feat_rgb, context_feat=temp_context_feat_rgb, tubes=None, targets=None)
            global_prob_of, _,_,_, _,_,_ = nets_of['det_net0'](pooled_feat_of, context_feat=temp_context_feat_of, tubes=None, targets=None)
            global_prob = 0.6*global_prob_rgb + 0.4*global_prob_of
            #################### Evaluation #################

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

    fout.close()

    all_metrics = []
    for i in range(args.max_iter):
        fouts[i].close()

        metrics = ava_evaluation(os.path.join(args.data_root, 'label/'), output_files[i], gt_file)
        all_metrics.append(metrics)
    
    return all_metrics


if __name__ == '__main__':
    main()

