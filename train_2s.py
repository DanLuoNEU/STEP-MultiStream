"""
05/29/2020, Dan
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
from external.maskrcnn_benchmark.roi_layers import nms
from utils.utils import inference, train_select, AverageMeter, get_gpu_memory
from utils.tube_utils import flatten_tubes, valid_tubes
from utils.solver import WarmupCosineLR, WarmupStepLR, get_params
from data.ava_2S import AVADataset, detection_collate, WIDTH, HEIGHT
from data.augmentations_2S import TubeAugmentation, BaseTransform
from utils.eval_utils import ava_evaluation
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap


args = parse_config()

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

args.label_dict = label_dict
args.id2class = id2class

## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

#args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,7"  # specify which GPU(s) to be used
gpu_count = torch.cuda.device_count()
torch.backends.cudnn.benchmark=True
best_mAP = 0

def main():
    global best_mAP

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
    train_dataset = AVADataset(args.data_root, 'train', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, augmentation, proposal_path=args.proposal_path_train, stride=1, anchor_mode=args.anchor_mode, num_classes=args.num_classes, foreground_only=True)
    val_dataset = AVADataset(args.data_root, 'val', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, BaseTransform(size=args.image_size, scale=args.scale_norm, input_type='2s'), proposal_path=args.proposal_path_val, stride=1, anchor_mode=args.anchor_mode, num_classes=args.num_classes, foreground_only=True)#foreground_only=False

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
            nets_rgb['det_net%d' % i] = TwoBranchNet(args, input_type='rgb')
            nets_of['det_net%d' % i] = TwoBranchNet(args, input_type='flow')
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

    # Load pretrained model if needed
    if args.pretrain_path is not None:
        if os.path.isfile(args.pretrain_path):
            print ("Loading pretrain model from %s" % args.pretrain_path)
            checkpoint = torch.load(args.pretrain_path, map_location='cuda:0')

            nets_rgb['base_net'].load_state_dict(checkpoint['base_net_rgb'])
            nets_of['base_net'].load_state_dict(checkpoint['base_net_of'])
            if not args.no_context and 'context_net' in checkpoint:
                nets_rgb['context_net'].load_state_dict(checkpoint['context_net_rgb'])
                nets_of['context_net'].load_state_dict(checkpoint['context_net_of'])
            for i in range(args.max_iter):
                # RGB stream
                model_dict = nets_rgb['det_net%d' % i].state_dict()
                pretrained_dict = checkpoint.get('det_net_rgb%d' % i, checkpoint["det_net_rgb0"])    # load from classfication pretrained model, so only det_net0 is loaded
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and k.find('global_cls') <= -1}    # last layer (classifier) is not loaded
                model_dict.update(pretrained_dict)
                nets_rgb['det_net%d' % i].load_state_dict(model_dict)
                model_dict = nets_of['det_net%d' % i].state_dict()
                # OF stream
                pretrained_dict = checkpoint.get('det_net_of%d' % i, checkpoint["det_net_of0"])    # load from classfication pretrained model, so only det_net0 is loaded
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and k.find('global_cls') <= -1}    # last layer (classifier) is not loaded
                model_dict.update(pretrained_dict)
                nets_of['det_net%d' % i].load_state_dict(model_dict)
        else:
            raise ValueError("Pretrain model not found!", args.pretrain_path)

        del checkpoint
        torch.cuda.empty_cache()

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
            checkpoint = torch.load(model_path, map_location='cuda:0')

            nets_rgb['base_net'].load_state_dict(checkpoint['base_net_rgb'])
            nets_of['base_net'].load_state_dict(checkpoint['base_net_of'])
            if not args.no_context and 'context_net' in checkpoint:
                nets_rgb['context_net'].load_state_dict(checkpoint['context_net_rgb'])
                nets_of['context_net'].load_state_dict(checkpoint['context_net_of'])
            for i in range(args.max_iter):
                nets_rgb['det_net%d' % i].load_state_dict(checkpoint['det_net_rgb%d' % i])
                nets_of['det_net%d' % i].load_state_dict(checkpoint['det_net_of%d' % i])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

            args.start_iteration = checkpoint['iteration']
            if checkpoint['iteration'] % int(np.ceil(len(train_dataset)/args.batch_size)) == 0:
                args.start_epochs = checkpoint['epochs']
            else:
                args.start_epochs = checkpoint['epochs'] - 1
            best_mAP = checkpoint['val_mAP']

            del checkpoint
            torch.cuda.empty_cache()

    ######################################################

    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')

    for i in range(args.max_iter):
        log_file.write(str(nets_rgb['det_net%d' % i])+'\n\n')
        log_file.write(str(nets_of['det_net%d' % i])+'\n\n')
    
    # Start training
    train(args, nets_rgb, nets_of, optimizer, scheduler, train_dataloader, val_dataloader, log_file)


def train(args, nets_rgb, nets_of, optimizer, scheduler, train_dataloader, val_dataloader, log_file):
    global best_mAP

    for _, net in nets_rgb.items():
        net.train()
    for _, net in nets_rgb.items():
        net.train()

    # loss counters
    batch_time = AverageMeter(200)
    losses = [AverageMeter(200) for _ in range(args.max_iter)]
    losses_global_cls = AverageMeter(200)
    losses_local_loc = AverageMeter(200)
    losses_neighbor_loc = AverageMeter(200)

#    writer = SummaryWriter(args.save_root+"summary"+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

    ################ Training loop #################

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    epochs = args.start_epochs
    iteration = args.start_iteration
    epoch_size = int(np.ceil(len(train_dataloader.dataset) / args.batch_size))

    while epochs < args.max_epochs:
        for _, (images_c, targets, tubes, infos) in enumerate(train_dataloader):
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

            ############# Inference to get candidates for each iteration ########

            # randomly sample a fixed number of tubes
            if args.NUM_SAMPLE > 0 and args.NUM_SAMPLE < tubes[0].shape[0]:
                sampled_idx = np.random.choice(tubes[0].shape[0], args.NUM_SAMPLE, replace=False)
                for i in range(len(tubes)):
                    tubes[i] = tubes[i][sampled_idx]

            for _, net in nets_rgb.items():
                net.eval()
            with torch.no_grad():
                history, _ = inference(args, conv_feat_rgb, context_feat_rgb, nets_rgb, args.max_iter-1, tubes)
            for _, net in nets_rgb.items():
                net.train()
            for _, net in nets_of.items():
                net.train()

            ########### Forward pass for each iteration ############
            optimizer.zero_grad()
            loss_back = 0.

            # loop for each step
            for i in range(1, args.max_iter+1):    # index from 1

                # adaptively get the start chunk
                chunks = args.NUM_CHUNKS[i]
                max_chunks = args.NUM_CHUNKS[args.max_iter]
                T_start = int((args.NUM_CHUNKS[args.max_iter] - chunks) / 2) * args.T
                T_length = chunks * args.T
                T_mid = int(chunks/2) * args.T   # center chunk within T_length
                chunk_idx = [j*args.T + int(args.T/2) for j in range(chunks)]    # used to index the middel frame of each chunk

                # select training samples
                selected_tubes, target_tubes = train_select(i, history[i-2], targets, tubes, args)

                ######### Start training ########

                # flatten list of tubes
                flat_targets, _ = flatten_tubes(target_tubes, batch_idx=False)
                flat_tubes, _ = flatten_tubes(selected_tubes, batch_idx=True)    # add batch_idx for ROI pooling
                flat_targets = torch.FloatTensor(flat_targets).to(conv_feat_rgb)
                flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat_rgb)
                flat_targets_of = flat_targets.to(conv_feat_of)
                flat_tubes_of = flat_tubes.to(conv_feat_of)

                # ROI Pooling
                pooled_feat_rgb = nets_rgb['roi_net'](conv_feat_rgb[:, T_start:T_start+T_length].contiguous(), flat_tubes)
                pooled_feat_of = nets_of['roi_net'](conv_feat_of[:, T_start:T_start+T_length].contiguous(), flat_tubes_of)
                _,C,W,H = pooled_feat_rgb.size()
                pooled_feat_rgb = pooled_feat_rgb.view(-1, T_length, C, W, H)
                pooled_feat_of = pooled_feat_of.view(-1, T_length, C, W, H)

                temp_context_feat_rgb = None
                temp_context_feat_of = None
                if not args.no_context:
                    temp_context_feat_rgb = torch.zeros((pooled_feat_rgb.size(0),context_feat_rgb.size(1),T_length,1,1)).to(context_feat_rgb)
                    temp_context_feat_of = torch.zeros((pooled_feat_of.size(0),context_feat_of.size(1),T_length,1,1)).to(context_feat_of)
                    for p in range(pooled_feat_rgb.size(0)):
                        temp_context_feat_rgb[p] = context_feat_rgb[int(flat_tubes[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()
                        temp_context_feat_of[p] = context_feat_of[int(flat_tubes_of[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()

                _,_,_,_, cur_loss_global_cls_rgb, cur_loss_local_loc_rgb, cur_loss_neighbor_loc_rgb = nets_rgb['det_net%d' % (i-1)](pooled_feat_rgb, context_feat=temp_context_feat_rgb, tubes=flat_tubes, targets=flat_targets)
                _,_,_,_, cur_loss_global_cls_of, cur_loss_local_loc_of, cur_loss_neighbor_loc_of = nets_of['det_net%d' % (i-1)](pooled_feat_of, context_feat=temp_context_feat_of, tubes=flat_tubes_of, targets=flat_targets_of)
                # Fuse the loss
                cur_loss_global_cls = 0.5*cur_loss_global_cls_rgb+0.5*cur_loss_global_cls_of
                cur_loss_global_cls = cur_loss_global_cls.mean()
                cur_loss_local_loc = 0.5*cur_loss_local_loc_rgb+0.5*cur_loss_local_loc_of
                cur_loss_local_loc = cur_loss_local_loc.mean()
                cur_loss_neighbor_loc = 0.5*cur_loss_neighbor_loc_rgb+0.5*cur_loss_neighbor_loc_of
                cur_loss_neighbor_loc = cur_loss_neighbor_loc.mean()

                cur_loss = cur_loss_global_cls + \
                            cur_loss_local_loc * args.lambda_reg + \
                            cur_loss_neighbor_loc * args.lambda_neighbor
                loss_back += cur_loss.to(conv_feat_rgb.device)

                losses[i-1].update(cur_loss.item())
                if cur_loss_neighbor_loc.item() > 0:
                    losses_neighbor_loc.update(cur_loss_neighbor_loc.item())

            ########### Gradient updates ############
            # record last step only
            losses_global_cls.update(cur_loss_global_cls.item())
            losses_local_loc.update(cur_loss_local_loc.item())

            if args.fp16:
                loss_back /= args.max_iter    # prevent gradient overflow
                with amp.scale_loss(loss_back, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_back.backward()
            optimizer.step()

            ############### Print logs and save models ############

            iteration += 1

            if iteration % args.print_step == 0 and iteration>0:

                gpu_memory = get_gpu_memory()

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time.update(t1 - t0)

                print_line = 'Epoch {}/{}({}) Iteration {:06d} lr {:.2e} '.format(
                                epochs+1, args.max_epochs, epoch_size, iteration, lr)
                for i in range(args.max_iter):
                    print_line += 'loss-{} {:.3f} '.format(i+1, losses[i].avg)
                print_line += 'loss_global_cls {:.3f} loss_local_loc {:.3f} loss_neighbor_loc {:.3f} Timer {:0.3f}({:0.3f}) GPU usage: {}'.format(
                                losses_global_cls.avg, losses_local_loc.avg, losses_neighbor_loc.avg, batch_time.val, batch_time.avg, gpu_memory)

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

            # For consistency when resuming from the middle of an epoch
            if iteration % epoch_size == 0 and iteration > 0:
                break

        
        ##### Validation at the end of each epoch #####

        validate_epochs = [0,1,5,9,13,14]
        if epochs in validate_epochs:
            torch.cuda.synchronize()
            tvs = time.perf_counter()
    
            for _, net in nets_rgb.items():
                net.eval() # switch net to evaluation mode
            for _, net in nets_of.items():
                net.eval() # switch net to evaluation mode
            print('Validating at ', iteration)
            all_metrics = validate(args, val_dataloader, nets_rgb, nets_of, iteration, iou_thresh=args.iou_thresh)
    
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

            history_rgb, _ = inference(args, conv_feat_of, context_feat_of, nets_of, args.max_iter, tubes)
            history_of, _ = inference(args, conv_feat_of, context_feat_of, nets_of, args.max_iter, tubes)
            history = history_rgb

            #################### Evaluation #################

            # loop for each  iteration
            for i in range(len(history)):
                pred_prob = history[i]['pred_prob'].cpu()
                pred_prob = pred_prob[:,int(pred_prob.shape[1]/2)]
                pred_tubes = history[i]['pred_loc'].cpu()
                pred_tubes = pred_tubes[:,int(pred_tubes.shape[1]/2)]
                tubes_nums = history[i]['tubes_nums']

                # loop for each sample in a batch
                tubes_count = 0
                for b in range(len(tubes_nums)):
                    info = infos[b]
                    seq_start = tubes_count
                    tubes_count = tubes_count + tubes_nums[b]
    
                    cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
                    cur_pred_tubes = pred_tubes[seq_start:seq_start+tubes_nums[b]]

                    # do NMS first
                    all_scores = []
                    all_boxes = []
                    all_idx = []
                    for cl_ind in range(args.num_classes):
                        scores = cur_pred_prob[:, cl_ind].squeeze().reshape(-1)
                        c_mask = scores.gt(args.conf_thresh) # greater than minmum threshold
                        scores = scores[c_mask]
                        idx = np.where(c_mask.numpy())[0]
                        if len(scores) == 0:
                            all_scores.append([])
                            all_boxes.append([])
                            continue
                        boxes = cur_pred_tubes.clone()
                        l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                        boxes = boxes[l_mask].view(-1, 4)
    
                        boxes = valid_tubes(boxes.view(-1,1,4)).view(-1,4)
                        keep = nms(boxes, scores, args.nms_thresh)
                        boxes = boxes[keep].numpy()
                        scores = scores[keep].numpy()
                        idx = idx[keep]
    
                        boxes[:, ::2] /= width
                        boxes[:, 1::2] /= height
                        all_scores.append(scores)
                        all_boxes.append(boxes)
                        all_idx.append(idx)

                    # get the top scores
                    scores_list = [(s,cl_ind,j) for cl_ind,scores in enumerate(all_scores) for j,s in enumerate(scores)]
                    if args.evaluate_topk > 0:
                        scores_list.sort(key=lambda x: x[0])
                        scores_list = scores_list[::-1]
                        scores_list = scores_list[:args.topk]

                    for s,cl_ind,j in scores_list:
                        # write to files
                        box = all_boxes[cl_ind][j]
                        fouts[i].write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6},{7:.4}\n'.format(
                                                    info['video_name'],
                                                    info['fid'],
                                                    box[0],box[1],box[2],box[3],
                                                    label_dict[cl_ind],
                                                    s))
    fout.close()

    all_metrics = []
    for i in range(args.max_iter):
        fouts[i].close()

        metrics = ava_evaluation(os.path.join(args.data_root, 'label/'), output_files[i], gt_file)
        all_metrics.append(metrics)
    
    return all_metrics


if __name__ == '__main__':
    main()
