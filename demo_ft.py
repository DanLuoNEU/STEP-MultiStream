
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
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
import sys

from config import parse_config
from models import BaseNet, ROINet, TwoBranchNet, ContextNet
from external.maskrcnn_benchmark.roi_layers import nms
from utils.utils import inference, train_select, AverageMeter, get_gpu_memory, Timer
from utils.tube_utils import flatten_tubes, valid_tubes, compute_box_iou
from utils.vis_utils import overlay_image
from data.customize import CustomizedDataset, detection_collate, WIDTH, HEIGHT
from data.augmentations import BaseTransform

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"  # specify which GPU(s) to be used

if "--visualize" in sys.argv:
    VIS = True
else:
    VIS = False

def main():

    ################## Customize your configuratons here ###################

    # checkpoint_path = '/data/Dan/ava_v2_1/cache/STEP-max3-i3d-two_branch/checkpoint_best.pth'
    checkpoint_path = '/data/truppr/ava/cache/STEP-max3-i3d-two_branch/checkpoint_best.pth'

    if os.path.isfile(checkpoint_path):
        print ("Loading pretrain model from %s" % checkpoint_path)
        map_location = 'cuda:0'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        args = checkpoint['cfg']
    else:
        raise ValueError("Pretrain model not found!", checkpoint_path)

    # TODO: Set data_root to the customized input dataset
    # args.data_root = 'examples/rgb_frames'
    args.data_root = '/data/CLASP-DATA/DEMO-DATA/20191024-test-cam20exp1/'

    args.save_root = os.path.join(os.path.dirname(args.data_root), 'results/')
    args.batch_size=1
    if not os.path.isdir(args.save_root):
        print("making " + str(args.save_root))
        os.makedirs(args.save_root)
    else:
        print("already exists " + str(args.save_root))

    # TODO: modify this setting according to the actual frame rate and file name
    source_fps = 30
    im_format = 'frame%05d.jpg'
    conf_thresh = 0.4
    global_thresh = 0.8    # used for cross-class NMS
    args.id2class = {1:'transfer', 2: 'transfer', 3:'background'}
    # args.id2class[1] = 'transfer' #'give'
    # args.id2class[2] = 'transfer' #'take'
    
    ################ Define models #################

    gpu_count = torch.cuda.device_count()
    nets = OrderedDict()
    # backbone network
    nets['base_net'] = BaseNet(args)
    # ROI pooling
    nets['roi_net'] = ROINet(args.pool_mode, args.pool_size)

    # detection network
    for i in range(args.max_iter):
        if args.det_net == "two_branch":
            nets['det_net%d' % i] = TwoBranchNet(args)
        else:
            raise NotImplementedError
    if not args.no_context:
        # context branch
        nets['context_net'] = ContextNet(args)

    for key in nets:
        nets[key] = nets[key].cuda()

    nets['base_net'] = torch.nn.DataParallel(nets['base_net'])
    if not args.no_context:
        nets['context_net'] = torch.nn.DataParallel(nets['context_net'])
    for i in range(args.max_iter):
        nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
        nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

    # load pretrained model 
    nets['base_net'].load_state_dict(checkpoint['base_net'])
    if not args.no_context and 'context_net' in checkpoint:
        nets['context_net'].load_state_dict(checkpoint['context_net'])
    for i in range(args.max_iter):
        pretrained_dict = checkpoint['det_net%d' % i]
        nets['det_net%d' % i].load_state_dict(pretrained_dict)

    
    ################ DataLoader setup #################

    dataset = CustomizedDataset(args.data_root, args.T, args.NUM_CHUNKS[args.max_iter], source_fps, args.fps, BaseTransform(args.image_size, args.means, args.stds,args.scale_norm), anchor_mode=args.anchor_mode, im_format=im_format)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)

    ################ Inference #################

    for _, net in nets.items():
        net.eval()

    fout = open(os.path.join(args.save_root, 'results.txt'), 'w')
    tout = open(os.path.join(args.save_root, 'tim-results.txt'), 'w')
    fout.close()
    tout.close()

    torch.cuda.synchronize()
    t0 = time.time()
    t_base = Timer("base")
    t_context = Timer("context")
    t_2b = Timer("2b")
    t_rest = Timer("rest")
    with torch.no_grad():
        for _, (images, tubes, infos) in enumerate(dataloader):

            fout = open(os.path.join(args.save_root, 'results.txt'), 'a')
            tout = open(os.path.join(args.save_root, 'tim-results.txt'), 'a')

            _, _, channels, height, width = images.size()
            images = images.cuda()

            # get conv features
            t_base.start()
            conv_feat = nets['base_net'](images)
            t_base.stop()
            context_feat = None
            if not args.no_context:
                t_context.start()
                context_feat = nets['context_net'](conv_feat)
                t_context.stop()

            t_2b.start()
            history, _ = inference(args, conv_feat, context_feat, nets, args.max_iter, tubes)
            t_2b.stop()
            t_rest.start()
            
            # collect result of the last step
            pred_prob = history[-1]['pred_prob'].cpu()
            pred_prob = pred_prob[:,int(pred_prob.shape[1]/2)]
            pred_tubes = history[-1]['pred_loc'].cpu()
            pred_tubes = pred_tubes[:,int(pred_tubes.shape[1]/2)]
            pred_tubes_other1 = history[-1]['pred_loc_other1'].cpu()
            pred_tubes_other1 = pred_tubes_other1[:,int(pred_tubes_other1.shape[1]/2)]
            pred_tubes_other2 = history[-1]['pred_loc_other2'].cpu()
            pred_tubes_other2 = pred_tubes_other2[:,int(pred_tubes_other2.shape[1]/2)]
            
            tubes_nums = history[-1]['tubes_nums']

            # loop for each batch
            tubes_count = 0
            for b in range(len(tubes_nums)):
                info = infos[b]
                seq_start = tubes_count
                tubes_count = tubes_count + tubes_nums[b]

                cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
                cur_pred_tubes = pred_tubes[seq_start:seq_start+tubes_nums[b]]
                cur_pred_tubes_other1 = pred_tubes_other1[seq_start:seq_start+tubes_nums[b]]
                cur_pred_tubes_other2 = pred_tubes_other2[seq_start:seq_start+tubes_nums[b]]
                
                # do NMS first
                all_scores = []
                all_boxes = []
                all_other_1 = []
                all_other_2 = []
                all_idx = []
                for cl_ind in range(args.num_classes):
                    scores = cur_pred_prob[:, cl_ind].squeeze()
                    c_mask = scores.gt(conf_thresh) # greater than a threshold
                    scores = scores[c_mask]
                    idx = np.where(c_mask.numpy())[0]
                    
                    if len(scores) == 0:
                        all_scores.append([])
                        all_boxes.append([])
                        all_other_1.append([])
                        all_other_2.append([])
                        continue

                    boxes = cur_pred_tubes.clone()
                    boxes_other1 = cur_pred_tubes_other1.clone()
                    boxes_other2 = cur_pred_tubes_other2.clone()

                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    boxes_other1 = boxes_other1[l_mask].view(-1, 4)
                    boxes_other2 = boxes_other2[l_mask].view(-1, 4)

                    boxes = valid_tubes(boxes.view(-1,1,4)).view(-1,4)
                    # boxes_other1 = valid_tubes(boxes_other1.view(-1,1,4)).view(-1,4)
                    # boxes_other2 = valid_tubes(boxes_other2.view(-1,1,4)).view(-1,4)

                    keep = nms(boxes, scores, args.nms_thresh)
                    boxes = boxes[keep].numpy()
                    boxes_other1 = boxes_other1[keep].numpy()
                    boxes_other2 = boxes_other2[keep].numpy()
                    scores = scores[keep].numpy()
                    idx = idx[keep]

                    boxes[:, ::2] /= width
                    boxes[:, 1::2] /= height
                    boxes_other1[:, ::2] /= width
                    boxes_other1[:, 1::2] /= height
                    boxes_other2[:, ::2] /= width
                    boxes_other2[:, 1::2] /= height

                    all_scores.append(scores)
                    all_boxes.append(boxes)
                    all_idx.append(idx)
                    all_other_1.append(boxes_other1)
                    all_other_2.append(boxes_other2)

                # get the top scores
                scores_list = [(s,cl_ind,j) for cl_ind,scores in enumerate(all_scores) for j,s in enumerate(scores)]
                if args.evaluate_topk > 0:
                    scores_list.sort(key=lambda x: x[0])
                    scores_list = scores_list[::-1]
                    scores_list = scores_list[:args.topk]
                
                # merge high overlapping boxes (a simple greedy method)
                merged_result = {}
                flag = [1 for _ in range(len(scores_list))]
                for i in range(len(scores_list)):
                    if flag[i]:
                        s, cl_ind, j = scores_list[i]
                        box = all_boxes[cl_ind][j]
                        box_1 = all_other_1[cl_ind][j]
                        box_2 = all_other_2[cl_ind][j]

                        temp = ([box], [args.label_dict[cl_ind]], [s])

                        # find all high IoU boxes
                        for ii in range(i+1, len(scores_list)):
                            if flag[ii]:
                                s2, cl_ind2, j2 = scores_list[ii]
                                box2 = all_boxes[cl_ind2][j2]
                                if compute_box_iou(box, box2) > global_thresh:
                                    flag[ii] = 0
                                    temp[0].append(box2)
                                    temp[1].append(args.label_dict[cl_ind2])
                                    temp[2].append(s2)
                        
                        merged_box = np.mean(np.concatenate(temp[0], axis=0).reshape(-1,4), axis=0)
                        key = ','.join(merged_box.astype(str).tolist())
                        merged_result[key] = [(l, s) for l,s in zip(temp[1], temp[2])]
                

                        tout.write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6:.4},{7:.4},{8:.4},{9:.4},{10:.4},{11:.4},{12:.4},{13:.4},{14},{15:.4}\n'.format(
                                                    info['video_name'],
                                                    info['fid'],
                                                    box[0],box[1],box[2],box[3],
                                                    box_1[0],box_1[1],box_1[2],box_1[3],
                                                    box_2[0],box_2[1],box_2[2],box_2[3],
                                                    args.label_dict[cl_ind],
                                                    s))

                t_rest.stop()
                # visualize results
                if VIS:
                    if not os.path.isdir(os.path.join(args.save_root, info['video_name'])):
                        os.makedirs(os.path.join(args.save_root, info['video_name']))
                    print (info)
                    overlay_image(os.path.join(args.data_root, info['video_name'], im_format % info['fid']),
                                  os.path.join(args.save_root, info['video_name'], im_format % info['fid']),
                                  pred_boxes = merged_result,
                                  id2class = args.id2class)

                    # write to files
                    for key in merged_result:
                        box = np.asarray(key.split(','), dtype=np.float32)
                        for l, s in merged_result[key]:
                            fout.write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6},{7:.4}\n'.format(
                                                    info['video_name'],
                                                    info['fid'],
                                                    box[0],box[1],box[2],box[3],
                                                    l, s))
            torch.cuda.synchronize()
            t1 = time.time()
            print ("Batch time: ", t1-t0)

            torch.cuda.synchronize()
            t0 = time.time()
                    
        fout.close()
        tout.close()
    print(f"base total time: {Timer.timers['base']:0.4f} seconds")
    print(f"context total time: {Timer.timers['context']:0.4f} seconds")
    print(f"2b total time: {Timer.timers['2b']:0.4f} seconds")
    print(f"rest total time: {Timer.timers['rest']:0.4f} seconds")

if __name__ == "__main__":
    main()
