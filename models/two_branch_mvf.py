""" Build Two Branch Model based on MVF module
    05/17/2021, Dan
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import weights_init
sys.path.append('../')
from data.mvf_cls import TEM_REDUCE
from utils.tube_utils import encode_coef
from .rec2d_mvf import Recognizer2D_MVF
from models.utils_mvf.checkpoint import load_checkpoint

__all__ = ['TwoBranchNet']


def build_mvf_cls( mvf_pretrain=None, num_classes=3, freeze_affine=True):
    """
    """
    print("Building MVF head for Global Branch...")
    mvf = Recognizer2D_MVF(num_cls=num_classes)
    if mvf_pretrain is not None:
        if os.path.isfile(mvf_pretrain):
            print ("Loading MVF-Resnet50 pretrained on Kinetics dataset from {}...".format(mvf_pretrain))
            # i3d.load_state_dict(torch.load(kinetics_pretrain))
            load_checkpoint(mvf, mvf_pretrain, map_location='cpu')
        else:
            raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(mvf_pretrain))
    # Build Base Model from the Whole Model
    model = nn.Sequential(mvf.backbone.layer4)
    cls_head = mvf.cls_head
    ################### If Freeze Batch Normalization ###############
    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        model.apply(set_bn_fix)
    #################################################################
    return model, cls_head


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck_resample(nn.Module):
    """ 
    Arguments:
        inplanes:     input channels
        outplanes:    output channels
        planes:       intermediate channels
    Output:
        features:
    """
    def __init__(self, inplanes, outplanes, planes, stride=1):
        super(Bottleneck_resample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv4 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = self.conv1(x)

        out = self.conv2(x)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        
        out += residual
        out = self.relu(out)

        return out


class TwoBranchNet(nn.Module):
    def __init__(self, cfg, cls_only=False):
        super(TwoBranchNet, self).__init__()
        self.T = cfg.T
        self.freeze_stats = cfg.freeze_stats
        self.freeze_affine = cfg.freeze_affine
        self.mvf_pretrain = cfg.kinetics_pretrain
        self.fc_dim = cfg.fc_dim
        self.dropout_prob = cfg.dropout
        self.pool_size = cfg.pool_size
        self.fp16 = cfg.fp16
        self.cls_only = cls_only
        self.num_classes = cfg.num_classes
        
        self.mvf_head, self.cls_head = build_mvf_cls(mvf_pretrain=self.mvf_pretrain,num_classes=self.num_classes)

        if not self.cls_only:
            self.local_conv = nn.Sequential(
                    # C(global_feat) + C(global_feat_conv)
                    Bottleneck_resample((128+256)*TEM_REDUCE, 1024, 256),
                    Bottleneck(1024, 256),
                    Bottleneck(1024, 256))
            self.downsample_loc = nn.Conv2d(1024, self.fc_dim,
                        kernel_size=1, stride=1, bias=True)
            self.dropout = nn.Dropout(self.dropout_prob)
            self.local_reg = nn.Linear(self.fc_dim * 7 **2, 4)
            self.neighbor_reg1 = nn.Linear(self.fc_dim * 7 **2, 4)    # for tube t-1
            self.neighbor_reg2 = nn.Linear(self.fc_dim * 7 **2, 4)    # for tube t+1

        self._init_net()


    def _init_net(self):
        if not self.cls_only:
            self.local_conv.apply(weights_init)
            self.downsample_loc.apply(weights_init)
            self.local_reg.apply(weights_init)
            self.neighbor_reg1.apply(weights_init)
            self.neighbor_reg2.apply(weights_init)
    
    def set_device(self, device):
        self.device = device

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        # Mode Train: not update the running statistics
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    if self.fp16:
                        m.eval().half()
                    else:
                        m.eval()

            if self.freeze_stats:
                self.mvf_head.apply(set_bn_eval)
                self.cls_head.apply(set_bn_eval)

    def forward(self, global_feat, tubes=None, targets=None):
        """
        Args:
            global_feat: ROI pooled feat for global branch. Shape: [num_tubes, T, C, W, H]
            tubes: flatten proposal tubes. Shape: [num_tubes, T*TEM_REDUCE, 5]   (dim 0 is batch idx)
                   should have absolute value of the position
            targets: flatten target tubes. 
                    for training: Shape: [num_tubes, 3, boxes(0:3)+1+1+num_classes]   
                            (:4 is regression labels
                             4 is indicator for classification,
                             5 is indicator for regression, 
                             6: is label for action classfication)
                   should have absolute value of the position
                   [:, 0] is for tube t-1
                   [:, 1] is for center tube
                   [:, -1] is for tube t+1
        """

        global_feat = global_feat.to(self.device)
        N, T, C, W, H = global_feat.size() # (num_tubes x batch_size, args.T x TEM_REDUCE, 1024, 7, 7)

        # chunks = int(T / self.T)
        chunks = int(T/TEM_REDUCE/self.T)
        chunk_idx = [j*self.T + int(self.T/2) for j in range(chunks)]    # used to index the middel frame of each chunk
        half_T = int(self.T/2)
        #### global branch ####
        # Make sure Input feature from Pooled is 14x14
        if H != 14:    
            global_feat_upsample = F.interpolate(global_feat, scale_factor=(1,int(14/W),int(14/H))) # (4,24,1024,14,14)
            N_tubes, N_frames, C_upsample, W_upsample ,H_upsample = global_feat_upsample.shape
            global_feat_conv = self.mvf_head(global_feat_upsample.view(-1,C_upsample,W_upsample,H_upsample)) #(96,2048,7,7)
        else:
            global_feat_conv = self.mvf_head(global_feat.view(-1, C, W, H))
        global_class = self.cls_head(global_feat_conv, num_seg=TEM_REDUCE)
        global_class = global_class.view(N, -1, self.num_classes).mean(1)

        #### local branch ####

        local_loc, first_loc, last_loc = torch.tensor([0.]).to(global_class), torch.tensor([0.]).to(global_class), torch.tensor([0.]).to(global_class)
        if not self.cls_only:
            # TODO: T needs to be clarified
            _, C_conv, W_conv, H_conv = global_feat_conv.shape
            global_feat_downsample = F.interpolate(global_feat, scale_factor=(0.125,W_conv/W,H_conv/H)) #(num_tubes x batch_size, TEM_REDUCE, 1024, 14, 14) -> (num_tubes x batch_size, TEM_REDUCE, 1024, 7, 7)
            global_feat_conv_downsample = F.interpolate(global_feat_conv.view(N, T, -1, W_conv, H_conv), scale_factor=(0.125, 1, 1)) #(num_tubes x batch_size, TEM_REDUCE, 2048, 14, 14) -> (num_tubes x batch_size, TEM_REDUCE, 1024, 7, 7)
            local_feat = torch.cat([global_feat_downsample.view(-1,128*TEM_REDUCE,W_conv,H_conv), global_feat_conv_downsample.view(-1,256*TEM_REDUCE,W_conv,H_conv)], dim=1) # (34x1,(1024+2048)x8,7,7)
            local_feat = self.local_conv(local_feat)
            local_feat = self.downsample_loc(local_feat)
            local_feat = self.dropout(local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_loc = self.local_reg(local_feat).view(N, int(T/TEM_REDUCE), -1)
    
            first_loc = local_loc[:, chunk_idx[0]-half_T:chunk_idx[0]+half_T+1].contiguous().clone()
            last_loc = local_loc[:, chunk_idx[-1]-half_T:chunk_idx[-1]+half_T+1].contiguous().clone()
    
            # neighbor prediction
            first_loc += self.neighbor_reg1(local_feat.view(N,int(T/TEM_REDUCE),-1)[:, chunk_idx[0]-half_T:chunk_idx[0]+half_T+1].contiguous().view(-1, self.fc_dim*7**2)).view(N,self.T,-1)
            last_loc += self.neighbor_reg2(local_feat.view(N,int(T/TEM_REDUCE),-1)[:, chunk_idx[-1]-half_T:chunk_idx[-1]+half_T+1].contiguous().view(-1, self.fc_dim*7**2)).view(N,self.T,-1)
    
            center_pred = local_loc[:, chunk_idx[int(chunks/2)]].contiguous().view(N, -1)
            first_pred = first_loc[:, half_T].contiguous().view(N, -1)
            last_pred = last_loc[:, half_T].contiguous().view(N, -1)

        #### compute losses ####

        loss_global_cls = torch.tensor(0.).to(global_class)
        loss_local_loc = torch.tensor(0.).to(local_loc)
        loss_neighbor_loc = torch.tensor(0.).to(local_loc)
        if targets is not None:
            tubes = tubes.to(self.device)
            targets = targets.to(self.device)

            center_targets = targets[:, 1].contiguous()
            first_targets = targets[:, 0].contiguous()
            last_targets = targets[:, -1].contiguous()
            center_tubes = tubes[:, chunk_idx[int(chunks/2)]].contiguous()
            first_tubes = tubes[:, chunk_idx[0]].contiguous()
            last_tubes = tubes[:, chunk_idx[-1]].contiguous()

            ######### classification loss for center clip #########

            with torch.no_grad():
                mask = center_targets[:, 4].view(-1, 1)
            if mask.sum():
                target = center_targets[:, 6:] * mask    # mask out background samples
                loss_global_cls = F.binary_cross_entropy_with_logits(global_class, 
                        target, reduction='none')

            if not self.cls_only:
                ######### regression loss for center clip #########
    
                # transform target to regression parameterization
                center_targets_loc = center_targets[:, :4].clone()
                center_targets_loc = encode_coef(center_targets_loc, center_tubes.view(-1,5)[:, 1:])
    
                with torch.no_grad():
                    mask = center_targets[:, 5].view(-1, 1).repeat(1,4)
                if mask.sum():
                    loss_local_loc = F.smooth_l1_loss(center_pred, center_targets_loc, reduction='none')
                    loss_local_loc = torch.sum(loss_local_loc * mask.detach()) / torch.sum(mask.detach())    # masked average
    
                ######### regression loss for neighbor clips #########
    
                # transform target to regression parameterization
                first_targets_loc = first_targets[:, :4].clone()
                last_targets_loc = last_targets[:, :4].clone()
                neighbor_targets_loc = torch.cat([first_targets_loc, last_targets_loc], dim=0)
                neighbor_targets_loc = encode_coef(neighbor_targets_loc, 
                                        torch.cat([first_tubes.view(-1,5)[:, 1:],
                                                   last_tubes.view(-1,5)[:, 1:]], dim=0))
    
                with torch.no_grad():
                    first_mask = first_targets[:, 5].view(-1, 1).repeat(1,4)
                    last_mask = last_targets[:, 5].view(-1, 1).repeat(1,4)
                    neighbor_mask = torch.cat([first_mask, last_mask], dim=0)
    
                if neighbor_mask.sum():
                    neighbor_loc = torch.cat([first_pred, last_pred], dim=0)
    
                    loss_neighbor_loc = F.smooth_l1_loss(neighbor_loc, neighbor_targets_loc, reduction='none')
                    loss_neighbor_loc = torch.sum(loss_neighbor_loc * neighbor_mask.detach()) / torch.sum(neighbor_mask.detach())

        #### Output ####

        global_prob = torch.sigmoid(global_class)
        loss_global_cls = loss_global_cls.view(-1)
        loss_local_loc = loss_local_loc.view(-1)
        loss_neighbor_loc = loss_neighbor_loc.view(-1)

        return global_prob, local_loc, first_loc, last_loc, loss_global_cls, loss_local_loc, loss_neighbor_loc

