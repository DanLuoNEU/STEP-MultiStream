""" Build Base Model
    05/17/2021, Dan
"""

# System
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
# Package
from .rec2d_mvf import Recognizer2D_MVF
from models.utils_mvf.checkpoint import load_checkpoint


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def build_base_mvf(mvf_pretrain="pretrained/mvf_R50_8x8.pth", freeze_affine=True):
    # Building Whole Model
    print ("Building Recognizer2D-ResNet50-MVF model...")
    mvf = Recognizer2D_MVF(module_cfg=dict(
                            type='MVF',
                            n_segment=8,
                            alpha=0.125,
                            mvf_freq=(0, 0, 1, 1),
                            mode='THW'))

    if mvf_pretrain is not None:
        if os.path.isfile(mvf_pretrain):
            print ("Loading MVF pretrained on Kinetics dataset from {}...".format(mvf_pretrain))
            # i3d.load_state_dict(torch.load(kinetics_pretrain))
            load_checkpoint(mvf, mvf_pretrain, map_location='cpu')
        else:
            raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(mvf_pretrain))
    # Build Base Model from the Whole Model
    base_model = nn.Sequential(mvf.backbone.conv1,
                               mvf.backbone.bn1,
                               mvf.backbone.relu,
                               mvf.backbone.maxpool,
                               mvf.backbone.layer1,
                               mvf.backbone.layer2,
                               mvf.backbone.layer3
                            )
    ################### If Freeze Batch Normalization ###############
    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        base_model.apply(set_bn_fix) 
    #################################################################
    
    return base_model


class BaseNet(nn.Module):
    """
    Backbone network of the model
    """

    def __init__(self, cfg):
        super(BaseNet, self).__init__()
        
        self.resnet_pretrain = cfg.resnet_pretrain
        self.kinetics_pretrain = cfg.kinetics_pretrain
        self.freeze_stats = cfg.freeze_stats
        self.freeze_affine = cfg.freeze_affine
        # self.fp16 = cfg.fp16

        self.base_model = build_base_mvf(mvf_pretrain=self.kinetics_pretrain, freeze_affine=self.freeze_affine)

    def forward(self, imgs, **kwargs):
        """
        Applies network layers on input images
        Args:
            img_group: input image sequences. Shape: [batch_size, T, C, W, H]
        """
        num_batch, T, C, W, H = imgs.shape
        imgs = imgs.reshape((-1, C)+ imgs.shape[3:])
        num_seg = imgs.shape[0] // num_batch
        conv_feat = self.base_model(imgs)

        return conv_feat

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)