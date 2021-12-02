""" Recognizer2D Using MVF Module Based on resnet50
    https://github.com/whwu95/MVFNet/blob/main/codes/models/recognizers/recognizer2d.py
    05/16/2021, Dan Luo
"""
# System
import logging
from abc import ABCMeta, abstractmethod
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as cp
from torch.nn.functional import softmax
from torch.nn.modules.batchnorm import _BatchNorm
# Package-BaseNet
from .utils_mvf.logger import get_root_logger
from .utils_mvf.norm import get_norm_type, build_norm_layer
from .utils_mvf.weight_init import constant_init, kaiming_init
from .utils_mvf.checkpoint import load_checkpoint
from .utils_mvf.fp16 import auto_fp16

######################### BaseNet ###################################
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super(BasicBlock, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm2_name, norm2)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """norm2"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """forward"""
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 avd=False,
                 avd_first=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        self.avd = avd and stride > 1
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        # self.bn1 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)

        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        self.add_module(self.norm3_name, norm3)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """norm2"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """norm3"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """forward"""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.avd and self.avd_first:
                out = self.avd_layer(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.avd and not self.avd_first:
                out = self.avd_layer(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   norm_cfg=None,
                   with_cp=False,
                   avg_down=False,
                   avd=False,
                   avd_first=False):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.

    Returns:
        A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        norm_type = get_norm_type(norm_cfg)
        down_layers = []
        if avg_down:
            if dilation == 1:
                down_layers.append(
                    nn.AvgPool2d(kernel_size=stride,
                                 stride=stride,
                                 ceil_mode=True,
                                 count_include_pad=False))
            else:
                down_layers.append(
                    nn.AvgPool2d(kernel_size=1,
                                 stride=1,
                                 ceil_mode=True,
                                 count_include_pad=False))
            down_layers.append(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False))
        else:
            down_layers.append(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))
        down_layers.append(norm_type(planes * block.expansion))
        downsample = nn.Sequential(*down_layers)

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            avd=avd,
            avd_first=avd_first))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation,
                  style=style, norm_cfg=norm_cfg, with_cp=with_cp,
                  avd=avd, avd_first=avd_first))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str): Name of pretrained model. Default None.
        num_stages (int): Resnet stages. Default 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default `pytorch`.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default -1.
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default True.
        bn_frozen (bool): Whether to freeze weight and bias of BN layersn
            Default False.
        partial_bn (bool): Whether to freeze weight and bias of **all
            but the first** BN layersn Default False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 norm_frozen=False,
                 partial_norm=False,
                 with_cp=False,
                 avg_down=False,
                 avd=False,
                 avd_first=False,
                 deep_stem=False,
                 stem_width=64):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.norm_frozen = norm_frozen
        self.partial_norm = partial_norm
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_width * 2 if deep_stem else 64
        norm_type = get_norm_type(norm_cfg)
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2,
                          padding=1, bias=False),
                norm_type(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3,
                          stride=1, padding=1, bias=False),
                norm_type(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3,
                          stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.inplanes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                avg_down=avg_down,
                avd=avd,
                avd_first=avd_first)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    def init_weights(self):
        """init weight"""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained,
                            map_location='cpu', strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """forward"""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        """train"""
        super(ResNet, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    if self.norm_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_norm:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, _BatchNorm):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.norm1.parameters():
                param.requires_grad = False
            self.norm1.eval()
            self.norm1.weight.requires_grad = False
            self.norm1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

######################### Cls Head ##################################
class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        spatial_size=7,
        dropout_ratio=0.8,
        in_channels=1024,
        num_classes=101,
        init_std=0.001,
    ):
        super(BaseHead, self).__init__()
        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)

        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.Logits = None

    @abstractmethod
    def forward(self, x):
        pass

    def init_weights(self):
        pass

    def loss(self, cls_score, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        losses['loss_cls'] = F.cross_entropy(cls_score, labels)

        return losses


class TSNClsHead(BaseHead):
    """ cls head for 2D input"""
    def __init__(
        self,
        spatial_type='avg',
        spatial_size=7,
        consensus_cfg=dict(type='avg', dim=1),
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.8,
        in_channels=1024,
        num_classes=101,
        init_std=0.001,
        fcn_testing=False,
    ):
        super(TSNClsHead, self).__init__(spatial_size, dropout_ratio,
                                         in_channels, num_classes, init_std)
        self.spatial_type = spatial_type
        self.consensus_type = consensus_cfg['type']
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.cls_pool_size = (self.temporal_feature_size,
                              self.spatial_feature_size,
                              self.spatial_feature_size)
        self.with_avg_pool = with_avg_pool

        if self.consensus_type == 'avg':
            from .utils_mvf.segmental_consensuses import SimpleConsensus
            self.segmental_consensus = SimpleConsensus(
                self.consensus_type, consensus_cfg['dim'])

        elif self.consensus_type in ['TRN', 'TRNmultiscale']:
            from .utils_mvf.segmental_consensuses import return_TRN
            # consensus_cfg = dict(type='TRN', num_frames=3)
            self.segmental_consensus = return_TRN(
                self.consensus_type, in_channels,
                consensus_cfg['num_frames'], num_classes)
        else:
            raise NotImplementedError

        if self.spatial_size == -1:
            self.pool_size = (1, 1)
            if self.spatial_type == 'avg':
                self.Logits = nn.AdaptiveAvgPool2d(self.pool_size)
            if self.spatial_type == 'max':
                self.Logits = nn.AdaptiveMaxPool2d(self.pool_size)
        else:
            self.pool_size = self.spatial_size
            if self.spatial_type == 'avg':
                self.Logits = nn.AvgPool2d(
                    self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.Logits = nn.MaxPool2d(
                    self.pool_size, stride=1, padding=0)

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d(self.cls_pool_size)

        if self.consensus_type in ['TRN', 'TRNmultiscale']:
            img_feature_dim = 256
            self.new_fc = nn.Linear(self.in_channels, img_feature_dim)
        else:
            self.new_fc = nn.Linear(self.in_channels, self.num_classes)

        self.fcn_testing = fcn_testing
        self.new_cls = None


    def forward(self, x, num_seg):
        """forward"""
        if not self.fcn_testing:
            # [4*3*10 2048 7 7]
            x = self.Logits(x)
            # [4*3*10 2048 1 1]
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
                # [8*10 2048 1 1 1]
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.dropout is not None:
                x = self.dropout(x)

            x = x.view(x.size(0), -1)  # [4*3*10 2048]
            cls_score = self.new_fc(x)  # [4*3*10 400]
            # [1 4*3*10 400]
            cls_score = cls_score.reshape((-1, num_seg) + cls_score.shape[1:])
            cls_score = self.segmental_consensus(cls_score)  # [1 1 400]
            cls_score = cls_score.squeeze(1)  # [1 400]
            return cls_score
        else:
            # [3*10 2048 4 8 8]
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(
                    self.in_channels,
                    self.num_classes,
                    1, 1, 0).cuda()
                self.new_cls.load_state_dict(
                    {'weight': self.new_fc.weight.unsqueeze(-1).unsqueeze(
                        -1).unsqueeze(-1),
                     'bias': self.new_fc.bias})
            class_map = self.new_cls(x)
            # [3*10 400 4 8 8]
            cls_score = class_map.mean([2, 3, 4])  # [3*10 400]
            return cls_score

    def init_weights(self):
        """init weight"""
        nn.init.normal_(self.new_fc.weight, 0, self.init_std)
        nn.init.constant_(self.new_fc.bias, 0)


class Recognizer2D_MVF(nn.Module, metaclass=ABCMeta):
    """Recognizer2D with MVF inserted into ResNet"""
    def __init__(self, 
                 modality='RGB',
                #  fcn_testing=False,
                 fcn_testing=False, # Added for testing
                 backbone=dict(
                        type='ResNet',
                        pretrained='pretrained/resnet50.pth',
                        depth=50,
                        out_indices=(3,),
                        norm_eval=False,
                        partial_norm=False,
                        norm_cfg=dict(type='BN', requires_grad=True),
                            ),
                 cls_head=dict(
                        type='TSNClsHead',
                        spatial_size=-1,
                        spatial_type='avg',
                        with_avg_pool=False,
                        temporal_feature_size=1,
                        spatial_feature_size=1,
                        dropout_ratio=0.5,
                        in_channels=2048,
                        init_std=0.01,
                        num_classes=400),
                 module_cfg=dict(
                            type='MVF',
                            n_segment=8,
                            alpha=0.125,
                            mvf_freq=(0, 0, 1, 1),
                            mode='THW'),
                 nonlocal_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_cls=400):
        # prepare the backbone and classification head
        super(Recognizer2D_MVF, self).__init__()
        self.backbone = ResNet(pretrained='pretrained/resnet50.pth',
                               depth=50,
                               out_indices=(3,),
                               norm_eval=False,
                               partial_norm=False,
                               norm_cfg=dict(type='BN', requires_grad=True))
        self.cls_head = TSNClsHead(spatial_size=-1,
                                   spatial_type='avg',
                                   with_avg_pool=False,
                                   temporal_feature_size=1,
                                   spatial_feature_size=1,
                                   dropout_ratio=0.5,
                                   in_channels=2048,
                                   init_std=0.01,
                                   fcn_testing=fcn_testing, # Added for testing
                                   num_classes=num_cls)
        self.with_cls_head = hasattr(self, 'cls_head') and self.cls_head is not None
        self.init_weights()
        # Configurations
        self.fp16_enabled = False
        self.fcn_testing = fcn_testing
        self.modality = modality
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.module_cfg = module_cfg
        # insert module into backbone
        if self.module_cfg:
            self._prepare_base_model(backbone, self.module_cfg, nonlocal_cfg)
        # Configure the input channel
        assert modality in ['RGB', 'Flow', 'RGBDiff']
        if modality in ['Flow', 'RGBDiff']:
            length = 5
            if modality == 'Flow':
                self.in_channels = 2 * length
            # elif modality == 'RGBDiff':
            #     self.in_channels = 3 * length
            # self._construct_2d_backbone_conv1(self.in_channels)
        elif modality == 'RGB':
            self.in_channels = 3
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')
    
    def init_weights(self):
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def _prepare_base_model(self, backbone, module_cfg, nonlocal_cfg):
        # module_cfg, example:
        # tsm: dict(type='tsm', n_frames=8 , n_div=8,
        #           shift_place='blockres',
        #           temporal_pool=False, two_path=False)
        # nolocal: dict(n_segment=8)
        backbone_name = backbone['type']

        if 'type' in module_cfg:
            module_name = module_cfg.pop('type')
        else:
            module_name = 'MVF'
        self.module_name = module_name
        if backbone_name == 'ResNet':
            # Add module for 2D backbone
            if module_name == 'MVF':
                # print('Adding MVF module...')
                from .utils_mvf.MVF import make_multi_view_fusion
                make_multi_view_fusion(self.backbone, **module_cfg)

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x
    
    @auto_fp16(apply_to=('img_group', ))
    def forward(self, img_group, label, return_loss=True,
                return_numpy=True, **kwargs):
        if return_loss:
            return self.forward_train(img_group, label, **kwargs)
        else:
            return self.forward_test(img_group, return_numpy, **kwargs)
    

    def forward_train(self, imgs, labels, **kwargs):
        """train"""
        #  [B S C H W]
        #  [BS C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[3:])
        num_seg = imgs.shape[0] // num_batch

        x = self.extract_feat(imgs)  # 64 2048 7 7
        losses = dict()
        if self.with_cls_head:
            temporal_pool = imgs.shape[0] // x.shape[0]
            cls_score = self.cls_head(x, num_seg // temporal_pool)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, **kwargs):
        """test"""
        #  imgs: [B tem*crop*clip C H W]
        #  imgs: [B*tem*crop*clip C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[3:])
        num_seg = imgs.shape[0] // num_batch

        x = self.extract_feat(imgs)
        if self.with_cls_head:
            if self.fcn_testing:
                # view to 3D, [120, 2048, 8, 8] -> [30, 4, 2048, 8, 8]
                temporal_pool = imgs.shape[0] // x.shape[0]
                x = x.reshape((-1, self.module_cfg['n_segment']//temporal_pool) + x.shape[1:])
                x = x.transpose(1, 2)  # [30, 2048, 4, 8, 8]
                cls_score = self.cls_head(x, num_seg//temporal_pool)  # [30 400]
                cls_score = softmax(cls_score, 1).mean(0, keepdim=True)
            else:
                # [120 2048 8 8]
                temporal_pool = imgs.shape[0] // x.shape[0]
                cls_score = self.cls_head(x, num_seg // temporal_pool)

        if return_numpy:
            return cls_score.cpu().numpy()
        else:
            return cls_score

    def average_clip(self, cls_score):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        return:
            torch.Tensor: Averaged class score.
        """
        if self.test_cfg is None:
            self.test_cfg = {}
            self.test_cfg['average_clips'] = None

        if 'average_clips' not in self.test_cfg.keys():
            # self.test_cfg['average_clips'] = None
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips == 'prob':
            cls_score = softmax(cls_score, dim=1).mean(dim=0, keepdim=True)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=0, keepdim=True)
        return cls_score