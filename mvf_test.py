"""This file is used to test mvf-backbone network
    05/26/2021, Dan
"""
# System
import os
import glob
import time
import numpy as np
from datetime import datetime
from collections import OrderedDict
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
# Packages
from models import ROINet
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap

from models.rec2d_mvf import Recognizer2D_MVF
from models.utils_mvf.checkpoint import load_checkpoint

def test1():
    """
    Running time and GPU usage computation
    For Report on 05/25/2021
    """
    model = Recognizer2D_MVF()
    load_checkpoint(model, "pretrained/mvf_R50_8x8.pth", map_location='cpu')
    model.cuda(0)
    list_timer_all,list_timer_mvf, list_timer_roi=[],[],[]
    tic_all = time.perf_counter()
    for i_e in range(100):
        tic_mvf = time.perf_counter()
        imgs_test = torch.rand((1,8,3,224,224)).cuda(0)
        c_s = model(return_loss=False, return_numpy=False, img_group=imgs_test, label=None)
        toc_mvf = time.perf_counter()
        list_timer_mvf.append(toc_mvf-tic_mvf)
        
        tic_roi = time.perf_counter()
        roi_net=ROINet('align',7).cuda(0)
        roi_net.eval()
        conv_feat = torch.rand(1,9,832,25,25).cuda(0)
        for i in range(3):
            tubes=torch.rand(34,3,5).cuda(0)
            pooled_feat = roi_net(conv_feat, tubes)
        toc_roi = time.perf_counter()
        list_timer_roi.append(toc_roi-tic_roi)

    toc_all = time.perf_counter()
    print("timer_all_avg:", (toc_all-tic_all)/100)
    print("timer_mvf_avg:", np.asarray(list_timer_mvf).mean())
    print("timer_roi_avg:", np.asarray(list_timer_roi).mean())
    exit()

def test2():
    pass

if __name__ == '__main__':
    test1()
    test2()