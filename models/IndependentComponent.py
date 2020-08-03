'''Independent Component Layers in Pytorch.'''
'''(Both static and dynamic) '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

cfg = {
    "STATIC-BATCH" : ['BN', .05],
    "STATIC-GROUP" : ['GN', .05],
    "STATIC-INSTANCE" : ['IN', .05],
    "STATIC-LAYER" : ['LN', .05],
    "DYNAMIC-BATCH" : ['BN', 'r'],
    "DYNAMIC-GROUP" : ['GN', 'r'],
    "DYNAMIC-INSTANCE" : ['IN', 'r'],
    "DYNAMIC-LAYER" : ['LN', 'r']
}

class IndependentComponentLayer(nn.Module):
    def __init__(self, ic_type, H, W, in_channels=3, dimensions='2D'):
        super(IndependentComponentLayer, self).__init__()
        self.p_value = -1;
        self.max_p = .1  # 1 # formally .8;
        self.a = 1
        self.b = 0

        self.features = self._make_icl(cfg[ic_type], H, W, in_channels, dimensions)
        # print("p value set to: ", self.p_value)

    def update_p(self):
        _former = self.p_value
        self.p_value = self.max_p * np.random.beta(self.a, self.b);
        print("p value set to: ", self.p_value, " (from ", _former,")")
        return

    def get_p(self):
        return self.p_value

    def set_a_and_b(self, correct, incorrect):
        self.a = float(incorrect) / 1000.0
        self.b = float(correct) / 1000.0
        print("a set to: ", self.a, " & b set to: ", self.b)
        return
    
    def forward(self, x):
        out = self.features(x)
        out = F.dropout(out, p=self.p_value, training=self.training)
        return out

    def _make_icl(self, cfg, H, W, in_channels, dimensions):
        layers = []
        in_channels = in_channels
        num_groups = 32
        HW = H * W

        if cfg[0] == 'BN':
            if dimensions == '1D':
                layers.append(nn.BatchNorm1d(num_features=HW))
            elif dimensions == '2D':
                layers.append(nn.BatchNorm2d(num_features=in_channels))
            elif dimensions == '3D':
                layers.append(nn.BatchNorm3d(num_features=in_channels))

        elif cfg[0] == 'GN':
            layers.append(nn.GroupNorm(num_groups,in_channels))

        elif cfg[0] == 'IN':
            if dimensions == '1D':
                layers.append(nn.InstanceNorm1d(HW))
            elif dimensions == '2D':
                layers.append(nn.InstanceNorm2d(HW))
            elif dimensions == '3D':
                layers.append(nn.InstanceNorm3d(HW))

        elif cfg[0] == 'LN':
            layers.append(nn.LayerNorm([in_channels, H, W]))

        if cfg[1] != 'r':
            self.p_value = cfg[1];
        elif cfg[1] == 'r':
            self.p_value = 0;

        # layers.append(nn.Dropout(self.p_value))

        return nn.Sequential(*layers)


