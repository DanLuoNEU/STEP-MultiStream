"""simple consensus module"""
import torch
import torch.nn as nn

# from ...builder import SEGMENTAL_CONSENSUSES


class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    def __init__(self,
                 consensus_type='avg',
                 dim=1):
        super(_SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        """forward"""
        self.shape = x.size()
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim, keepdim=True)
        else:
            output = None
        return output

    def backward(self, grad_output):
        """backward"""
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(
                self.shape) / float(self.shape[self.dim])
        else:
            grad_in = None
        return grad_in


# @SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    """simple"""
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        """init weights"""
        pass

    def forward(self, x):
        """forward"""
        self.shape = x.size()
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim, keepdim=True)
        else:
            output = None
        return output
        # return _SimpleConsensus(self.consensus_type, self.dim)(input)
