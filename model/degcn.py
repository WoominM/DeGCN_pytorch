import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Basic_Block


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)            

    
class DeGCN(nn.Sequential):
    def __init__(self, block_args, A, k, eta):
        super(DeGCN, self).__init__()
        for i, [in_channels, out_channels, stride, residual, num_frame, num_joint] in enumerate(block_args):
            self.add_module(f'block-{i}_tcngcn', Basic_Block(in_channels, 
                                                             out_channels, 
                                                             A, 
                                                             k,
                                                             eta,
                                                             stride=stride, 
                                                             num_frame=num_frame, 
                                                             num_joint=num_joint, 
                                                             residual=residual))  
            

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, k=8, eta=4, num_stream=2, 
                 graph=None, graph_args=dict(), in_channels=3, drop_out=0,):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args) 

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        base_frame = 64
        
        self.blockargs = [
            [in_channels, base_channel, 1, False, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel*2, 2, True, base_frame, num_point],
            [base_channel*2, base_channel*2, 1, True, base_frame//2, num_point],
            [base_channel*2, base_channel*2, 1, True, base_frame//2, num_point],
            [base_channel*2, base_channel*4, 2, True, base_frame//2, num_point],
            [base_channel*4, base_channel*4, 1, True, base_frame//4, num_point],
            [base_channel*4, base_channel*4, 1, True, base_frame//4, num_point]
        ]
        
        self.num_stream = num_stream
        self.streams = nn.ModuleList([DeGCN(self.blockargs, A, k, eta) for _ in range(self.num_stream)])
        self.fc = nn.ModuleList([nn.Linear(base_channel*4, num_class) for _ in range(self.num_stream)])
        
        for fc in self.fc:
            nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)        
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        
    
    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x_ = x  
        out = []
        for stream, fc in zip(self.streams, self.fc):
            x = x_
            x = stream(x)
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)
            x = self.drop_out(x)
            out.append(fc(x))

        return out