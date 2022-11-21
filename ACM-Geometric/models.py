import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, MLP
import torch
import numpy as np
from torch.nn.parameter import Parameter
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from torch_sparse import SparseTensor, matmul
import time
import argparse
import tqdm
import math

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, nnodes, dropout, model_type, structure_info, variant=False):
        super(GCN, self).__init__()
        if model_type =='acmgcnpp':
            self.mlpX = MLP(nfeat, nhid, nhid, num_layers=1, dropout=0)
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.structure_info, self.nlayers, self.nnodes = model_type, structure_info, nlayers, nnodes
        
        if self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':
            self.gcns.append(GraphConvolution(nfeat, nhid, nnodes, model_type = model_type, variant = variant,  structure_info = structure_info))
            self.gcns.append(GraphConvolution(1*nhid, nclass, nnodes, model_type = model_type, output_layer=1, variant = variant, structure_info = structure_info))
        elif self.model_type =='acmsgc':
            self.gcns.append(GraphConvolution(nfeat, nclass, model_type = model_type))
        elif self.model_type =='acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution(k * nhid + nfeat, nhid, model_type = model_type, variant = variant))
            self.gcns.append(GraphConvolution(nlayers * nhid + nfeat, nclass, model_type = model_type, variant = variant))
        self.dropout = dropout
        self.fea_param, self.xX_param  = Parameter(torch.FloatTensor(1,1).to(device)), Parameter(torch.FloatTensor(1,1).to(device))          
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.model_type =='acmgcnpp':
            self.mlpX.reset_parameters()
        else:
            pass

    def forward(self, x, adj_low, adj_high, adj_low_unnormalized):

        if self.model_type =='acmgcn' or self.model_type =='acmsgc' or self.model_type =='acmsnowball' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':

            x = F.dropout(x, self.dropout, training=self.training)
            if self.model_type =='acmgcnpp':
                xX = F.dropout(F.relu(self.mlpX(x, input_tensor=True)), self.dropout, training=self.training)
        if self.model_type =='acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea1 = (self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized))
        
        if  self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp': 
            
            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)
            
            if self.model_type =='acmgcnpp':
                fea2 = self.gcns[1](fea1+xX, adj_low, adj_high, adj_low_unnormalized)
            else:
                fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)
        return fea2
