import torch.nn as nn
import torch.nn.functional as F
from baseline_models.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, model_type):
        super(GCN, self).__init__()
        self.gcns = nn.ModuleList()
        self.model_type = model_type
        if self.model_type == "mlp":
            self.gcns.append(GraphConvolution(nfeat, nhid, model_type=model_type))
            self.gcns.append(
                GraphConvolution(nhid, nclass, model_type=model_type, output_layer=1)
            )
        elif self.model_type == "gcn" or self.model_type == "acmgcn":
            self.gcns.append(GraphConvolution(nfeat, nhid, model_type=model_type))
            self.gcns.append(
                GraphConvolution(nhid, nclass, model_type=model_type, output_layer=1)
            )
        elif self.model_type == "sgc" or self.model_type == "acmsgc":
            self.gcns.append(GraphConvolution(nfeat, nclass, model_type=model_type))
        self.dropout = dropout

    def forward(self, x, adj_low, adj_high):
        if self.model_type == "acmgcn" or self.model_type == "acmsgc":
            x = F.dropout(x, self.dropout, training=self.training)

        fea = self.gcns[0](x, adj_low, adj_high)  #

        if self.model_type == "gcn" or self.model_type == "mlp":
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)
        elif self.model_type == "acmgcn":
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)
        elif self.model_type == "sgc" or self.model_type == "acmsgc":
            pass
        return fea
