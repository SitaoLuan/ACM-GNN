import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar as https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, model_type, output_layer=0):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type = (
            in_features,
            out_features,
            output_layer,
            model_type,
        )
        self.low_act, self.high_act, self.mlp_act = (
            nn.ELU(alpha=3),
            nn.ELU(alpha=3),
            nn.ELU(alpha=3),
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = (
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
            )
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
            )
            self.low_param, self.high_param, self.mlp_param = (
                Parameter(torch.FloatTensor(1, 1).cuda()),
                Parameter(torch.FloatTensor(1, 1).cuda()),
                Parameter(torch.FloatTensor(1, 1).cuda()),
            )
            self.attention_param = Parameter(
                torch.FloatTensor(3 * out_features, 3).cuda()
            )

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = (
                Parameter(torch.FloatTensor(in_features, out_features)),
                Parameter(torch.FloatTensor(in_features, out_features)),
                Parameter(torch.FloatTensor(in_features, out_features)),
            )
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
                Parameter(torch.FloatTensor(out_features, 1)),
                Parameter(torch.FloatTensor(out_features, 1)),
                Parameter(torch.FloatTensor(out_features, 1)),
            )
            self.low_param, self.high_param, self.mlp_param = (
                Parameter(torch.FloatTensor(1, 1)),
                Parameter(torch.FloatTensor(1, 1)),
                Parameter(torch.FloatTensor(1, 1)),
            )
            self.attention_param = Parameter(torch.FloatTensor(3 * out_features, 3))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)
        self.attention_param.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T,
            1,
        )
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        if self.model_type == "mlp":
            output_mlp = torch.mm(input, self.weight_mlp)
            return output_mlp
        elif self.model_type == "sgc" or self.model_type == "gcn":
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == "acmgcn":
            output_low = F.relu(torch.spmm(adj_low, torch.mm(input, self.weight_low)))
            output_high = F.relu(
                torch.spmm(adj_high, torch.mm(input, self.weight_high))
            )
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )
        elif self.model_type == "acmsgc":
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(
                adj_high, torch.mm(input, self.weight_high)
            )
            output_mlp = torch.mm(input, self.weight_mlp)

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_mlp * output_mlp
                + self.att_high * output_high
            )
