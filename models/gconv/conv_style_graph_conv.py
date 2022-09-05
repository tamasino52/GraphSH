from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn


class ConvStyleGraphConv(nn.Module):
    """
    Convolution-style graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ConvStyleGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(3, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        adj = self.adj[None, :].to(input.device)

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])

        E0 = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E1 = torch.triu(torch.ones_like(adj), diagonal=1)
        E2 = 1 - E1 - E0

        output = torch.matmul(adj * E0, h0) + torch.matmul(adj * E1, h1) + torch.matmul(adj * E2, h2)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'