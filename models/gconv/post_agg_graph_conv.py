from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn


class DecouplePostAggGraphConv(nn.Module):
    """
    Post-aggregation graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, decouple=True, bias=True):
        super(DecouplePostAggGraphConv, self).__init__()
        self.decouple = decouple
        self.in_features = in_features
        self.out_features = out_features
        self.n_pts = adj.size(1)

        if decouple:
            self.W = nn.Parameter(torch.zeros(size=(2, self.n_pts, in_features, out_features), dtype=torch.float))
        else:
            self.W = nn.Parameter(torch.zeros(size=(1, self.n_pts, in_features, out_features), dtype=torch.float))
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

        if self.decouple:
            E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)

            h0 = torch.matmul(adj * E, input)
            h1 = torch.matmul(adj * (1 - E), input)

            output = torch.einsum('bjn,jnm->bjm', h0, self.W[0]) + torch.einsum('bjn,jnm->bjm', h1, self.W[1])
        else:
            h0 = torch.matmul(self.adj, input)
            output = torch.einsum('bjn,jnm->bjm', h0, self.W)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'