from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn


class NoSharingGraphConv(nn.Module):
    """
    No-sharing graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(NoSharingGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.n_pts = adj.size(1)
        self.W = nn.Parameter(torch.zeros(size=(self.n_pts, self.n_pts, in_features, out_features), dtype=torch.float))
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

        h0 = torch.einsum('bhn,hwnm->bhwm', input, self.W)
        output = torch.einsum('bhw, bhwm->bwm', adj, h0)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'