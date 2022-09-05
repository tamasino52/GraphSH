from __future__ import absolute_import
import torch
import torch.nn as nn
from functools import reduce
from common.graph_utils import adj_mx_from_edges

from models.gconv.vanilla_graph_conv import DecoupleVanillaGraphConv
from models.gconv.pre_agg_graph_conv import DecouplePreAggGraphConv
from models.gconv.post_agg_graph_conv import DecouplePostAggGraphConv
from models.gconv.conv_style_graph_conv import ConvStyleGraphConv
from models.gconv.no_sharing_graph_conv import NoSharingGraphConv
from models.gconv.modulated_gcn_conv import ModulatedGraphConv

from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, gcn_type=None):
        super(_GraphConv, self).__init__()

        if gcn_type == 'vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'convst':
            self.gconv = ConvStyleGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'nosharing':
            self.gconv = NoSharingGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'modulated':
            self.gconv = ModulatedGraphConv(input_dim, output_dim, adj)
        else:
            assert False, 'Invalid graph convolution type'

        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _Hourglass(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim1, hid_dim2, nodes_group, p_dropout, gcn_type):
        super(_Hourglass, self).__init__()

        adj_mid = adj_mx_from_edges(8, [[0, 2], [1, 2], [2, 3], [3, 7], [4, 7], [5, 7], [6, 7]], sparse=False)
        adj_low = adj_mx_from_edges(4, [[0, 1], [1, 2], [2, 3]], sparse=False)

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim1, p_dropout, gcn_type)
        self.gconv2 = _GraphConv(adj_mid, hid_dim1, hid_dim2, p_dropout, gcn_type)
        self.gconv3 = _GraphConv(adj_low, hid_dim2, hid_dim2, p_dropout, gcn_type)
        self.gconv4 = _GraphConv(adj_mid, hid_dim2, hid_dim1, p_dropout, gcn_type)
        self.gconv5 = _GraphConv(adj, hid_dim1, output_dim, p_dropout, gcn_type)

        self.pool = _SkeletalPool(nodes_group)
        self.unpool = _SkeletalUnpool(nodes_group)

    def forward(self, x):
        skip1 = x
        skip2 = self.gconv1(skip1)
        skip3 = self.gconv2(self.pool(skip2))
        out = self.gconv3(self.pool(skip3))
        out = self.gconv4(self.unpool(out) + skip3)
        out = self.gconv5(self.unpool(out) + skip2)
        return out + skip1


class _SkeletalPool(nn.Module):
    def __init__(self, nodes_group):
        super(_SkeletalPool, self).__init__()
        self.nodes_group = sum(nodes_group, [])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.shape[1] == 16:
            out = self.pool(x[:, self.nodes_group].transpose(1, 2))
            return out.transpose(1, 2)
        elif x.shape[1] == 8:
            out = self.pool(x.transpose(1, 2))
            return out.transpose(1, 2)
        else:
            assert False, 'Invalid Type in Skeletal Pooling : x.shape is {}'.format(x.shape)


class _SkeletalUnpool(nn.Module):
    def __init__(self, nodes_group):
        super(_SkeletalUnpool, self).__init__()
        self.nodes_group = sum(nodes_group, [])
        self.inv_nodes_group = [0 for _ in range(len(self.nodes_group))]
        for inv, i in enumerate(self.nodes_group):
            self.inv_nodes_group[i] = inv

        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        if x.shape[1] == 8:
            return self.unpool(x.transpose(1, 2)).transpose(1, 2)[:, self.inv_nodes_group]
        elif x.shape[1] == 4:
            return self.unpool(x.transpose(1, 2)).transpose(1, 2)
        else:
            assert False, 'Invalid Type in Skeletal Unpooling : x.shape is {}'.format(x.shape)


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SEBlock(nn.Module):
    def __init__(self, adj, input_dim, p_dropout=0., reduction_ratio=8):
        super(SEBlock, self).__init__()
        hid_dim = input_dim // reduction_ratio
        self.fc1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.fc2 = nn.Linear(hid_dim, input_dim, bias=True)
        self.gap = nn.AvgPool1d(kernel_size=adj.shape[-1])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        out = self.gap(x)
        out = self.relu(self.drop(self.fc1(out.squeeze())))
        out = self.sigmoid(self.fc2(out))

        return x * out[:,:,None]


class GraphSH(nn.Module):
    def __init__(self, adj, hid_dim, nodes_group, coords_dim=(2, 3), num_layers=4, p_dropout=None, gcn_type=None):
        super(GraphSH, self).__init__()

        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout, gcn_type=gcn_type)
        self.num_layers = num_layers
        _gconv_layers = []
        _conv_layers = []

        group_size = len(nodes_group[0])
        assert group_size > 1

        grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
        restored_order = [0] * len(grouped_order)
        for i in range(len(restored_order)):
            for j in range(len(grouped_order)):
                if grouped_order[j] == i:
                    restored_order[i] = j
                    break

        for i in range(num_layers):
            _gconv_layers.append(_Hourglass(adj, hid_dim, hid_dim, int(hid_dim * 1.5), hid_dim * 2, nodes_group, p_dropout, gcn_type))
            _conv_layers.append(nn.Conv1d(hid_dim, hid_dim // num_layers, 1))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.conv_layers = nn.ModuleList(_conv_layers)

        self.gconv_output = nn.Sequential(SEBlock(adj, hid_dim), nn.Conv1d(hid_dim, coords_dim[1], 1))

    def forward(self, x):
        out = self.gconv_input(x)
        inter_fs = []
        for l in range(self.num_layers):
            out = self.gconv_layers[l](out)
            inter_fs.append(self.conv_layers[l](out.transpose(1,2)).transpose(1,2))
        f_out = torch.cat(inter_fs, dim=2)
        out = self.gconv_output(f_out.transpose(1,2)).transpose(1,2)
        return out
