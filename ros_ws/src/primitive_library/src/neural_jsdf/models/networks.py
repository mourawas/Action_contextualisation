"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import copy

# torch related
import torch
import torch.nn as nn
from collections import OrderedDict
import functools


class pure_MLP(nn.Module):
    """
    Neural Distance Field by Multi-layer Perceptron
    """

    def __init__(self, input_c, output_c, h_layers=None):
        super(pure_MLP, self).__init__()
        if h_layers is None:
            h_layers = [250, 130, 50, 25]
        h_layers = copy.deepcopy(h_layers)
        h_layers.insert(0, input_c)
        h_layers.append(output_c)

        layers = OrderedDict()
        num_layers = len(h_layers)
        for i in range(num_layers - 1):
            layers['linear_{}'.format(i)] = nn.Linear(h_layers[i], h_layers[i + 1])
            if i < num_layers - 2:
                layers['relu_{}'.format(i)] = nn.ReLU()

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)

    def load_model(self, load_path):
        state_dict = torch.load(load_path)['model']
        self.load_state_dict(state_dict, strict=False)


########################################################################################################################
########################################################################################################################
# SOMETHING NEW
########################################################################################################################


class content_MLP(nn.Module):
    """
    Multi-layer Perceptron returns multi-layers feature
    """

    def __init__(self, input_c, output_c, h_layers=None):
        super(content_MLP, self).__init__()
        if h_layers is None:
            h_layers = [64, 32, 16, 8]
        self.h_layers = copy.deepcopy(h_layers)
        self.h_layers.insert(0, input_c)
        self.h_layers.append(output_c)

        layers = OrderedDict()
        num_layers = len(self.h_layers)
        for i in range(num_layers - 1):
            layers['linear_{}'.format(i)] = nn.Linear(self.h_layers[i], self.h_layers[i + 1])
            if i < num_layers - 2:
                # layers['bn_{}'.format(i)] = nn.BatchNorm1d(self.h_layers[i + 1])
                # layers['relu_{}'.format(i)] = nn.ReLU()
                layers['relu_{}'.format(i)] = nn.GELU()

        self.resBlock = resBlock(input_c)
        self.model = nn.Sequential(layers)

    def forward(self, x, feat_layers=None, fused_feats=None, fused_layers=None):
        x = self.resBlock(x)

        # feat_layers = [1, 3, 5, 7]
        if (feat_layers is not None) and (fused_feats is None):
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                # print("layer id:{}".format(layer_id), feat)
                if layer_id in feat_layers:
                    feats.append(feat)
            return feat, feats
        elif (feat_layers is not None) and (fused_feats is not None) and (fused_layers is not None):
            assert len(fused_feats) == len(fused_layers)
            feat = x
            feats = []
            fused_id = 0
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                # print("layer id:{}".format(layer_id), feat)
                if layer_id in feat_layers:
                    feat = torch.add(feat, fused_feats[fused_id])
                    fused_id += 1
                if layer_id in feat_layers:
                    feats.append(feat)
            return feat, feats
        else:
            return self.model(x), None


class resBlock(nn.Module):
    def __init__(self, dim):
        super(resBlock, self).__init__()

        self.fc_1 = nn.Linear(dim, round(dim / 2))
        self.fc_2 = nn.Linear(round(dim / 2), dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc_1(x)
        out = self.act(out)

        out = self.fc_2(out)
        out += residual
        out = self.act(out)
        return out


class resBackbone(nn.Module):
    def __init__(self, input_c=10, N=64, dropout_rate=0.2):
        super(resBackbone, self).__init__()
        self.fc_1 = nn.Linear(3 * input_c, N)
        self.fc_2 = nn.Linear(N, round(N / 2))
        self.fc_3 = nn.Linear(round(N / 2), round(N / 4))
        self.fc_4 = nn.Linear(round(N / 4), round(N / 2))
        self.fc_5 = nn.Linear(round(N / 2), N - 3 * input_c)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # position encoding
        x_skip = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        x = self.fc_1(x_skip)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_3(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc_4(x)
        x = self.act(x)

        # skip connection
        x = self.dropout(x)
        x = self.fc_5(x)
        x = self.act(x)
        x = torch.cat((x, x_skip), dim=1)

        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Identity(nn.Module):
    def forward(self, x):
        return x


class PoseRegressor(nn.Module):
    def __init__(self, input_c, dim=256):
        super(PoseRegressor, self).__init__()
        self.linear_1 = nn.Linear(input_c, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.linear_3 = nn.Linear(dim, dim)
        self.linear_pose = nn.Linear(dim, 7)
        self.bn_1 = nn.BatchNorm1d(dim, affine=True)
        self.bn_2 = nn.BatchNorm1d(dim, affine=True)
        self.bn_3 = nn.BatchNorm1d(dim, affine=True)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.act(x)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = self.act(x)

        x = self.linear_3(x)
        x = self.bn_3(x)
        x = self.act(x)

        pose = self.linear_pose(x)

        return pose


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss



