"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import torch
import torch.nn as nn
import numpy as np
from .networks import resBackbone, content_MLP, pure_MLP, PoseRegressor


class multi_head_MLP(nn.Module):
    def __init__(self, input_c=10, N=256, head_layers=None):
        super(multi_head_MLP, self).__init__()
        assert N > (3 * input_c + 10)
        if head_layers is None:
            head_layers = [128, 64, 32, 16]
        self.linear_1 = nn.Linear(3 * input_c, N)
        self.linear_2 = nn.Linear(N, N)
        self.linear_3 = nn.Linear(N, N - 3 * input_c)
        self.linear_4 = nn.Linear(N, N)
        self.relu = nn.ReLU()
        self.mlp_0 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_1 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_2 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_3 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_4 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_5 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_6 = pure_MLP(N, 1, h_layers=head_layers)
        self.mlp_7 = pure_MLP(N, 1, h_layers=head_layers)

    def forward(self, x):
        x_skip = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        x = self.linear_1(x_skip)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.relu(x)
        x = self.linear_4(x)
        self.relu(x)
        x_0 = self.mlp_0(x)
        x_1 = self.mlp_1(x)
        x_2 = self.mlp_2(x)
        x_3 = self.mlp_3(x)
        x_4 = self.mlp_4(x)
        x_5 = self.mlp_5(x)
        x_6 = self.mlp_6(x)
        x_7 = self.mlp_7(x)
        pred_x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        return pred_x

    def load_model(self, load_path):
        state_dict = torch.load(load_path)['model']
        self.load_state_dict(state_dict, strict=True)


class mk_MLP(nn.Module):
    """
    Original Implementation from M. Koptev "Implicit Distance Functions: Learning and Applications in Control"
    """

    def __init__(self, input_c=10, output_c=8, N=256):
        super(mk_MLP, self).__init__()
        assert N > (3 * input_c + 10)
        self.linear_1 = nn.Linear(3 * input_c, N)
        self.linear_2 = nn.Linear(N, N)
        self.linear_3 = nn.Linear(N, N - 3 * input_c)
        self.linear_4 = nn.Linear(N, N)
        self.linear_5 = nn.Linear(N, output_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        x = self.linear_1(x_skip)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.linear_4(x)
        self.relu(x)
        x = self.linear_5(x)

        return x

    def load_model(self, load_path):
        state_dict = torch.load(load_path)['model']
        self.load_state_dict(state_dict, strict=True)


######################################################################################################################
# Novelty Part
#######################################################################################################################


class ResMultiMLP(nn.Module):
    def __init__(self, input_c=10, N=64, start_layer_index=1, end_layer_index=2, dropout_rate=0.2):
        super(ResMultiMLP, self).__init__()
        self.start_index = start_layer_index
        self.end_index = end_layer_index

        assert N > (3 * input_c + 10)
        self.backbone = resBackbone(input_c=input_c, N=N, dropout_rate=dropout_rate)
        self.mlp_0 = content_MLP(input_c=N, output_c=1)
        self.mlp_1 = content_MLP(input_c=N, output_c=1)
        self.mlp_2 = content_MLP(input_c=N, output_c=1)
        self.mlp_3 = content_MLP(input_c=N, output_c=1)
        self.mlp_4 = content_MLP(input_c=N, output_c=1)
        self.mlp_5 = content_MLP(input_c=N, output_c=1)
        self.mlp_6 = content_MLP(input_c=N, output_c=1)
        self.mlp_7 = content_MLP(input_c=N, output_c=1)

    def forward(self, x):
        # after each Relu function
        # layers = np.arange(len(self.mlp_0.h_layers[1:-1]))
        layers = np.arange(len(self.mlp_0.h_layers[self.start_index: self.end_index]))
        layers = 2*layers + 1

        x = self.backbone(x)
        x_0, feats = self.mlp_0(x, feat_layers=layers)
        x_1, feats = self.mlp_1(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_2, feats = self.mlp_2(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_3, feats = self.mlp_3(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_4, feats = self.mlp_4(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_5, feats = self.mlp_5(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_6, feats = self.mlp_6(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_7, _ = self.mlp_7(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        pred_x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        return pred_x

    def load_model(self, load_path):
        state_dict = torch.load(load_path)['model']
        self.load_state_dict(state_dict, strict=True)


class ResMultiMLPv2(nn.Module):
    def __init__(self, input_c=10, N=64, start_layer_index=0, end_layer_index=1, dropout_rate=0.2):
        super(ResMultiMLPv2, self).__init__()
        self.start_index = start_layer_index
        self.end_index = end_layer_index

        assert N > (3 * input_c + 10)
        self.backbone = resBackbone(input_c=input_c, N=N, dropout_rate=dropout_rate)
        self.mlp_0 = content_MLP(input_c=N, output_c=1)
        self.mlp_1 = content_MLP(input_c=N, output_c=1)
        self.mlp_2 = content_MLP(input_c=N, output_c=1)
        self.mlp_3 = content_MLP(input_c=N, output_c=1)
        self.mlp_4 = content_MLP(input_c=N, output_c=1)
        self.mlp_5 = content_MLP(input_c=N, output_c=1)
        self.mlp_6 = content_MLP(input_c=N, output_c=1)
        self.mlp_7 = content_MLP(input_c=N, output_c=1)
        self.PoseMLP = PoseRegressor(input_c=N, dim=2*N)

    def forward(self, x):
        # after each Relu function
        # layers = np.arange(len(self.mlp_0.h_layers[1:-1]))
        layers = np.arange(len(self.mlp_0.h_layers[self.start_index: self.end_index]))
        layers = 2*layers + 1
        x = self.backbone(x)
        pred_pose = self.PoseMLP(x)

        x_0, feats = self.mlp_0(x, feat_layers=layers)
        x_1, feats = self.mlp_1(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_2, feats = self.mlp_2(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_3, feats = self.mlp_3(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_4, feats = self.mlp_4(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_5, feats = self.mlp_5(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_6, feats = self.mlp_6(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        x_7, _ = self.mlp_7(x, feat_layers=layers, fused_feats=feats, fused_layers=layers)
        pred_x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        return pred_x, pred_pose

    def load_model(self, load_path):
        state_dict = torch.load(load_path)['model']
        self.load_state_dict(state_dict, strict=True)


# test model initialization
if __name__ == "__main__":
    from torch.autograd import Variable
    from torchsummary import summary
    from torchstat import stat
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim_data = Variable(torch.rand(4, 10)).to(device)
    # sim_data = Variable(torch.rand(4, 256)).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # net = pure_MLP(256, 1, h_layers=[128, 64, 32, 16]).to(device)
    # net = mk_MLP(10, 8).to(device)
    # net = ResMultiMLP(N=64).to(device)
    # net = content_MLP(128, 1).to(device)
    net = multi_head_MLP(N=64).to(device)
    print(count_parameters(net))



    # print(net)
    x, ee = net(sim_data)
    print(x.shape)
    print(ee.shape)
    # print(net.h_layers)
    # feat, feats = net(sim_data, feat_layers=[1])
    # feat, feats = net(sim_data, feat_layers=[1, 3, 5, 7], fused_feats=feats, fused_layers=[1, 3, 5, 7])
    # print(feats[0].shape)

    # print(feat)

