"""
@Author: Yiting CHEN
@Time: 2023/8/24 下午4:30
@Email: chenyiting@whu.edu.cn
version: python 3.9
Created by PyCharm
"""
import torch
from models import multi_head_MLP, ResMultiMLP, mk_MLP

if __name__ == "__main__":
    model = mk_MLP()
    model.load_state_dict(torch.load("JSDF/arm/SOTA_model/mk_mlp_5.pth")["model"])
    torch.save(model, "mk_mlp_5.pkl")
