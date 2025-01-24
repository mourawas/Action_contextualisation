"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import os
import torch
from abc import ABC
import torch.utils.data
import glob
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class NDF_dataset(torch.utils.data.Dataset, ABC):
    def __init__(self, dataroot='validation', type='near'):
        self.points_files = glob.glob(os.path.join(dataroot, type + "/*.npy"))

        self.points_data = []
        self.load_data()

        self._length = len(self.points_data)

        print("Shuffling the longer side to ensure a balanced training ...")
        np.random.shuffle(self.points_data)
        print("Shuffle Done")

    def load_data(self):
        assert len(self.points_data) == 0

        pbar = tqdm(total=len(self.points_files))
        for file in self.points_files:
            data_array = np.load(file)
            self.points_data.append(data_array)
            pbar.update(1)
        del pbar
        self.points_data = np.vstack(self.points_data)

    @staticmethod
    def numpy2torch(data):
        return torch.from_numpy(data)

    @property
    def length(self):
        return self._length

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return {'sample': self.points_data[item][:10],
                'label': self.points_data[item][10:]}


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss


if __name__ == "__main__":
    # model = torch.load("JSDF/arm/SOTA_model/128_res_multi_15.pkl").to('cuda:0')
    model = torch.load("JSDF/arm/SOTA_model/mk_mlp_4.pkl").to('cuda:0')

    dataset = NDF_dataset(type='near')
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=256,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=True,
                                             shuffle=True)

    criterion = RMSELoss()
    print(len(dataset))
    overall_loss = 0
    for data in dataloader:
        x = data['sample'].to('cuda:0')
        y = data['label'].to('cuda:0')
        with torch.no_grad():
            pred = model(x)
        loss = criterion(pred, y)
        overall_loss += loss.detach()

    print(torch.sum(overall_loss, dim=0)/len(dataset))
    print(torch.mean(torch.sum(overall_loss, dim=0)/len(dataset)))

# [0.0013, 0.0013, 0.0013, 0.0013, 0.0015, 0.0017, 0.0020, 0.0015]
# far [0.0013, 0.0013, 0.0013, 0.0013, 0.0015, 0.0016, 0.0020, 0.0017]