
import numpy as np
import torch
from torch.utils.data import Dataset


class PPGIMUData(Dataset):
    def __init__(self, input_npy_path, label_npy_path):
        super(PPGIMUData, self).__init__()
        self.x = torch.from_numpy(np.load(input_npy_path)).float()
        self.y = torch.from_numpy(np.load(label_npy_path)).float()

    def __getitem__(self, item):
        return self.x[item].T, self.y[item][0]

    def __len__(self):
        return len(self.x)
