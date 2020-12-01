
import numpy as np
from torch.utils.data import Dataset


class PPGIMUData(Dataset):
    def __init__(self, input_npy_path, label_npy_path):
        super(PPGIMUData, self).__init__()
        self.x = np.load(input_npy_path)
        self.y = np.load(label_npy_path)

    def __getitem__(self, item):
        return self.x[item].T, self.y[item]

    def __len__(self):
        return len(self.x)
