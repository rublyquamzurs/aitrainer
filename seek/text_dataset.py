
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, data, label, trans=None):
        self.dt = torch.asarray(data, dtype=torch.int32)
        self.lb = torch.asarray(label, dtype=torch.float32)
        self.trans = trans

    def __len__(self):
        return len(self.lb)

    def __getitem__(self, idx):
        dt = self.dt[idx]
        if self.trans:
            dt = self.trans(dt)
        return dt, self.lb[idx]
