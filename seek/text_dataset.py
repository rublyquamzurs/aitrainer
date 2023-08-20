
from torch.utils.data import Dataset
import torch
import numpy as np


VECTOR_LEN = 80


class TextDataset(Dataset):
    def __init__(self, data, label, trans=None):
        self.char_map = {"-": 1, "0": 2, "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8, "7": 9, "8": 10, "9": 11,
                         "a": 12, "b": 13, "c": 14, "d": 15, "e": 16, "f": 17, "g": 18, "h": 19, "i": 20, "j": 21,
                         "k": 22, "l": 23, "m": 24, "n": 25, "o": 26, "p": 27, "q": 28, "r": 29, "s": 30, "t": 31,
                         "u": 32, "v": 33, "w": 34, "x": 35, "y": 36, "z": 37}
        trd, trl = self._transform(data, label)
        self.dt = torch.asarray(trd, dtype=torch.int32)
        self.lb = torch.asarray(trl, dtype=torch.float32)
        self.trans = trans

    def __len__(self):
        return len(self.lb)

    def __getitem__(self, idx):
        dt = self.dt[idx]
        if self.trans:
            dt = self.trans(dt)
        return dt, self.lb[idx]

    @classmethod
    def get_dump_ass(cls, raw_data):
        return cls(raw_data[:, 0], raw_data[:, 1])

    def _transform(self, data, label):
        count = data.shape[0]
        t = np.zeros(shape=[count, VECTOR_LEN], dtype=np.int32)
        lb = np.zeros(shape=[count, 2], dtype=np.float32)
        for i in range(count):
            t[i] = self.__get_flat_vector(data[i])
            lb[i][int(label[i])] = 1.0
        return t, lb

    def __get_flat_vector(self, domain: str):
        out = np.zeros(shape=VECTOR_LEN, dtype=np.int32)
        if len(domain) > VECTOR_LEN:
            raise ValueError("domain %s is beyond %d" % (domain, VECTOR_LEN))
        for i, c in enumerate(domain):
            if c in self.char_map:
                out[i] = self.char_map[c]
            else:
                out[i] = 0
        return out
