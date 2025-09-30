# simvp/datasets/fpv_single.py
import numpy as np, torch, os
from torch.utils.data import Dataset

class FPVSingleNPZ(Dataset):
    def __init__(self, path_npz, pre=10, aft=10):
        self.pre, self.aft = pre, aft
        arr = np.load(path_npz)  # si pesa mucho, considera .npy + mmap
        self.data = arr["data"]  # (N, T, C, H, W) o (N, T, H, W) si gris
        if self.data.ndim == 4:  # (N, T, H, W) -> agrega canal
            self.data = self.data[:, :, None, ...]
        self.N, self.T, self.C, self.H, self.W = self.data.shape
        assert self.T >= self.pre + self.aft

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        seq = self.data[i]               # (T, C, H, W)
        x = torch.from_numpy(seq[:self.pre]).float()   # (pre, C, H, W)
        y = torch.from_numpy(seq[self.pre:self.pre+self.aft]).float()
        return x, y
