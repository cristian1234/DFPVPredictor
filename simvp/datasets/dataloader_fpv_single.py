import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FPVSingleNPZ(Dataset):
    def __init__(self, path, pre_seq_length=10, aft_seq_length=10):
        arr = np.load(path)['data']  # (N, T, C, H, W)
        self.data = arr.astype(np.float32) / 255.0  # normalizar 0-1
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        # Calcular mean y std sobre todo el dataset
        self.mean = self.data.mean()
        self.std = self.data.std() + 1e-6  # evitar división por cero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip = self.data[idx]  # (T, C, H, W)
        x = clip[:self.pre_seq_length]
        y = clip[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length]
        return torch.from_numpy(x), torch.from_numpy(y)

def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, **kwargs):
    train_set = FPVSingleNPZ(os.path.join(data_root, 'fpv_train.npz'),
                           pre_seq_length, aft_seq_length)
    val_set = FPVSingleNPZ(os.path.join(data_root, 'fpv_val.npz'),
                         pre_seq_length, aft_seq_length)
    test_set = FPVSingleNPZ(os.path.join(data_root, 'fpv_test.npz'),
                          pre_seq_length, aft_seq_length)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader
