import typing as t
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import torch
import random
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data: list[np.ndarray], dt: float, state_size: int, length: int = 10, n_repeat: int = 10, device: str = 'cpu'):
        self.data = data
        self.dt = dt
        self.length = length
        self.device = device
        self.state_size = state_size
        self.n_repeat = n_repeat
        self._data = [self._sample_path(id) for id in range(len(data))]
    
    def _sample_path(self, idx: int):
        y = torch.tensor(self.data[idx]).to(self.device)
        h = torch.tensor(hilbert(self.data[idx], axis=0).imag).to(self.device)
        ts = self.dt*torch.arange(y.shape[0]).to(self.device)
        return ts, y, h

    def __len__(self):
        return len(self._data)*self.n_repeat

    def __getitem__(self, idx):
        ts, y, h = self._data[idx%len(self._data)]
        n = len(y)
        m = min(n, self.length)
        start = random.randint(0, n-m)
        return ts[start:start+m], y[start:start+m], h[start:start+m]
    
    @staticmethod
    def from_directory(cfg, pattern: str = "*.npy", max_num: int = 100):
        p = Path(cfg.data_dir)
        
        files_train = []
        files_test = []

        dir = sorted(list(p.glob(pattern)))[:max_num]
        print(f"Found {len(dir)} files in {p} matching {pattern}")
        print("The first n files are: ", dir[:5])
        for file in dir:
            x = np.load(file)
            i = int(x.shape[0] * cfg.split)
            files_train.append(x[:i])
            files_test.append(x[i:])

        train_dataset = TimeSeriesDataset(files_train, dt=cfg.dt, state_size=cfg.state_size, length=cfg.length_train, n_repeat=cfg.n_repeat, device=cfg.device)
        test_dataset = TimeSeriesDataset(files_test, dt=cfg.dt, state_size=cfg.state_size, length=cfg.length_val, n_repeat=cfg.n_repeat, device=cfg.device)

        return train_dataset, test_dataset

    @staticmethod
    def collate_fn(batch):
        # batch: list of (ts, y, h)
        ts_list = [b[0] for b in batch]
        y_list = [b[1] for b in batch]
        h_list = [b[2] for b in batch]

        # Stack tensors (assumes all have same length)
        y_batch = torch.stack(y_list)
        h_batch = torch.stack(h_list)
        
        # Create relative time vector based on the first sample
        # Assuming constant dt and equal length for all samples in batch
        ts = ts_list[0]
        ts = ts - ts[0] # Shift to start at 0

        x0 = torch.cat([y_batch[:, 0, :], h_batch[:, 0, :]], dim=-1)
        y_batch = torch.cat([y_batch, h_batch], dim=-1)
        
        return ts, y_batch, x0
    


    def dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)
