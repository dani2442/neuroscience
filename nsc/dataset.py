import typing as t
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import torch
import random
from torch.utils.data import Dataset, DataLoader


def pad_and_mask(batch_ts: t.List[torch.Tensor], batch_y: t.List[torch.Tensor], eps: float = 1e-5):
    """Pad variable-length time-series to the same length and produce a mask.

    - batch_ts: list of 1D tensors (T_i,)
    - batch_y: list of 2D tensors (T_i, d)

    Returns:
        ts_pad: 1D tensor (T_max,) -- strictly increasing timeline suitable for solver
        y_pad: tensor (B, T_max, d) -- padded with hold-last-value
        mask: tensor (B, T_max) -- 1 for valid timesteps, 0 for padded
    """
    if len(batch_ts) == 0:
        raise ValueError("Empty batch passed to pad_and_mask")

    B = len(batch_ts)
    d = batch_y[0].shape[-1]
    lengths = [t.shape[0] for t in batch_ts]
    T_max = max(lengths)

    ts_pad_list = []
    for ts in batch_ts:
        T = len(ts)
        if T < T_max:
            # create strictly increasing padded times after the last valid ts
            Δ = eps * torch.arange(1, T_max - T + 1, dtype=ts.dtype)
            ts_ext = torch.cat([ts, ts[-1:] + Δ], dim=0)
        else:
            ts_ext = ts
        ts_pad_list.append(ts_ext)

    # now all have length T_max and strictly increasing
    # we can safely take one of them (e.g., the max elementwise) or just ts_pad_list[0]
    ts_pad = torch.stack(ts_pad_list, dim=0).max(dim=0).values  # (T_max,)

    # Pad y with last value; create mask
    y_pad = torch.zeros(B, T_max, d,)
    mask = torch.zeros(B, T_max)
    for i, (y, L) in enumerate(zip(batch_y, lengths)):
        y_pad[i, :L] = y
        if L < T_max:
            y_pad[i, L:] = y[L - 1]  # hold-last-value padding
        mask[i, :L] = 1.0

    return ts_pad, y_pad, mask


class TimeSeriesDataset(Dataset):
    def __init__(self, data: list[np.ndarray], dt: float, state_size: int, length: int = 10, n_repeat: int = 10):
        self.data = data
        self.dt = dt
        self.length = length
        self.state_size = state_size
        self.n_repeat = n_repeat
        self._data = [self._sample_path(id) for id in range(len(data))]
    
    def _sample_path(self, idx: int):
        y = torch.tensor(self.data[idx])
        h = torch.tensor(hilbert(self.data[idx], axis=0).imag)
        ts = self.dt*torch.arange(y.shape[0])
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
    def from_directory(cfg, pattern: str = "*.npy"):
        p = Path(cfg.data_dir)
        files = []
        dir = list(p.glob(pattern))
        print(f"Found {len(dir)} files in {p} matching {pattern}")
        for file in dir:
            files.append(np.load(file))
        return TimeSeriesDataset(files, dt=cfg.dt, state_size=cfg.state_size, length=cfg.length, n_repeat=cfg.n_repeat)

    @staticmethod
    def collate_fn(batch):
        # batch: list of (ts, y)
        ts_list = [b[0] for b in batch]
        y_list = [b[1] for b in batch]
        h_list = [b[2] for b in batch]
        ts_pad, y_pad_0, mask_0 = pad_and_mask(ts_list, y_list)
        ts_pad, y_pad_1, mask_1 = pad_and_mask(ts_list, h_list)

        x0 = torch.stack([y[0] for y in y_list], dim=0)  # (B, d)
        y0 = torch.stack([h[0] for h in h_list], dim=0)  # (B, d)

        y_pad = torch.cat([y_pad_0, y_pad_1], dim=-1)
        x0 = torch.cat([x0, y0], dim=-1)  # shape (B, 2d)
        return ts_pad, y_pad, mask_0, x0
    


    def dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)