import torch
from torch.utils.data.dataset import Dataset
import numpy 

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with shape [Time, Batch, Features]"""
    def __init__(self, data, window_length: int = 100, duplicate: int= 1):
        # Load data: expected shape [Time, Batch, Features]
        self.y = torch.tensor(data["y"])
        self.u = torch.tensor(data["u"])
        self.t = torch.tensor(data["t"])
        self.T, self.B, self.F = self.y.shape
        self.dt = self.t[1]-self.t[0]
        self.duplicate = duplicate
        self.window_length = self.T if self.T<window_length else window_length
        
    def __len__(self):
        return self.duplicate*self.B
    
    def __getitem__(self, idx):
        start_idx = torch.randint(0, self.T-self.window_length+1, (1,)).item()
        end_idx = start_idx + self.window_length
        ts = self.dt*torch.arange(0,self.window_length)
        # Return single trajectory: [Time, Features]
        return (
            self.y[start_idx:end_idx, idx%self.B], 
            self.u[start_idx:end_idx, idx%self.B],
            ts
        )
