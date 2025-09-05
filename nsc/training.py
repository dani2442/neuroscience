import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torchsde

from . import set_seed

def masked_mse(pred, target, mask):
    # pred, target: (B, T, d); mask: (B, T)
    diff = (pred - target) ** 2  # (B, T, d)
    diff = diff.mean(dim=-1)     # (B, T)
    diff = diff * mask           # zero out padded steps
    # average over valid elements only
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def train(epochs: int, model: callable, train_loader: DataLoader, dt: float, lr: float, method: str, device: str, val_loader: DataLoader|None = None, seed: int|None = None):
    if seed is not None:
        set_seed(123)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for ts_pad, y_pad, mask, x0 in train_loader:
            # ts_pad: (T_max,), y_pad: (B, T_max, d), mask: (B, T_max), x0: (B, d)
            ts_pad = ts_pad.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)
            x0 = x0.to(device)

            optimizer.zero_grad()

            x_sim = torchsde.sdeint(model, x0, ts_pad, method=method, dt=dt)  # (T_max, B, d)
            x_sim = x_sim.transpose(0, 1)  # (B, T_max, d)

            loss = masked_mse(x_sim, y_pad, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if val_loader is not None:
            pass

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch:03d} | train loss: {avg_loss:.6f}")