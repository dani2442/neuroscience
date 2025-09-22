"""Minimal example to train an SDE model using the nsc package.

Usage:
    python3 examples/train_sde.py

Ensure required packages are installed (see pyproject.toml).
"""
from pathlib import Path

import argparse

import torch

from nsc.dataset import TimeSeriesDataset
from nsc.model import make_model
from nsc.training import Trainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_processed/ts_young/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-repeat", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-size", type=int, default=200)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--brownian-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dt-num", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="neuroscience", help="wandb project name (optional)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cfg = TrainerConfig(epochs=args.epochs, data_dir=data_dir, device=args.device, wandb_project=args.wandb_project, dt=args.dt, dt_num=args.dt_num, state_size=args.state_size, length=args.length, brownian_size=args.brownian_size, hidden_size=args.hidden_size, batch_size=args.batch_size, lr=args.lr, n_repeat=args.n_repeat)

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Using device: {args.device}")

    # load dataset on CPU for indexing; Trainer will move model/data to configured device
    ds = TimeSeriesDataset.from_directory(cfg, pattern="timeseries_*.npy")
    # simple split
    n = len(ds)
    n_train = int(0.8 * n)
    print(f"Dataset size: {n}, train: {n_train}, val: {n - n_train}")
    train_ds = torch.utils.data.Subset(ds, list(range(0, n_train)))
    val_ds = torch.utils.data.Subset(ds, list(range(n_train, n)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=ds.collate_fn)

    # infer state size from first sample
    model = make_model("mlp", state_size=cfg.state_size, brownian_size=cfg.brownian_size, hidden_size=cfg.hidden_size)
    model = torch.compile(model).to(cfg.device)


    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit(seed=42)


if __name__ == "__main__":
    main()
