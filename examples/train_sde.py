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
    parser.add_argument("--n-repeat", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-size", type=int, default=200)
    parser.add_argument("--length-train", type=int, default=20)
    parser.add_argument("--length-val", type=int, default=80)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--brownian-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--noise-type", type=str, default="additive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--num-patients", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=50.0)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--dt-num", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="neuroscience", help="wandb project name (optional)")
    args = parser.parse_args()
    dict_args = vars(args).copy()

    data_dir = Path(args.data_dir)
    cfg = TrainerConfig(**dict_args)

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Using device: {args.device}")

    # load dataset on CPU for indexing; Trainer will move model/data to configured device
    train_ds, val_ds = TimeSeriesDataset.from_directory(cfg, pattern="timeseries_*.npy", max_num=args.num_patients)
    # simple split
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=val_ds.collate_fn)

    # infer state size from first sample
    model = make_model("mlp", **dict_args)
    model = torch.compile(model).to(cfg.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("total number of paramers", total_params)

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit(seed=args.seed)


if __name__ == "__main__":
    main()
