"""Minimal example to train an SDE model using the nsc package.

Usage:
    python3 examples/train_sde.py
"""
from pathlib import Path
import argparse
import torch

from nsc.dataset import TimeSeriesDataset
from nsc.utils import set_seed, make_model
from nsc.training import Trainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_processed/ts_young/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--n-repeat", type=int, default=2*1024)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="hopf", choices=["mlp", "hopf"])
    parser.add_argument("--method", type=str, default='euler')
    parser.add_argument("--sde-type", type=str, default="ito")
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-size", type=int, default=200)
    parser.add_argument("--length-train", type=int, default=80)
    parser.add_argument("--length-val", type=int, default=80)
    parser.add_argument("--adjoint", type=bool, default=False)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--brownian-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--load_ckpt", type=str, default=None)#"runs/best_model_run_20250924-143800.pt")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--noise-type", type=str, default="additive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--num-patients", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--dt-num", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="neuroscience", help="wandb project name (optional)")
    args = parser.parse_args()
    dict_args = vars(args).copy()

    data_dir = Path(args.data_dir)
    cfg = TrainerConfig(**dict_args)

    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    set_seed(args.seed)
    print(f"Using device: {args.device}")

    # load dataset on CPU for indexing; Trainer will move model/data to configured device
    train_ds, val_ds = TimeSeriesDataset.from_directory(cfg, pattern="timeseries_*.npy", max_num=args.num_patients)
    # simple split
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=val_ds.collate_fn)

    # infer state size from first sample
    model = make_model("mlp", **dict_args)
    model = torch.compile(model).to(cfg.device)

    if args.load_ckpt:
        chk = torch.load(args.load_ckpt, weights_only=True, map_location=args.device)
        state_dict = chk["model_state_dict"]
        model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("total number of paramers", total_params)

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit(seed=args.seed)


if __name__ == "__main__":
    main()
