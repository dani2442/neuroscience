"""Minimal example to train an SDE model using the nsc package.

Usage:
    python3 examples/train_sde.py
"""
from pathlib import Path
import argparse
import torch

from nsc.dataset import TimeSeriesDataset
from nsc.utils import set_seed, make_model
from nsc.training import Trainer, TrainerConfig, LatentTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="hopf", choices=["mlp", "linear", "latent", "hopf"])
    parser.add_argument("--data-dir", type=str, default="data_processed/ts_young/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--n-repeat", type=int, default=256)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-3)
    parser.add_argument("--method", type=str, default='midpoint')
    parser.add_argument("--sde-type", type=str, default="stratonovich", choices=["ito", "stratonovich"])
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-size", type=int, default=200)
    parser.add_argument("--length-train", type=int, default=60)
    parser.add_argument("--length-val", type=int, default=60)
    parser.add_argument("--adjoint", type=bool, default=False)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--brownian-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--load-ckpt", type=str, default="") #runs/hopf_0.187745_20260114-153307.pt")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--noise-type", type=str, default="diagonal", choices=["diagonal", "additive", "general"])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--num-patients", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5)
    parser.add_argument("--loss", type=str, default="ks+mae", choices=["mse", "mae", "ks", "ks+mae"])
    parser.add_argument("--dt-num", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="neuroscience", help="wandb project name (optional)")
    parser.add_argument("--likelihood", type=str, default="laplace", choices=["laplace", "normal"])
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--kl-anneal-iters", type=int, default=1000)
    parser.add_argument("--latent-hidden", type=int, default=200)
    parser.add_argument("--latent-layers", type=int, default=2)
    parser.add_argument("--prior-theta", type=float, default=1.0)
    parser.add_argument("--prior-mu", type=float, default=0.0)
    parser.add_argument("--prior-sigma", type=float, default=0.5)
    parser.add_argument("--adaptive", type=bool, default=False)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
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
    collate_fn = TimeSeriesDataset.collate_fn
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # build model
    model = make_model(**dict_args).to(cfg.device)
    model = torch.compile(model).to(cfg.device)

    if args.load_ckpt:
        chk = torch.load(args.load_ckpt, weights_only=True, map_location=args.device)
        state_dict = chk["model_state_dict"]
        model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("total number of paramers", total_params)

    if args.model_type == "latent":
        trainer = LatentTrainer(model, train_loader, val_loader, cfg)
    else:
        trainer = Trainer(model, train_loader, val_loader, cfg) 
        
    trainer.fit(seed=args.seed)


if __name__ == "__main__":
    main()
