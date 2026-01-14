"""Load a trained SDE model checkpoint and visualize simulation vs data.

Usage:
    python3 examples/test_sde.py --ckpt runs/best_epoch_016_val_1.000133.pt

This script will:
 - instantiate the model architecture via nsc.model.make_model
 - load model weights from a checkpoint
 - load a small batch from the dataset
 - run multiple stochastic simulations using torchsde.BrownianInterval
 - plot a heatmap of mean simulated trajectories and quantile bands for a few dimensions
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import torchsde
from torchsde import BrownianInterval

from nsc.dataset import TimeSeriesDataset
from nsc.training import TrainerConfig
from nsc.utils import (
    compute_metrics,
    find_checkpoint,
    make_model,
    plot_heatmap,
    plot_quantile_trajectories,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='runs/hopf_0.188578_20260114-152642.pt', help='Checkpoint file or directory containing best_epoch_*.pt')
    parser.add_argument('--model-type', type=str, choices=['mlp', 'hopf', 'linear', 'latent'], help='Override model type; otherwise inferred from checkpoint cfg or --model')
    parser.add_argument('--data-dir', type=str, default='data_processed/ts_young/')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of stochastic sample paths to simulate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--num-patients", type=int, default=1)
    #parser.add_argument('--ids', type=str, default='0,1,2,3,4,5,6,7,8', help='Comma separated variable indices to plot')
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    ckpt_path = find_checkpoint(Path(args.ckpt))
    print(f"Using checkpoint: {ckpt_path}")

    chk = torch.load(ckpt_path, weights_only=False, map_location=device)
    state_dict, cfg_dict = chk["model_state_dict"], chk.get("cfg") or {}

    # Build a TrainerConfig from saved cfg when available; else use defaults
    cfg_dict = dict(cfg_dict)  # shallow copy so we can tweak without side effects
    cfg_dict['length_train'] = 150
    cfg_dict['length_val'] = 150
    cfg = TrainerConfig(**cfg_dict)
    cfg.device = device

    model_type = (args.model_type or cfg_dict.get("model_type") or args.model).lower()
    print(f"Detected model type: {model_type}")

    # create model architecture matching saved config
    model = make_model(**cfg_dict)
    model.to(device)
    #model = torch.compile(model)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # load dataset and get one batch (we'll use small batch_size=1 for visualization)
    # load dataset on CPU for indexing; Trainer will move model/data to configured device
    train_ds, val_ds = TimeSeriesDataset.from_directory(cfg, pattern="timeseries_*.npy", max_num=args.num_patients)
    # simple split
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)
    ts, y, x0 = next(iter(val_loader))

    sdeint_fn = torchsde.sdeint_adjoint if getattr(cfg, "adjoint", False) else torchsde.sdeint

    with torch.no_grad():
        ts = ts.to(device)
        y = y.to(device)
        if model_type == "latent":
            # Sample from the learned approximate posterior (same path KL objective as training)
            y_sim, _ = model(
                ts=ts,
                batch_size=args.n_samples,
                sdeint_fn=sdeint_fn,
                method=cfg.method,
                dt=cfg.dt_num,
                adaptive=getattr(cfg, "adaptive", False),
                rtol=getattr(cfg, "rtol", 1e-3),
                atol=getattr(cfg, "atol", 1e-3),
            )
        else:
            # Create BrownianInterval for multiple samples
            x0_batch = x0.repeat(args.n_samples, 1).to(device)
            if cfg.noise_type == 'diagonal':
                size = (args.n_samples, cfg.state_size)
            else:
                size = (args.n_samples, cfg.brownian_size)
            bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=size, device=device, dt=cfg.dt, levy_area_approximation="space-time")
            y_sim = sdeint_fn(model, x0_batch, ts, bm=bm, dt=cfg.dt_num, method=cfg.method)
    ys = y_sim.cpu().numpy()
    # Transpose to (S, T, D) for easier handling
    ys = np.transpose(ys, (1, 0, 2))
    y = y.cpu().numpy()
    ts = ts.cpu().numpy()

    mean_sim = ys.mean(axis=0)  # (T, d)

    # heatmap of mean simulated vs ground truth for first few dims
    ids = np.arange(0, min(90, mean_sim.shape[1]), 10)
    #plot_heatmap(mean_sim, ids)

    # Quantile plots for selected variables
    plot_quantile_trajectories(ts, y, ys, ids, output_path=f'images/sim_{model_type}_{ckpt_path.stem}.png')

    # Metrics
    metrics = compute_metrics(y, ys)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
