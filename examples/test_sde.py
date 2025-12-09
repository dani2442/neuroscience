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
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchsde
from torchsde import BrownianInterval

from nsc.dataset import TimeSeriesDataset
from nsc.nsde import make_model
from nsc.training import TrainerConfig
from nsc.utils import plot_heatmap, plot_quantile_trajectories



def find_checkpoint(path: Path) -> Path:
    # prefer explicit file, otherwise pick the latest best_epoch_*.pt in the directory
    if path.is_file():
        return path
    pts = sorted(path.glob('best_model*.pt'))
    if not pts:
        raise SystemExit(f"No checkpoint files found in: {path}")
    return pts[-1]


def compute_metrics(y_true: np.ndarray, y_sim: np.ndarray) -> dict:
    """Compute MSE and KL (approx via NLL) for different time intervals.
    
    Args:
        y_true: (1, T, D)
        y_sim: (S, T, D)
    """
    T = y_true.shape[1]
    # Define intervals
    intervals = {
        'short': slice(0, T // 3),
        'med': slice(T // 3, 2 * T // 3),
        'long': slice(2 * T // 3, T),
        'full': slice(0, T)
    }
    
    metrics = {}
    mu_sim = y_sim.mean(axis=0) # (T, D)
    std_sim = y_sim.std(axis=0) + 1e-6 # (T, D)
    
    for name, sl in intervals.items():
        # MSE: || mu_sim - y_true ||^2
        mse = np.mean((mu_sim[sl] - y_true[0, sl])**2)
        metrics[f'mse_{name}'] = mse
        
        # KL approx (NLL): -log N(y_true | mu_sim, std_sim)
        # = 0.5 * log(2pi) + log(sigma) + 0.5 * ((y - mu)/sigma)^2
        nll = 0.5 * np.log(2 * np.pi) + np.log(std_sim[sl]) + \
              0.5 * ((y_true[0, sl] - mu_sim[sl]) / std_sim[sl])**2
        metrics[f'kl_{name}'] = np.mean(nll)
        
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='runs/best_model_run_20250924-113704.pt', help='Checkpoint file or directory containing best_epoch_*.pt')
    parser.add_argument('--data-dir', type=str, default='data_processed/ts_young/')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of stochastic sample paths to simulate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--num-patients", type=int, default=1)
    #parser.add_argument('--ids', type=str, default='0,1,2,3,4,5,6,7,8', help='Comma separated variable indices to plot')
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt_path = find_checkpoint(Path(args.ckpt))
    print(f"Using checkpoint: {ckpt_path}")

    chk = torch.load(ckpt_path, weights_only=False, map_location=device)
    state_dict, cfg_dict = chk["model_state_dict"], chk.get("cfg", None)

    # Build a TrainerConfig from saved cfg when available; else use defaults
    cfg_dict['length_train'] = 150
    cfg_dict['length_val'] = 150
    if cfg_dict:
        cfg = TrainerConfig(**cfg_dict)
    else:
        cfg = TrainerConfig()
    cfg.device = device

    # create model architecture matching saved config
    model = make_model('mlp', **cfg_dict)
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

    # Create BrownianInterval for multiple samples
    # repeat x0 to match n_samples
    x0_batch = x0.repeat(args.n_samples, 1)
    if cfg.noise_type=='diagonal':
        size = (args.n_samples, cfg.state_size)
    else:
        size = (args.n_samples, cfg.brownian_size)
    bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=size, device=device, dt=cfg.dt, levy_area_approximation="space-time")

    with torch.no_grad():
        ts = ts.to(device)
        y = y.to(device)
        x0_batch = x0_batch.to(device)
        y_sim = torchsde.sdeint(model, x0_batch, ts, bm=bm, dt=cfg.dt_num)
    ys = y_sim.cpu().numpy()
    # Transpose to (S, T, D) for easier handling
    ys = np.transpose(ys, (1, 0, 2))
    y = y.cpu().numpy()
    ts = ts.cpu().numpy()

    mean_sim = ys.mean(axis=0)  # (T, d)

    # heatmap of mean simulated vs ground truth for first few dims
    ids = np.arange(0, 90, 10)
    plot_heatmap(mean_sim, ids)

    # Quantile plots for selected variables
    plot_quantile_trajectories(ts, y, ys, ids, output_path=f'images/sde_simulation_{ckpt_path.stem}.png')

    # Metrics
    metrics = compute_metrics(y, ys)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
