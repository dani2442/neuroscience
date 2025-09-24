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
from nsc.model import make_model
from nsc.training import TrainerConfig



def find_checkpoint(path: Path) -> Path:
    # prefer explicit file, otherwise pick the latest best_epoch_*.pt in the directory
    if path.is_file():
        return path
    pts = sorted(path.glob('best_model*.pt'))
    if not pts:
        raise SystemExit(f"No checkpoint files found in: {path}")
    return pts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='runs', help='Checkpoint file or directory containing best_epoch_*.pt')
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
    ts, y, mask, x0 = next(iter(train_loader))

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
    y = y.cpu().numpy()
    ts = ts.cpu().numpy()

    mean_sim = ys.mean(axis=0)  # (T, d)
    q5, q25, q75, q95 = np.percentile(ys, [5, 25, 75, 95], axis=0)

    # heatmap of mean simulated vs ground truth for first few dims
    ids = np.arange(0, 90, 10)
    n_ids = len(ids)
    cols = 2
    rows = math.ceil(n_ids / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, var_idx in enumerate(ids):
        ax = axes[ax_i]
        im = ax.imshow(mean_sim[:, var_idx:var_idx+1].T, aspect='auto')
        ax.set_title(f"Mean sim (var {var_idx})")
        ax.set_ylabel('dim')
        ax.set_xlabel('time')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for k in range(len(ids), len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Quantile plots for selected variables
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10), squeeze=False)
    ids = np.arange(0, 90, 10)

    for i, var_idx in enumerate(ids):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        yi = ys[:, :, var_idx].transpose(1,0)  # (N_samples, T)
        mean = yi.mean(axis=0)
        q5, q10, q25, q75, q90, q95 = np.percentile(yi, [5, 10, 25, 75, 90, 95], axis=0)

        ax.fill_between(ts, q5, q95, color='royalblue', alpha=0.15)
        ax.fill_between(ts, q10, q90, color='royalblue', alpha=0.2)
        ax.fill_between(ts, q25, q75, color='royalblue', alpha=0.35)
        
        for j in range(args.n_samples):
            ax.plot(ts, yi[j], color="purple", alpha=0.2, linewidth=1)
    
        # Mean and ground truth
        ax.plot(ts, mean, color="black", linewidth=1, label="Simulated")
        ax.plot(ts, y[0, :, var_idx], color="red", linewidth=1, label="Ground truth")
        
        ax.set_title(f"ID {var_idx}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$Y_t$")
        if i == 0:  # Legend only on first subplot
            ax.legend()

    plt.xlabel('t')
    plt.tight_layout()
    plt.savefig(f'images/sde_simulation_{ckpt_path.stem}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
