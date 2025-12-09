from __future__ import annotations

import math
import random
import typing as t

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		# torch might not be installed in the environment running static analysis
		pass


def make_model(name: str, **kwargs) -> t.Callable:
    name = name.lower()
    if name == "linear":
        from nsc.nsde import LinearSDE
        return LinearSDE(**kwargs)
    elif name in {"mlp", "mlpsde", "mlp_sde"}:
        from nsc.nsde import MLPSDE
        return MLPSDE(**kwargs)
    elif name in {"hopf", "hopf_coupled", "hopf_coupled_sde"}:
        from .hopf import HopfCoupledSDE
        return HopfCoupledSDE(**kwargs)
    raise ValueError(f"Unknown model name: {name}")


def plot_heatmap(mean_sim: np.ndarray, ids: np.ndarray, show: bool = True) -> None:
    """Plot heatmap of mean simulated trajectories."""
    n_ids = len(ids)
    cols = 2
    rows = math.ceil(n_ids / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, var_idx in enumerate(ids):
        ax = axes[ax_i]
        # mean_sim is (T, D)
        im = ax.imshow(mean_sim[:, var_idx:var_idx+1].T, aspect='auto')
        ax.set_title(f"Mean sim (var {var_idx})")
        ax.set_ylabel('dim')
        ax.set_xlabel('time')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for k in range(len(ids), len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()


def plot_quantile_trajectories(ts: np.ndarray, y_true: np.ndarray, y_sim: np.ndarray, ids: np.ndarray, output_path: str = None, show: bool = True) -> None:
    """Plot quantile bands and samples for selected variables.
    
    Args:
        ts: Time steps (T,)
        y_true: Ground truth (1, T, D)
        y_sim: Simulated samples (S, T, D)
        ids: Indices of variables to plot
    """
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10), squeeze=False)
    
    for i, var_idx in enumerate(ids):
        if i >= n_rows * n_cols:
            break
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        yi = y_sim[:, :, var_idx]  # (S, T)
        mean = yi.mean(axis=0)
        q5, q10, q25, q75, q90, q95 = np.percentile(yi, [5, 10, 25, 75, 90, 95], axis=0)

        ax.fill_between(ts, q5, q95, color='royalblue', alpha=0.15)
        ax.fill_between(ts, q10, q90, color='royalblue', alpha=0.2)
        ax.fill_between(ts, q25, q75, color='royalblue', alpha=0.35)
        
        for j in range(yi.shape[0]):
            ax.plot(ts, yi[j], color="purple", alpha=0.2, linewidth=1)
    
        # Mean and ground truth
        ax.plot(ts, mean, color="black", linewidth=1, label="Simulated")
        ax.plot(ts, y_true[0, :, var_idx], color="red", linewidth=1, label="Ground truth")
        
        ax.set_title(f"ID {var_idx}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$Y_t$")
        if i == 0:  # Legend only on first subplot
            ax.legend()

    plt.xlabel('t')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    if show:
        plt.show()