from __future__ import annotations

import math
import random
import typing as t
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats


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


def make_model(model_type: str, *args, **kwargs) -> t.Callable:
    model_type = model_type.lower()
    if model_type == "linear":
        from nsc.nsde import LinearSDE
        return LinearSDE(**kwargs)
    elif model_type in {"mlp", "mlpsde", "mlp_sde"}:
        from nsc.nsde import MLPSDE
        return MLPSDE(**kwargs)
    elif model_type in {"hopf", "hopf_coupled", "hopf_coupled_sde"}:
        from .hopf import HopfCoupledSDE
        return HopfCoupledSDE(**kwargs)
    elif model_type in {"latent", "latent_sde"}:
        from nsc.nsde import LatentSDE
        return LatentSDE(**kwargs)
    raise ValueError(f"Unknown model name: {model_type}")


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


def phase_int_matrix(ts: np.ndarray) -> np.ndarray:
    """Compute the Phase-Interaction Matrix for an input BOLD signal."""
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError(f"Expected a 2D array (N, T); got shape {ts.shape}")
    if np.isnan(ts).any():
        warnings.warn("############ Warning in phase_int_matrix: NAN found ############")
        return np.array([np.nan])

    phases = np.angle(signal.hilbert(ts, axis=1))
    phase_diff = phases[:, None, :] - phases[None, :, :]  # (N, N, T)
    phase_diff = np.abs(phase_diff)
    phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)
    ph_int = np.cos(phase_diff)
    return np.moveaxis(ph_int, 2, 0)  # (T, N, N)


def phFCD(ts: np.ndarray, windowsize_phase: int = 3) -> np.ndarray:
    """Compute the Phase Functional Connectivity Dynamics (phFCD) of an input BOLD signal."""
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError(f"Expected a 2D array (N, T); got shape {ts.shape}")
    (N, Tmax) = ts.shape
    if Tmax < windowsize_phase:
        warnings.warn("############ Warning in phFCD: insufficient timepoints ############")
        return np.array([np.nan])

    npattmax = Tmax
    size_kk3 = int((npattmax - windowsize_phase) * (npattmax - (windowsize_phase - 1)) / 2)

    phIntMatr = phase_int_matrix(ts)
    if phIntMatr.ndim < 3 or np.isnan(phIntMatr).any():
        warnings.warn("############ Warning in phFCD: NAN found ############")
        return np.array([np.nan])

    triu_indices = np.triu_indices(phIntMatr.shape[1], k=1)
    phIntMatr_upTri = np.zeros((npattmax, int(N * (N - 1) / 2)))
    for t in range(npattmax):
        phIntMatr_upTri[t, :] = phIntMatr[t][triu_indices]
    phfcd = np.zeros((size_kk3))
    kk3 = 0
    for t in range(npattmax - (windowsize_phase - 1)):
        p1 = np.mean(phIntMatr_upTri[t:t + windowsize_phase, :], axis=0)
        for t2 in range(t + 1, npattmax - (windowsize_phase - 1)):
            p2 = np.mean(phIntMatr_upTri[t2:t2 + windowsize_phase, :], axis=0)
            phfcd[kk3] = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)) if np.linalg.norm(p1) * np.linalg.norm(p2) != 0 else np.nan
            kk3 = kk3 + 1
    return phfcd


def ts_kolmogorov_phfcd(ts1: np.ndarray, ts2: np.ndarray, **fcd_kwargs) -> float:
    """Kolmogorov distance between the phFCD of two timeseries."""
    phFCD1 = phFCD(ts1, **fcd_kwargs)
    phFCD2 = phFCD(ts2, **fcd_kwargs)

    mask1 = ~np.isnan(phFCD1)
    mask2 = ~np.isnan(phFCD2)
    if not mask1.any() or not mask2.any():
        return np.nan

    return stats.ks_2samp(phFCD1[mask1], phFCD2[mask2])[0]


def compute_metrics(y_true: np.ndarray, y_sim: np.ndarray) -> dict:
    """Compute MSE, KL (approx via NLL), and phFCD metrics for simulated trajectories."""
    T = y_true.shape[1]
    intervals = {
        "short": slice(0, T // 3),
        "med": slice(T // 3, 2 * T // 3),
        "long": slice(2 * T // 3, T),
        "full": slice(0, T),
    }

    metrics = {}
    mu_sim = y_sim.mean(axis=0)
    std_sim = y_sim.std(axis=0) + 1e-6

    for name, sl in intervals.items():
        mse = np.mean((mu_sim[sl] - y_true[0, sl]) ** 2)
        metrics[f"mse_{name}"] = mse

        nll = 0.5 * np.log(2 * np.pi) + np.log(std_sim[sl]) + 0.5 * ((y_true[0, sl] - mu_sim[sl]) / std_sim[sl]) ** 2
        metrics[f"kl_{name}"] = np.mean(nll)

    bold_dim = y_true.shape[2] // 2 if y_true.shape[2] >= 2 else y_true.shape[2]
    true_ts = y_true[0, :, :bold_dim].T
    phfcd_distances = []
    for sample in y_sim:
        sim_ts = sample[:, :bold_dim].T
        dist = ts_kolmogorov_phfcd(true_ts, sim_ts)
        if not np.isnan(dist):
            phfcd_distances.append(dist)
    if phfcd_distances:
        metrics["phfcd_ks_mean"] = float(np.mean(phfcd_distances))
        metrics["phfcd_ks_std"] = float(np.std(phfcd_distances))

    return metrics


def find_checkpoint(path: Path) -> Path:
    """Pick a checkpoint file, preferring an explicit path or the latest best_model*.pt."""
    if path.is_file():
        return path
    pts = sorted(path.glob("best_model*.pt"))
    if not pts:
        raise SystemExit(f"No checkpoint files found in: {path}")
    return pts[-1]
