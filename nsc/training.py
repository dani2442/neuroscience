from __future__ import annotations

import dataclasses
import typing as t

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchsde
import wandb
import os
import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributions as D

from .utils import set_seed
from .latent import LatentSDE


def masked_mse(pred, target, mask, loss_fn: str = "mse") -> torch.Tensor:
    if loss_fn == "mse":
        diff = (pred - target) ** 2      # [B, T, F]
    elif loss_fn == "mae":
        diff = torch.abs(pred - target)  # [B, T, F]
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    
    diff = diff.mean(dim=-1)         # [B, T]
    diff = (diff * mask).mean(dim=-1)        # [B]
    return diff.mean()        # [B]


class LinearScheduler:
    def __init__(self, iters: int, maxval: float = 1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class EMAMetric:
    def __init__(self, gamma: float = 0.99):
        self._val = 0.0
        self._gamma = gamma

    def step(self, x: t.Union[torch.Tensor, float]):
        val = x.detach().cpu().item() if isinstance(x, torch.Tensor) else float(x)
        self._val = self._gamma * self._val + (1 - self._gamma) * val
        return self._val

    @property
    def val(self):
        return self._val


@dataclasses.dataclass
class TrainerConfig:
    model_type: str = "mlp"
    epochs: int = 50
    lr: float = 1e-3
    lr_end: float = 1e-4
    method: str = "euler"
    dt: float = 1
    model: str = "mlp"
    num_patients: int = 100
    dt_num: float = 0.1
    split: float = 0.8
    state_size: int = 200
    length_train: int = 40
    length_val: int = 40
    reg_lambda: float = 1e-4
    batch_size: int = 32
    noise_type: str = 'additive'
    n_repeat: int = 50
    load_ckpt: t.Optional[str] = None
    brownian_size: int = 10
    hidden_size: int = 64
    loss: str = "mse"
    hidden_layers: int = 3
    seed: t.Optional[int] = None
    device: t.Union[str, torch.device] = "cpu"
    grad_clip: float = 10.0
    activation: str = "tanh"
    wandb_project: t.Optional[str] = None
    wandb_run_name: t.Optional[str] = None
    data_dir: str = "data_processed/ts_young/"
    output_dir: str = "runs"
    save_best: bool = True
    adjoint: bool = False
    sde_type: str = "ito"
    likelihood: str = "laplace"
    scale: float = 0.05
    kl_anneal_iters: int = 1000
    latent_hidden: int = 200
    latent_layers: int = 2
    prior_theta: float = 1.0
    prior_mu: float = 0.0
    prior_sigma: float = 0.5
    adaptive: bool = False
    rtol: float = 1e-3
    atol: float = 1e-3


class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: t.Optional[DataLoader], cfg: TrainerConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.device
        self.model.to(self.device)
        self.sdeint_fn = torchsde.sdeint_adjoint if cfg.adjoint else torchsde.sdeint
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # Run name is the date and time so we use a timestamp %Y%m%d-%H%M%S
        self.run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        # initialize wandb only if a project is provided; otherwise keep it None
        if cfg.wandb_project:
            self.wandb = wandb.init(project=cfg.wandb_project, name=self.run_name, config=dataclasses.asdict(cfg))
            wandb.Settings(
                http_proxy="http://proxy.nhr.fau.de:80",
                https_proxy="http://proxy.nhr.fau.de:80"
            )
        else:
            self.wandb = None

        self.loss_fn = nn.MSELoss() if cfg.loss == "mse" else nn.L1Loss()

        # checkpoint bookkeeping
        self.best_val = float('inf')
        self.ckpt_path = None
        os.makedirs(cfg.output_dir, exist_ok=True)

        # setup learning rate scheduler if requested
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr_end)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss, total_reg_loss = 0.0, 0.0
        batches = 0
        for ts_pad, y_pad, x0 in self.train_loader:
            ts_pad = ts_pad.to(self.device)
            y_pad = y_pad.to(self.device)
            x0 = x0.to(self.device)

            self.optimizer.zero_grad()
            x_sim = self.sdeint_fn(self.model, x0, ts_pad, method=self.cfg.method, dt=self.cfg.dt_num)
            x_sim = x_sim.transpose(0, 1)

            mse_loss = self.loss_fn(x_sim, y_pad)
            reg_loss = 0.0
            for param in self.model.parameters():
                reg_loss += torch.norm(param, p=2) ** 2
            reg_loss *= self.cfg.reg_lambda

            loss = mse_loss + reg_loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf loss encountered, skipping batch")
                continue

            loss.backward()
            if self.cfg.grad_clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.item())
            total_reg_loss += float(reg_loss.item())
            batches += 1

        avg, avg_reg = total_loss / max(1, batches), total_reg_loss / max(1, batches)
        return avg, avg_reg

    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for ts_pad, y_pad, x0 in self.val_loader:
                ts_pad = ts_pad.to(self.device)
                y_pad = y_pad.to(self.device)
                x0 = x0.to(self.device)

                x_sim = self.sdeint_fn(self.model, x0, ts_pad, method=self.cfg.method, dt=self.cfg.dt_num)
                x_sim = x_sim.transpose(0, 1)
                loss = self.loss_fn(x_sim, y_pad)
                total_loss += float(loss.item())
                batches += 1

        return total_loss / max(1, batches)

    
    def fit(self, seed: t.Optional[int] = None):
        if seed is not None:
            set_seed(seed)

        for epoch in range(1, self.cfg.epochs + 1):
            start_time = time.time()
            train_loss, train_reg_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f} | time: {epoch_time:.2f}s")

            # Log to wandb if enabled
            if self.wandb is not None:
                lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
                self.wandb.log({
                    "train/loss": train_loss,
                    "train/reg_loss": train_reg_loss,
                    "val/loss": val_loss,
                    "train/epoch_time": epoch_time,
                    "train/lr": lr,
                    "epoch": epoch,
                })

            self._save_best_model(epoch, val_loss)
            self.scheduler.step()

    def _save_best_model(self, epoch: int, val_loss: float):
        """Save model if it's the best validation loss so far."""
        if not (self.cfg.save_best and self.val_loader is not None):
            return
            
        if val_loss < self.best_val:
            self.best_val = val_loss
            
            # Remove previous best model file
            if hasattr(self, 'best_model_path') and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            # Save new best model
            timestamp = self.run_name.replace("run_", "")
            self.best_model_path = os.path.join(self.cfg.output_dir, f'{self.cfg.model}_{val_loss:.6f}_{timestamp}.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'cfg': dataclasses.asdict(self.cfg),
            }, self.best_model_path)
            
            # Log to wandb if enabled
            if self.wandb is not None:
                artifact = wandb.Artifact(f'best-model_{self.run_name}', type='model')
                artifact.add_file(self.best_model_path)
                self.wandb.log_artifact(artifact)


class LatentTrainer:
    """Trainer for the latent SDE model (variational path KL + masked likelihood)."""

    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: t.Optional[DataLoader], cfg: TrainerConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.device
        self.model.to(self.device)
        self.sdeint_fn = torchsde.sdeint_adjoint if cfg.adjoint else torchsde.sdeint
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr_end)
        self.kl_scheduler = LinearScheduler(cfg.kl_anneal_iters)
        self.likelihood_constructor = {"laplace": D.Laplace, "normal": D.Normal}[cfg.likelihood]

        self.run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        if cfg.wandb_project:
            self.wandb = wandb.init(project=cfg.wandb_project, name=self.run_name, config=dataclasses.asdict(cfg))
            wandb.Settings(
                http_proxy="http://proxy.nhr.fau.de:80",
                https_proxy="http://proxy.nhr.fau.de:80"
            )
        else:
            self.wandb = None

        self.logpy_metric = EMAMetric()
        self.kl_metric = EMAMetric()
        self.loss_metric = EMAMetric()

        self.best_val = float('inf')
        os.makedirs(cfg.output_dir, exist_ok=True)

<<<<<<< HEAD
<<<<<<< HEAD
    def _likelihood(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        likelihood = self.likelihood_constructor(loc=pred, scale=self.cfg.scale)
        log_prob = likelihood.log_prob(target).sum(dim=-1)  # (B, T)
        log_prob = log_prob.mean(dim=-1)  # (B,)
=======
=======
>>>>>>> 4fbdc7c (new)
    def _likelihood(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        likelihood = self.likelihood_constructor(loc=pred, scale=self.cfg.scale)
        log_prob = likelihood.log_prob(target).sum(dim=-1)  # (B, T)
        log_prob = (log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
<<<<<<< HEAD
>>>>>>> 4fbdc7c (new)
=======
>>>>>>> 4fbdc7c (new)
        return log_prob.mean()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
<<<<<<< HEAD
<<<<<<< HEAD
        for ts_pad, y_pad, _ in self.train_loader:
            ts_pad = ts_pad.to(self.device)
            y_pad = y_pad.to(self.device)
=======
=======
>>>>>>> 4fbdc7c (new)
        for ts_pad, y_pad, mask in self.train_loader:
            ts_pad = ts_pad.to(self.device)
            y_pad = y_pad.to(self.device)
            mask = mask.to(self.device)
<<<<<<< HEAD
>>>>>>> 4fbdc7c (new)
=======
>>>>>>> 4fbdc7c (new)
            batch_size = y_pad.size(0)

            self.optimizer.zero_grad()
            ys, kl = self.model(
                ts=ts_pad,
                batch_size=batch_size,
                sdeint_fn=self.sdeint_fn,
                method=self.cfg.method,
                dt=self.cfg.dt_num,
                adaptive=self.cfg.adaptive,
                rtol=self.cfg.rtol,
                atol=self.cfg.atol,
            )
            ys = ys.transpose(0, 1)  # (B, T, state)

<<<<<<< HEAD
<<<<<<< HEAD
            logpy = self._likelihood(ys, y_pad)
=======
            logpy = self._likelihood(ys, y_pad, mask)
>>>>>>> 4fbdc7c (new)
=======
            logpy = self._likelihood(ys, y_pad, mask)
>>>>>>> 4fbdc7c (new)
            loss = -logpy + kl * self.kl_scheduler.val
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.kl_scheduler.step()

            self.logpy_metric.step(logpy)
            self.kl_metric.step(kl)
            self.loss_metric.step(loss)
            total_loss += float(loss.item())
            batches += 1

        avg = total_loss / max(1, batches)
        return avg

    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
<<<<<<< HEAD
<<<<<<< HEAD
            for ts_pad, y_pad, _ in self.val_loader:
                ts_pad = ts_pad.to(self.device)
                y_pad = y_pad.to(self.device)
=======
=======
>>>>>>> 4fbdc7c (new)
            for ts_pad, y_pad, mask in self.val_loader:
                ts_pad = ts_pad.to(self.device)
                y_pad = y_pad.to(self.device)
                mask = mask.to(self.device)
<<<<<<< HEAD
>>>>>>> 4fbdc7c (new)
=======
>>>>>>> 4fbdc7c (new)
                batch_size = y_pad.size(0)

                ys, kl = self.model(
                    ts=ts_pad,
                    batch_size=batch_size,
                    sdeint_fn=self.sdeint_fn,
                    method=self.cfg.method,
                    dt=self.cfg.dt_num,
                    adaptive=self.cfg.adaptive,
                    rtol=self.cfg.rtol,
                    atol=self.cfg.atol,
                )
                ys = ys.transpose(0, 1)
<<<<<<< HEAD
<<<<<<< HEAD
                logpy = self._likelihood(ys, y_pad)
=======
                logpy = self._likelihood(ys, y_pad, mask)
>>>>>>> 4fbdc7c (new)
=======
                logpy = self._likelihood(ys, y_pad, mask)
>>>>>>> 4fbdc7c (new)
                loss = -logpy + kl

                total_loss += float(loss.item())
                batches += 1

        return total_loss / max(1, batches)

    def fit(self, seed: t.Optional[int] = None):
        if seed is not None:
            set_seed(seed)

        for epoch in range(1, self.cfg.epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | "
                f"train: {train_loss:.6f} | "
                f"val: {val_loss:.6f} | "
                f"loss(avg): {self.loss_metric.val:.6f} | "
                f"kl(avg): {self.kl_metric.val:.6f} | "
                f"logpy(avg): {self.logpy_metric.val:.6f} | "
                f"time: {epoch_time:.2f}s"
            )

            if self.wandb is not None:
                lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
                self.wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "metrics/loss_ema": self.loss_metric.val,
                    "metrics/kl_ema": self.kl_metric.val,
                    "metrics/logpy_ema": self.logpy_metric.val,
                    "train/epoch_time": epoch_time,
                    "train/lr": lr,
                    "epoch": epoch,
                })

            self._save_best_model(epoch, val_loss)
            self.scheduler.step()

    def _save_best_model(self, epoch: int, val_loss: float):
        if not (self.cfg.save_best and self.val_loader is not None):
            return
        if val_loss < getattr(self, "best_val", float("inf")):
            self.best_val = val_loss
            ckpt_path = os.path.join(self.cfg.output_dir, f'best_model_{self.run_name}.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'cfg': dataclasses.asdict(self.cfg),
            }, ckpt_path)
            self.best_model_path = ckpt_path

            if self.wandb is not None:
                artifact = wandb.Artifact(f'best-model_{self.run_name}', type='model')
                artifact.add_file(ckpt_path)
                self.wandb.log_artifact(artifact)
