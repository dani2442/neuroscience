from __future__ import annotations

import dataclasses
import typing as t

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchsde
import wandb
import os
import time
import torch.optim.lr_scheduler as lr_scheduler

from .utils import set_seed


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2      # [B, T, F]
    diff = diff.mean(dim=-1)         # [B, T]

    # masked sum per sequence
    masked_sum = (diff * mask).sum(dim=-1)        # [B]
    counts = mask.sum(dim=-1).clamp_min(1.0)      # [B]

    per_batch_mean = masked_sum / counts          # [B]
    return per_batch_mean.mean()                  # scalar


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    method: str = "euler"
    dt: float = 1
    dt_num: float = 0.1
    state_size: int = 200
    length: int = 10
    reg_lambda: float = 1e-4
    batch_size: int = 32
    n_repeat: int = 50
    brownian_size: int = 10
    hidden_size: int = 64
    device: t.Union[str, torch.device] = "cpu"
    grad_clip: float = 1.0
    wandb_project: t.Optional[str] = None
    wandb_run_name: t.Optional[str] = None
    data_dir: str = "data_processed/ts_young/"
    # Where to write checkpoints and other outputs
    output_dir: str = "runs"
    # Whether to save the best model (based on validation loss). If False, no checkpoint is saved.
    save_best: bool = True
    scheduler_min_lr: float = 5e-6


class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: t.Optional[DataLoader], cfg: TrainerConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.device
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # Run name is the date and time so we use a timestamp %Y%m%d-%H%M%S
        run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        # initialize wandb only if a project is provided; otherwise keep it None
        if cfg.wandb_project:
            self.wandb = wandb.init(project=cfg.wandb_project, name=run_name, config=dataclasses.asdict(cfg))
            wandb.Settings(
                http_proxy="http://proxy.nhr.fau.de:80",
                https_proxy="http://proxy.nhr.fau.de:80"
            )
        else:
            self.wandb = None

        # checkpoint bookkeeping
        self.best_val = float('inf')
        self.ckpt_path = None
        try:
            import os
            os.makedirs(cfg.output_dir, exist_ok=True)
        except Exception:
            pass
        # setup learning rate scheduler if requested
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.scheduler_min_lr)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss, total_reg_loss = 0.0, 0.0
        batches = 0
        for ts_pad, y_pad, mask, x0 in self.train_loader:
            ts_pad = ts_pad.to(self.device)
            y_pad = y_pad.to(self.device)
            mask = mask.to(self.device)
            x0 = x0.to(self.device)

            self.optimizer.zero_grad()
            x_sim = torchsde.sdeint(self.model, x0, ts_pad, method=self.cfg.method, dt=self.cfg.dt_num)
            x_sim = x_sim.transpose(0, 1)

            loss = masked_mse(x_sim, y_pad, mask)

            mse_loss = masked_mse(x_sim, y_pad, mask)
            reg_loss = 0.0
            for param in self.model.parameters():
                reg_loss += torch.norm(param, p=2) ** 2
            reg_loss *= self.cfg.reg_lambda

            loss = mse_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
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
            for ts_pad, y_pad, mask, x0 in self.val_loader:
                ts_pad = ts_pad.to(self.device)
                y_pad = y_pad.to(self.device)
                mask = mask.to(self.device)
                x0 = x0.to(self.device)

                x_sim = torchsde.sdeint(self.model, x0, ts_pad, method=self.cfg.method, dt=self.cfg.dt_num)
                x_sim = x_sim.transpose(0, 1)
                loss = masked_mse(x_sim, y_pad, mask)
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

            line = f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f} | time: {epoch_time:.2f}s"
            print(line)

            # log more diagnostics to wandb if enabled
            if self.wandb is not None:
                # learning rate may be in optimizer param groups
                lr = None
                try:
                    lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
                except Exception:
                    lr = None

                self.wandb.log({
                    "train/loss": train_loss,
                    "train/reg_loss": train_reg_loss,
                    "val/loss": val_loss,
                    "train/epoch_time": epoch_time,
                    "train/lr": lr,
                    "epoch": epoch,
                })

            # save best model checkpoint
            if self.cfg.save_best and self.val_loader is not None:
                improved = val_loss < self.best_val
                if improved:
                    self.best_val = val_loss

                    ckpt_name = f"best_epoch_{epoch:03d}_val_{val_loss:.6f}.pt"
                    ckpt_path = os.path.join(self.cfg.output_dir, ckpt_name)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'cfg': dataclasses.asdict(self.cfg),
                    }, ckpt_path)
                    self.ckpt_path = ckpt_path
                    #print(f"Saved improved checkpoint: {ckpt_path}")
                    if self.wandb is not None:
                        # upload checkpoint as artifact
                        artifact = wandb.Artifact('best-model', type='model')
                        artifact.add_file(ckpt_path)
                        self.wandb.log_artifact(artifact)

            # step learning rate scheduler if configured
            self.scheduler.step()
