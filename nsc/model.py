from __future__ import annotations

import typing as t

import torch
import torch.nn as nn


class LinearSDE(nn.Module):
    def __init__(self, state_size: int, brownian_size: int, *args, **kwargs):
        super().__init__()
        self.noise_type = "additive"
        self.sde_type = "ito"

        self.state_size = int(state_size)
        self.A = nn.Parameter(1e-3 * torch.randn((state_size, state_size)))
        self.C = nn.Parameter(torch.randn((state_size, brownian_size)))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # simple linear drift: (A - I) y
        return y @ (self.A.T - torch.eye(self.state_size, device=y.device, dtype=y.dtype))

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # broadcast C to (B, state_size, brownian_size)
        return self.C.unsqueeze(0).expand(y.size(0), -1, -1)


class MLPSDE(nn.Module):
    def __init__(self, state_size: int, brownian_size: int, hidden_size: int, *args, **kwargs):
        super().__init__()

        self.noise_type = "additive"
        self.sde_type = "ito"

        self.state_size = int(state_size)
        self.drift = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size),
        )
        self.C = nn.Parameter(torch.randn((state_size, brownian_size)))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # apply drift pointwise over batch
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.C.unsqueeze(0).expand(y.size(0), -1, -1)


def make_model(name: str, **kwargs) -> t.Callable:
    name = name.lower()
    if name == "linear":
        return LinearSDE(**kwargs)
    if name in {"mlp", "mlpsde", "mlp_sde"}:
        return MLPSDE(**kwargs)
    raise ValueError(f"Unknown model name: {name}")