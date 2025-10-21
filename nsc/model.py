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
    def __init__(self, state_size: int, brownian_size: int, activation: str, hidden_size: int, noise_type: str, sde_type: str, *args, **kwargs):
        super().__init__()

        self.noise_type = noise_type
        self.sde_type = sde_type

        self.state_size = int(state_size)

        class LipSwish(nn.Module):
            def forward(self, x):
                return 0.909 * torch.nn.functional.silu(x)

        if activation == 'silu':
            activation_fn = LipSwish()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        
        layers = [nn.Linear(state_size, hidden_size), activation_fn]
        for _ in range(kwargs['hidden_layers']):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation_fn])
        layers.append(nn.Linear(hidden_size, state_size))

        self.drift = nn.Sequential(*layers)

        if noise_type=="additive":
            self.C = nn.Parameter(torch.randn((state_size, brownian_size)))
        elif noise_type=="diagonal":
            self.C = nn.Parameter(torch.randn(state_size))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # apply drift pointwise over batch
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "additive":
            return self.C.unsqueeze(0).expand(y.size(0), -1, -1)
        elif self.noise_type == "diagonal":
            return self.C.unsqueeze(0).expand(y.size(0), -1)



def make_model(name: str, **kwargs) -> t.Callable:
    name = name.lower()
    if name == "linear":
        return LinearSDE(**kwargs)
    elif name in {"mlp", "mlpsde", "mlp_sde"}:
        return MLPSDE(**kwargs)
    raise ValueError(f"Unknown model name: {name}")