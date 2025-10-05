import torch
import torch.nn as nn
import torchsde
from typing import Optional

class MLPSDE(nn.Module):
    def __init__(self, state_size: int, control_size: int, brownian_size: int, hidden_size: int, hidden_layers: int, control_input=None, noise_type: Optional[str] = None, *args, **kwargs):
        super().__init__()

        self.noise_type = noise_type
        self.sde_type = "ito"

        self.state_size = state_size
        self.control_size = control_size
        self.augmented_size = self.state_size + self.control_size

        self.control_input = control_input
        
        layers = [nn.Linear(self.augmented_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, self.state_size))

        self.drift = nn.Sequential(*layers)

        if noise_type=="additive":
            self.C = nn.Parameter(torch.randn((self.state_size, brownian_size)))
        elif noise_type=="diagonal":
            self.C = nn.Parameter(torch.randn(self.state_size))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # apply drift pointwise over batch
        if self.control_input is not None:
            u = self.control_input(t, y)
        
        return  self.drift(torch.cat([y, u], dim=1))


    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "additive":
            return self.C.unsqueeze(0).expand(y.size(0), -1, -1)
        elif self.noise_type == "diagonal":
            return self.C.repeat(y.size(0), 1)
        else:
            return torch.zeros_like(y)


class SDEWrapper(nn.Module):
    """Wrapper for torchsde integration with MLPSDE"""
    def __init__(self, sde_model, method='euler', dt=0.01):
        super().__init__()
        self.sde = sde_model
        self.method = method
        self.dt = dt
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        
    def forward(self, x0, ts, u: Optional[torch.tensor] = None):
        """
        Args:
            x0: Initial state [batch_size, features]
            ts: Time points [num_timesteps]
        Returns:
            Trajectory [num_timesteps, batch_size, features]
        """
        if u is not None:
            self.sde.control_input.set_control(u, ts[0])

        # sdeint expects [batch, features] for x0
        ys = torchsde.sdeint(
            self.sde,
            x0,
            ts[0],
            method=self.method,
            dt=self.dt,
        ).transpose(1,0)
        return ys