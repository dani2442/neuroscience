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
        

def rbf_kernel(x1, x2, sigma, dim=1): # (B, F) or (B)
    if dim is None:
        norm = torch.square(x1 - x2)  # scalar
    else:
        norm = torch.square(x1-x2).sum(dim) #.sum(-1) # (B, B)
    return torch.exp(-1/(2*sigma**2)*norm)


def rbf_kernel_fast(x1: torch.Tensor, x2: torch.Tensor, sigma: float):
    """
    Efficient RBF kernel using torch.cdist.
    x1: (..., N, D)
    x2: (..., M, D)
    Returns: (..., N, M)
    """
    dist2 = torch.cdist(x1, x2, p=2).pow(2)
    return torch.exp(-dist2 / (2 * sigma ** 2))


class KernelSDE(nn.Module):
    def __init__(self, state_size: int, control_size: int, brownian_size: int, hidden_size: int, hidden_layers: int, control_input=None, noise_type: Optional[str] = None, *args, **kwargs):
        super().__init__()

        self.noise_type = noise_type
        self.sde_type = "ito"

        self.state_size = state_size
        self.control_size = control_size
        self.augmented_size = self.state_size + self.control_size

        self.num_points = 100
        
        self.u = -5+10*torch.rand(self.num_points, self.control_size)
        self.y = -5+10*torch.rand(self.num_points, self.state_size)

        self.alpha = nn.Parameter(torch.rand(self.num_points))
        self.vecs = nn.Parameter(torch.randn(self.num_points, self.state_size))

        self.control_input = control_input
        
        if noise_type=="additive":
            self.C = nn.Parameter(torch.randn((self.state_size, brownian_size)))
        elif noise_type=="diagonal":
            self.C = nn.Parameter(torch.randn(self.state_size))

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # apply drift pointwise over batch
        if self.control_input is not None:
            u = self.control_input(t, y)
        
        # K1 = rbf_kernel(self.y[None, :, :, None], y[:, None, None, :], sigma=4.0, dim=None) # [B, N, S, S]
        # K2 = rbf_kernel(self.u[None, :, :], u[:, None, :], sigma=4.0, dim=-1) # [B, N]
        # r = torch.einsum("bnss,ns->bns", K1, self.vecs)  # [B, N, S]
        # return torch.einsum("n,bns,bn->bs", self.alpha, r, K2)  # [B, S]
        
        K1 = rbf_kernel(self.y[None, :, :], y[:, None, :], sigma=2.0, dim=-1) # [B, N]
        K2 = rbf_kernel(self.u[None, :, :], u[:, None, :], sigma=2.0, dim=-1) # [B, N]
        r = torch.einsum("bn,ns->bns", K1*K2, self.vecs)  # [B, N, S]
        return torch.einsum("n,bns,bn->bs", self.alpha, r, K2)  # [B, S]

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