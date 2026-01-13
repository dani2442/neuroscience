from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import distributions, nn
import torchsde


def _stable_division(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Elementwise stable division that avoids exploding gradients when b≈0."""
    b_safe = torch.where(
        b.abs().detach() > epsilon,
        b,
        torch.full_like(b, fill_value=epsilon) * b.sign()
    )
    return a / b_safe


class LatentSDE(torchsde.SDEIto):
    """Latent SDE with learnable approximate posterior drift and prior Ornstein–Uhlenbeck drift.

    The implementation follows the reference example in `examples/latent_sde.py`, but it is
    parameterized to be reusable in the training pipeline (no reliance on global argparse args).
    """

    def __init__(
        self,
        state_size: int = 1,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        hidden_size: int = 200,
        hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2.0 * theta))
        self.state_size = int(state_size)
        self.num_osc = state_size // 2

        # Prior drift parameters.
        self.register_buffer("theta", torch.full((1, self.state_size), float(theta)))
        self.register_buffer("mu", torch.full((1, self.state_size), float(mu)))
        self.register_buffer("sigma", torch.full((1, self.state_size), float(sigma)))

        # p(y0).
        self.register_buffer("py0_mean", torch.full((1, self.state_size), float(mu)))
        self.register_buffer("py0_logvar", torch.full((1, self.state_size), float(logvar)))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        layers = [nn.Linear(self.state_size + 2, hidden_size), nn.Tanh()]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, self.state_size))
        self.net = nn.Sequential(*layers)
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)

        self.a = nn.Parameter(torch.rand(self.num_osc), requires_grad=True)       # growth rate a_n
        self.omega = nn.Parameter(torch.rand(self.num_osc), requires_grad=True)   # frequency ω_n
        self.C = nn.Parameter(0.05+0.05*torch.rand((self.num_osc, self.num_osc)))

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.full((1, self.state_size), float(mu)), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.full((1, self.state_size), float(logvar)), requires_grad=True)

    def _expand_time(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = torch.full_like(y[:, :1], fill_value=t)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        return t

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Approximate posterior drift."""
        t = self._expand_time(t, y)
        feats = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        return self.net(feats)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Shared diffusion."""
        return self.sigma.expand(y.size(0), -1)

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Prior drift (Ornstein–Uhlenbeck)."""
        y_x = y[:, :self.num_osc]
        y_y = y[:, self.num_osc:]

        r2 = y_x**2 + y_y**2  # (batch, num_osc)

        C_sum = self.C.sum(dim=0)  # (num_osc,)

        coupling_x = y_x @ self.C - y_x * C_sum
        coupling_y = y_y @ self.C - y_y * C_sum

        dxdt = (self.a - r2) * y_x - self.omega * y_y + coupling_x / self.num_osc
        dydt = (self.a - r2) * y_y + self.omega * y_x + coupling_y / self.num_osc

        # Stack back into (batch, state_size)
        dydt_full = torch.cat([dxdt, dydt], dim=-1)
        return dydt_full

    def f_aug(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift for augmented dynamics with logqp term."""
        y_state = y[:, 0 : self.state_size]
        f_val, g_val, h_val = self.f(t, y_state), self.g(t, y_state), self.h(t, y_state)
        u = _stable_division(f_val - h_val, g_val)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f_val, f_logqp], dim=1)

    def g_aug(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion for augmented dynamics with logqp term."""
        y_state = y[:, 0 : self.state_size]
        g_val = self.g(t, y_state)
        g_logqp = torch.zeros_like(y_state[:, :1])
        return torch.cat([g_val, g_logqp], dim=1)

    def forward(
        self,
        ts: torch.Tensor,
        batch_size: int,
        sdeint_fn=torchsde.sdeint,
        method: str = "euler",
        dt: float = 1e-2,
        adaptive: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        eps: Optional[torch.Tensor] = None,
    ):
        eps = torch.randn(batch_size, self.state_size, device=self.qy0_std.device) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1, device=y0.device, dtype=y0.dtype)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        ys, logqp_path = aug_ys[:, :, 0 : self.state_size], aug_ys[-1, :, self.state_size]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(
        self,
        ts: torch.Tensor,
        batch_size: int,
        sdeint_fn=torchsde.sdeint,
        method: str = "srk",
        dt: float = 1e-2,
        eps: Optional[torch.Tensor] = None,
        bm=None,
    ):
        """Sample from the prior."""
        eps = torch.randn(batch_size, self.state_size, device=self.py0_mean.device) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return sdeint_fn(self, y0, ts, bm=bm, method=method, dt=dt, names={"drift": "h"})

    def sample_q(
        self,
        ts: torch.Tensor,
        batch_size: int,
        sdeint_fn=torchsde.sdeint,
        method: str = "srk",
        dt: float = 1e-2,
        eps: Optional[torch.Tensor] = None,
        bm=None,
    ):
        """Sample from the approximate posterior."""
        eps = torch.randn(batch_size, self.state_size, device=self.qy0_mean.device) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return sdeint_fn(self, y0, ts, bm=bm, method=method, dt=dt)

    @property
    def py0_std(self):
        return torch.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(0.5 * self.qy0_logvar)
