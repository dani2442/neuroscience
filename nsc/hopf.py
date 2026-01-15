import torch
import torch.nn as nn

class HopfCoupledSDE(nn.Module):
    def __init__(self, state_size: int, brownian_size: int, noise_type: str, sde_type: str, *args, **kwargs):
        super().__init__()

        assert state_size % 2 == 0, "state_size must be even (x_n, y_n pairs)"
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.num_osc = state_size // 2
        
        # Parameters for each oscillator
        self.a = nn.Parameter(1+0.*torch.rand(self.num_osc), requires_grad=True)       # growth rate a_n
        self.omega = nn.Parameter(1+0.*torch.rand(self.num_osc), requires_grad=True)   # frequency ω_n
        self.C = nn.Parameter(0.0*torch.rand((self.num_osc, self.num_osc)))

        if noise_type=="additive":
            self.B = nn.Parameter(0.0*torch.randn((state_size, brownian_size)))
        elif noise_type=="diagonal":
            self.B = nn.Parameter(0.0*torch.randn(state_size))
    
    def f(self, t, y):
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
    
    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "additive":
            return self.B.unsqueeze(0).expand(y.size(0), -1, -1)
        elif self.noise_type == "diagonal":
            return self.B.unsqueeze(0).expand(y.size(0), -1)