import torch.nn as nn
import torch

class LinearSDE(nn.Module):
    def __init__(self, state_size: int, brownian_size: int):
        super().__init__()

        self.state_size = state_size
        self.A = nn.Parameter(1e-3*torch.randn((state_size,state_size)), requires_grad=True) # Temporal correlation and diffusion
        self.C = nn.Parameter(torch.randn((state_size, brownian_size)), requires_grad=True)  # Correlation matrix CC^T=\Sigma.
        self.noise_type = "additive"
        self.sde_type = "ito"
    
    def f(self, t, y):
        return y @ (self.A.T - torch.eye(self.state_size))
    
    def g(self, t, y):
        return self.C.repeat(y.size(0),1,1)
    

class MLPSDE(nn.Module):
    def __init__(self, state_size: int, brownian_size: int, hidden_size: int):
        super().__init__()

        self.drift = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size),
        )

        self.C = nn.Parameter(torch.randn((state_size, brownian_size)), requires_grad=True)  # Correlation matrix CC^T=\Sigma.
        self.noise_type = "additive"
        self.sde_type = "ito"
    
    def f(self, t, y):
        return self.drift(y)
    
    def g(self, t, y):
        return self.C.repeat(y.size(0),1,1)