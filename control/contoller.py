import torch
import torch.nn as nn

def compute_gramian(A, B, T, n_steps=200):
    """
    Compute controllability Gramian numerically with trapezoidal integration.
    Wc(T) = ∫_0^T e^{At} B B^T e^{A^T t} dt
    
    Vectorized implementation without explicit loops.
    """
    times = torch.linspace(0, T, n_steps + 1, dtype=A.dtype, device=A.device)
    
    # Compute all matrix exponentials at once using vmap
    # Shape: (n_steps+1, n, n)
    At = A.unsqueeze(0) * times.view(-1, 1, 1)
    Et = torch.vmap(torch.matrix_exp)(At)
    
    # Compute Mt = Et @ B @ B^T @ Et^T for all time steps
    # Shape: (n_steps+1, n, n)
    BBT = B @ B.T
    Mt = Et @ BBT @ Et.transpose(-2, -1)
    
    # Apply trapezoidal weights: 0.5 for first and last, 1.0 for middle
    weights = torch.ones(n_steps + 1, dtype=A.dtype, device=A.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    
    # Weighted sum with broadcasting
    Wc = torch.sum(Mt * weights.view(-1, 1, 1), dim=0)
    Wc *= (T / n_steps)
    
    return Wc


class OptimalRoute(nn.Module):
    def __init__(self, A, B, x_target, T, n_steps=200, regularize=1e-9):
        super().__init__()
        self.x_target = x_target.reshape(1, -1)
        self.A = A
        self.B = B
        self.T = T
        self.m_exp = torch.matrix_exp(A * T)

        # Gramian
        Wc = compute_gramian(A, B, T, n_steps=n_steps)
        Wc_reg = Wc + regularize * torch.eye(A.shape[0])
        self.Wc_inv = torch.inverse(Wc_reg)

    def cost(self, x):
        d = (self.x_target -  x @ self.m_exp.T)
        return d @ self.Wc_inv @ d.T

    def forward(self, t, y):
        s = self.T - t
        term = torch.matrix_exp(self.A.T * s)
        u = ((self.x_target - y @ self.m_exp.T) @ self.Wc_inv.T) @ term.T @ self.B
        return u  # return 1D tensor
    

class StepController(nn.Module):
    def __init__(self):
        super().__init__()
        #self.ts = torch.linspace(0,5,100) # TODO
        #self.u = torch.zeros((1,100))

    def set_control(self, u: torch.tensor, ts: torch.tensor):
        self.ts = ts
        self.u = u

    def forward(self, t, x):
        idx = (self.ts - t).abs().argmin()
        return self.u[:, idx]
