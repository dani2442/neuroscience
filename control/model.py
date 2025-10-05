import torch
import torch.nn as nn

class LinearControlSDE(torch.nn.Module):
    """
    General linear SDE with control input:
    dx = (Ax + Bu)dt + σdW
    
    where:
    - x is the state vector
    - u is the control input
    - A is the system matrix
    - B is the control input matrix
    - σ is the diffusion coefficient
    - W is a Wiener process
    """
    noise_type = 'diagonal'
    sde_type = 'ito'
    
    def __init__(self, A, B, control_input: callable = None, sigma=0.):
        """
        Args:
            A: System matrix (n x n)
            B: Control matrix (n x m)
            sigma: Diffusion coefficient (scalar or vector)
        """
        super().__init__()
        self.A = A
        self.B = B
        self.sigma = sigma
        self.state_size = A.shape[0]
        self.control_size = B.shape[1]
        self.control_input = control_input
        
    def f(self, t, y):
        """Drift function: Ax + Bu"""
        # Get control input at time t
        dx = torch.matmul(y, self.A.T)

        if self.control_input is not None:
            u = self.control_input(t, y)
            dx += torch.matmul(u, self.B.T)
        
        return dx
    
    def g(self, t, y):
        """Diffusion function: σ"""
        batch_size = y.shape[0]
        #if isinstance(self.sigma, (int, float)):
        state_diffusion = self.sigma * torch.ones(batch_size, self.state_size, 
                                          device=y.device, dtype=y.dtype)
        # control_diffusion = torch.zeros(batch_size, self.control_size,
        #                                device=y.device, dtype=y.dtype)
        
        return state_diffusion #torch.cat([state_diffusion, control_diffusion], dim=1)
        #return self.sigma.expand(batch_size, -1)



class SpringMassDamperSDE(LinearControlSDE):
    """
    Spring-mass-damper system:
    [ẋ₁]   [    0      1   ] [x₁]   [ 0  ]
    [ẋ₂] = [ -k/m   -c/m   ] [x₂] + [1/m ] u
    
    where:
    - x₁ is position
    - x₂ is velocity
    - k is spring constant
    - c is damping coefficient
    - m is mass
    - u is external force
    """
    
    def __init__(self, m=1.0, k=1.0, c=0.5, sigma=0.05):
        """
        Args:
            m: mass
            k: spring constant
            c: damping coefficient
            sigma: diffusion coefficient (noise intensity)
        """
        self.m = m
        self.k = k
        self.c = c
        
        # System matrix A
        A = torch.tensor([[0.0, 1.0],
                         [-k/m, -c/m]])
        
        # Control matrix B
        B = torch.tensor([[0.0],
                         [1.0/m]])
        
        super().__init__(A, B, sigma)
    
    def control_input(self, t, y):
        """
        Control law: Proportional-Derivative (PD) controller
        u = -Kp * x₁ - Kd * x₂
        (Targeting equilibrium at origin)
        """
        batch_size = y.shape[0]
        
        # PD gains
        Kp = 0.5
        Kd = 0.3
        
        # Extract position and velocity
        x1 = y[:, 0:1]  # position
        x2 = y[:, 1:2]  # velocity
        
        # PD control
        u = -Kp * x1 - Kd * x2
        
        return u
    

