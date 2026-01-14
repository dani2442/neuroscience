import torch
import torch.nn.functional as F
import numpy as np

def differentiable_hilbert(x):
    """
    Computes the analytic signal using FFT (differentiable Hilbert transform).
    Input: (Batch, N_regions, Time) or (N_regions, Time)
    """
    # Ensure input is 3D: (Batch, N, T)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    
    N_fft = x.shape[-1]
    f = torch.fft.fft(x, n=N_fft, dim=-1)
    
    # Create the Hilbert mask
    h = torch.zeros(N_fft, device=x.device)
    if N_fft % 2 == 0:
        h[0] = h[N_fft // 2] = 1
        h[1:N_fft // 2] = 2
    else:
        h[0] = 1
        h[1:(N_fft + 1) // 2] = 2
        
    # Apply mask and inverse FFT
    analytic = torch.fft.ifft(f * h, dim=-1)
    return analytic

def diff_phase_interaction_matrix(ts):
    """
    Differentiable calculation of phase interaction.
    Input: (Batch, N, T)
    """
    # 1. Analytic Signal -> Phase
    analytic = differentiable_hilbert(ts)
    phases = torch.angle(analytic) # (Batch, N, T)
    
    # 2. Phase Difference
    # We use broadcasting to get (Batch, N, N, T)
    # Note: cos(a - b) is identical to the complex wrapping logic in the original code
    # because cos(x) == cos(2pi - x) == cos(|x|).
    phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
    
    # 3. Phase Interaction (cos)
    ph_int = torch.cos(phase_diff) # (Batch, N, N, T)
    
    # Permute to match original shape logic (Batch, T, N, N)
    return ph_int.permute(0, 3, 1, 2)

def diff_phFCD(ts, window_size=3):
    """
    Vectorized, differentiable Phase FCD.
    """
    ts = ts.permute(0,2,1)  # (Batch, N, T)
    batch, N, Tmax = ts.shape
    
    # 1. Get Phase Interaction Matrix
    ph_int = diff_phase_interaction_matrix(ts) # (Batch, T, N, N)
    
    # 2. Extract Upper Triangle
    # Create mask for upper triangle (k=1)
    # We flatten the N*N matrix into a vector of edges E = N*(N-1)/2
    triu_idx = torch.triu_indices(N, N, offset=1, device=ts.device)
    # ph_int_edges shape: (Batch, T, E)
    ph_int_edges = ph_int[:, :, triu_idx[0], triu_idx[1]]
    
    # 3. Sliding Window Mean (Vectorized using AvgPool1d)
    # We treat the time dimension as the "spatial" dimension for 1D convolution
    # Input to pool needs to be (Batch, Channels, Length) -> (Batch, E, T)
    ph_edges_perm = ph_int_edges.permute(0, 2, 1)
    
    # avg_pool1d computes the mean over the window
    # stride=1 gives us the sliding window effect
    windowed_means = F.avg_pool1d(ph_edges_perm, kernel_size=window_size, stride=1)
    
    # Result shape: (Batch, E, T_windows)
    # Transpose back to (Batch, T_windows, E) to treat each timepoint as a vector
    B = windowed_means.permute(0, 2, 1)
    
    # 4. Compute Cosine Similarity between all time-pairs (FCD Matrix)
    # Normalize vectors first
    norm = torch.norm(B, p=2, dim=2, keepdim=True)
    B_norm = B / (norm + 1e-8) # Add epsilon for stability
    
    # FCD Matrix: (Batch, T_windows, T_windows) via matrix multiplication
    FCD_matrix = torch.bmm(B_norm, B_norm.transpose(1, 2))
    
    # 5. Extract Upper Triangle of the FCD Matrix (The "distribution")
    T_win = FCD_matrix.shape[1]
    fcd_triu_idx = torch.triu_indices(T_win, T_win, offset=1, device=ts.device)
    
    # Result: (Batch, Number_of_FCD_Pairs)
    fcd_distribution = FCD_matrix[:, fcd_triu_idx[0], fcd_triu_idx[1]]
    
    return fcd_distribution

def wasserstein_loss_1d(x, y):
    """
    1-Wasserstein distance (Earth Mover's Distance) for 1D distributions.
    For 1D, this is the L1 distance between the sorted values (quantiles).
    """
    # Sort both distributions
    x_sorted, _ = torch.sort(x, dim=-1)
    y_sorted, _ = torch.sort(y, dim=-1)
    
    # If shapes mismatch (rare if T is constant), we would need interpolation.
    # Assuming T is constant for optimization:
    return torch.mean(torch.abs(x_sorted - y_sorted))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create a "Target" Signal (e.g., empirical BOLD data)
    # (1 Batch, 20 Regions, 200 Timepoints)
    target_ts_np = np.random.rand(1, 20, 200).astype(np.float32) 
    target_ts = torch.tensor(target_ts_np, device=device)

    # 2. Compute Target Statistics (No gradient needed here)
    with torch.no_grad():
        target_fcd_dist = diff_phFCD(target_ts)

    # 3. Create a Learnable Input
    # This could be the weights of a neural mass model, or just raw timeseries
    # Here we optimize raw timeseries directly
    learnable_ts = torch.randn(1, 20, 200, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([learnable_ts], lr=0.05)

    # --- Optimization Loop ---
    print("Starting optimization...")
    for i in range(100):
        optimizer.zero_grad()
        
        # Forward pass: Compute FCD of current estimate
        current_fcd_dist = diff_phFCD(learnable_ts)
        
        # Calculate Loss: 1-Wasserstein distance
        loss = wasserstein_loss_1d(current_fcd_dist, target_fcd_dist)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}, Loss (Wasserstein): {loss.item():.6f}")

    print("Optimization complete.")