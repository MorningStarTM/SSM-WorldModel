import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSSM(nn.Module):
    """
    x_{t+1} = A x_t + B u_t

    x_t: (B, state_dim)
    u_t: (B, action_dim)
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # A: (state_dim x state_dim)
        self.A = nn.Parameter(0.01 * torch.randn(state_dim, state_dim))
        # B: (state_dim x action_dim)
        self.B = nn.Parameter(0.01 * torch.randn(state_dim, action_dim))

    def step(self, x_t, u_t):
        """
        One-step prediction.
        x_t: (B, state_dim)
        u_t: (B, action_dim)
        returns x_hat_{t+1}: (B, state_dim)
        """
        x_next = x_t @ self.A.T + u_t @ self.B.T
        return x_next

    def rollout(self, x0, u_seq):
        """
        Multi-step rollout in open-loop.

        x0:    (B, state_dim)
        u_seq: (B, H, action_dim)  # controls for H steps

        returns:
            xs: (B, H+1, state_dim)  # [x0, x1, ..., xH]
        """
        B, H, _ = u_seq.shape
        xs = [x0]
        x_t = x0
        for t in range(H):
            u_t = u_seq[:, t, :]
            x_t = self.step(x_t, u_t)
            xs.append(x_t)
        xs = torch.stack(xs, dim=1)
        return xs
