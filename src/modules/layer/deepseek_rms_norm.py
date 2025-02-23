import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

def rms_norm(x, weight, eps=1e-6):
    norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return weight * (x / norm)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return rms_norm(x, self.weight, self.eps)