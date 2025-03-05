
import torch
import torch.nn as nn
from .deepseek_mlp import MLP
from .deepseek_expert import Expert
from .deepseeek_utils import *

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts 
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.log_interval = 1
        self.count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        print(shape)
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        # print(f'indice shape:', indices.shape, 'last choice :', indices[-1])
        y = torch.zeros_like(x)
        
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        if self.count % self.log_interval == 0:
            # print(f'indices:', indices)
            # print(f'weights:', weights)
            print(f'counts:', counts)
            self.count=0
        self.count += 1
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)

        return (y + z).view(shape)