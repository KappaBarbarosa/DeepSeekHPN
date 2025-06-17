
import torch
import torch.nn as nn
from .deepseek_mlp import MLP
from .deepseek_expert import Expert
from .deepseeek_utils import *
from .gate import REGISTRY as gate_registry

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
        self.dim = args.transformer_embed_dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts 
        self.n_additional_experts = args.n_additional_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate_mode = args.gate
        self.gate = gate_registry[args.gate](args)
        self.moe_inter_dim = args.moe_inter_dim
        self.experts = nn.ModuleList([Expert(self.dim, args.moe_inter_dim) 
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(self.dim, args.n_shared_experts * args.moe_inter_dim)
        self.log_interval = 1
        self.count = 0
    
    def add_additional_experts(self):
        """
        Adds additional experts to the MoE module.
        This is a placeholder for future implementations.
        """
        for i in range(self.n_additional_experts):
            self.experts.append(Expert(self.dim, self.n_local_experts * self.moe_inter_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        if self.gate_mode == 'v1':
            weights, indices = self.gate(x, self.n_activated_experts)
        else:
            weights, indices, stats = self.gate(x)

        # print(f'indice shape:', indices.shape, 'last choice :', indices[-1])
        y = torch.zeros_like(x)
        
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        if self.count % self.log_interval == 0:
            # print(f'indices:', indices)
            # print(f'weights:', weights)
            # print(f'counts:', counts)
            self.count=0
        self.count += 1
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)

        if self.gate_mode != 'v1':
            return (y + z).view(shape), stats
        else:
            return (y + z).view(shape)