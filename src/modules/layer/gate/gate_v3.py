import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Gate_v3(nn.Module):
    """
    Gate that routes tokens to specific experts based on their position in the sequence.
    - First token (self info) -> Expert 0
    - Middle tokens (enemy info) -> Expert 1
    - Last tokens (ally info) -> Expert 2
    
    Input shape: (batch_size * n_agents * seq_len, dim)
    where seq_len = 1 + n_enemies + (n_agents-1) + 1
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.transformer_embed_dim
        self.args = args
        self.topk = args.n_activated_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass for the gating mechanism.
        Routes tokens to specific experts based on their position.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * n_agents * seq_len, dim)
                where seq_len = 1 + n_enemies + (n_agents-1) + 1

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: 
                - Routing weights
                - Selected expert indices
                - Statistics dictionary
        """
        batch_size_agents_seq, _ = x.shape
        seq_len = 1 + self.args.n_enemies + (self.args.n_agents - 1) + 1
        batch_size_agents = batch_size_agents_seq // seq_len
        
        # Reshape input to (batch_size_agents, seq_len, dim)
        x_reshaped = x.view(batch_size_agents, seq_len, -1)
        
        # Initialize weights and indices tensors
        weights = torch.zeros(batch_size_agents, seq_len, self.topk, device=x.device)
        indices = torch.zeros(batch_size_agents, seq_len, self.topk, dtype=torch.long, device=x.device)
        
        # Create a probability distribution for stats
        probs = torch.zeros(batch_size_agents, seq_len, self.args.n_routed_experts, device=x.device)
        
        # Route first token (self info) to Expert 0
        weights[:, 0, 0] = 1.0
        indices[:, 0, 0] = 0
        probs[:, 0, 0] = 1.0
        
        # Route middle tokens (enemy info) to Expert 1
        n_enemies = self.args.n_enemies
        weights[:, 1:1+n_enemies, 0] = 1.0
        indices[:, 1:1+n_enemies, 0] = 1
        probs[:, 1:1+n_enemies, 1] = 1.0
        
        # Route last tokens (ally info) to Expert 2
        n_allies = self.args.n_agents - 1
        weights[:, 1+n_enemies:1+n_enemies+n_allies, 0] = 1.0
        indices[:, 1+n_enemies:1+n_enemies+n_allies, 0] = 2
        probs[:, 1+n_enemies:1+n_enemies+n_allies, 2] = 1.0
        
        # Route hidden state token to Expert 0 (self info expert)
        weights[:, -1, 0] = 1.0
        indices[:, -1, 0] = 0
        probs[:, -1, 0] = 1.0
        
        # Reshape back to original shape
        weights = weights.view(batch_size_agents_seq, self.topk)
        indices = indices.view(batch_size_agents_seq, self.topk)
        probs = probs.view(batch_size_agents_seq, self.args.n_routed_experts)
        
        # Create stats dictionary
        stats = {
            "probs": probs.detach().cpu(),
            "indices": indices.detach().cpu()
        }
        
        return weights, indices, stats
