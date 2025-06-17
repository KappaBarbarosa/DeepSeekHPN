import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.

        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.transformer_embed_dim
        self.args = args
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Linear(self.dim, args.n_routed_experts)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 64 else None
        self.log_interval = 1
        self.count = 0
        self.score_history = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = self.weight(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        self.score_history.append(scores[-1].detach().cpu().numpy())
        if self.count % self.log_interval == 0:
            # print(f'adding bias:', scores[-1])
            self.count = 0
        self.count += 1
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices
    def start_one_training(self):
        self.count = 0
        self.score_history = []
        
    def end_one_training(self,t_env):
        self.count = 0
        score_history = np.array(self.score_history)  # shape: (iterations, n_experts)
        plt.figure(figsize=(10, 6))
        n_experts = score_history.shape[1]
        # 為每個 expert 畫一條曲線
        for i in range(n_experts):
            plt.plot(score_history[:, i], label=f'Expert {i}')
        self.score_history = []  # 清理歷史記錄
        plt.xlabel('Iteration')
        plt.ylabel('Gate Score')
        plt.title('Gate Score over Training Iterations')
        plt.legend()
        root = '/home/marl2024/DeepSeekHPN/results/gate_scores'
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root, self.args.name)
        if not os.path.exists(root):
            os.mkdir(root)
        save_dir = os.path.join(root,self.args.env_args['map_name'])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(f'{save_dir}/{t_env}.png')
        plt.close()
        
        
