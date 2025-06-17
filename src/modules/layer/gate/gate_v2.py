
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

class Gate_v2(nn.Module):
    """
    Gate with dynamic bias, noise injection, group routing,
    and optional stats output for auxiliary loss.
    """
    def __init__(self, args):
        super().__init__()
        self.dim           = args.transformer_embed_dim
        self.topk          = args.n_activated_experts      # K
        self.n_groups      = args.n_expert_groups          # G_total
        self.topk_groups   = args.n_limited_groups         # G_keep
        self.route_scale   = args.route_scale
        # self.noise_std     = args.noise_std                # e.g. 1.0 → 0.0
        self.weight        = nn.Parameter(torch.empty(args.n_routed_experts,
                                                      args.transformer_embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # ---- 動態偏置；不參與梯度 ----
        self.register_buffer("dyn_bias",
                             torch.zeros(args.n_routed_experts))

    # --------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                return_router_stats: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, dict]:

        B, E = x.size(0), self.weight.size(0)              # E = n_experts
        logits = F.linear(x, self.weight)                  # (B, E)

        # ① 加隨機噪聲（訓練期才開）
        # if self.training and self.noise_std > 0:
        #     logits = logits + torch.randn_like(logits) * self.noise_std

        # ② 加動態偏置，只影響 top‑k 決策
        route_logits = logits + self.dyn_bias              # (B, E)

        # ④ 選 top‑k 專家
        indices = route_logits.topk(self.topk, dim=-1).indices  # (B, K)
        
        # ⑤ softmax 權重（**不**含 dyn_bias）
        probs   = logits.softmax(dim=-1).detach()                   # (B, E)
        weights = probs.gather(1, indices) * self.route_scale  # (B, K)

        if return_router_stats:                            # 供外層計算 αᵢ βᵢ
            stats = {"probs": probs.detach().cpu(),
                     "indices": indices.detach().cpu()}
            return weights, indices, stats
        return weights, indices

    def start_one_training(self):
        self.count = 0
        self.score_history = []

    def end_one_training(self, t_env):
        if not self.score_history:
            return
        
        
        
    # --------------------------------------------------------------
    @torch.no_grad()
    def update_dyn_bias(self,
                        tokens_per_expert: torch.Tensor,
                        target: float,
                        gamma: float = 1e-2):
        """
        tokens_per_expert : 1-D (E,) 長度張量，統計本 batch 每個專家收到多少 token
        target            : 期望 token 數 (≈   total_tokens / E)
        gamma             : 偏置更新速度
        """
        overload = tokens_per_expert > target
        under    = tokens_per_expert < target
        self.dyn_bias[overload] -= gamma
        self.dyn_bias[under]    += gamma