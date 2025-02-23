import torch
import torch.nn as nn
import math
from typing import Optional
from .deepseeek_utils import *
from .deepseek_rms_norm import RMSNorm

class MLA(nn.Module):
    """
    Standard Multi-Head Attention Layer (MHA) without Cache.
    """

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim  # 模型總維度
        self.n_heads = args.n_heads  # 注意力頭數
        self.n_local_heads = args.n_heads  # 本地注意力頭數（可能用於分佈式）

        # Query, Key, Value 的拆分維度
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 非位置部分維度
        self.qk_rope_head_dim = args.qk_rope_head_dim      # 位置部分（RoPE）的維度
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 每個頭總維度
        self.v_head_dim = args.v_head_dim  # Value 維度

        # 確保設定正確
        assert self.qk_head_dim == self.qk_nope_head_dim + self.qk_rope_head_dim, (
            f"qk_head_dim ({self.qk_head_dim}) != qk_nope_head_dim ({self.qk_nope_head_dim}) + "
            f"qk_rope_head_dim ({self.qk_rope_head_dim})"
        )

        # Q, K, V 投影層（這裡假定 ColumnParallelLinear 和 RowParallelLinear 已定義）
        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        self.wk = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        self.wv = nn.Linear(self.dim, self.n_heads * self.v_head_dim)

        # 輸出層
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        # Softmax 縮放因子
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for Multi-Head Attention (without Cache).
        """
        bsz, seqlen, _ = x.size()

        # 計算 Q, K, V，並 reshape 為 [bsz, seqlen, n_heads, head_dim]
        q = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)

        # 拆分成非位置部分與位置部分（最後一維拆分）
        # q_nope: [bsz, seqlen, n_heads, qk_nope_head_dim]
        # q_pe:   [bsz, seqlen, n_heads, qk_rope_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 將位置部分應用 RoPE（假設 apply_rotary_emb() 不改變前3個維度）
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        k_pe = apply_rotary_emb(k_pe, freqs_cis)
        # 重新 reshape（如果 apply_rotary_emb 改變了形狀，則恢復為 [bsz, seqlen, n_heads, qk_rope_head_dim]）
        q_pe = q_pe.view(bsz, seqlen, self.n_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.view(bsz, seqlen, self.n_local_heads, self.qk_rope_head_dim)

        # 驗證除了最後一個維度之外，其他維度是否一致（bsz, seqlen, n_heads）
        assert q_nope.shape[:-1] == q_pe.shape[:-1], \
            f"q_nope.shape {q_nope.shape} 與 q_pe.shape {q_pe.shape} 的前三個維度不匹配"
        assert k_nope.shape[:-1] == k_pe.shape[:-1], \
            f"k_nope.shape {k_nope.shape} 與 k_pe.shape {k_pe.shape} 的前三個維度不匹配"

        # 拼接回完整的 Q 與 K，最後一維變為 qk_head_dim
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # 計算注意力分數：scores = Q K^T * scaling
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 計算注意力加權 V
        x = torch.einsum("bsht,bthd->bshd", scores, v)

        # Flatten 最後兩個維度，並投影回輸出維度
        x = self.wo(x.flatten(2))
        return x
