import torch
import torch.nn as nn

class SimpleLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, apply_layernorm_1p=False):
        """
        輕量版 LayerNorm，不使用 apex、不處理 sequence parallel。
        Args:
            normalized_shape: 通常是 hidden_size
            eps: 防止除以零的小常數
            apply_layernorm_1p: 若為 True，表示 weight = weight + 1（可選）
        """
        super().__init__()
        self.apply_layernorm_1p = apply_layernorm_1p
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        if self.apply_layernorm_1p:
            # +1 trick（較少用）
            return self.layer_norm(x + 1)
        return self.layer_norm(x)