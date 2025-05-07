import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from modules.layer.tokenformer.init_function import get_init_methods
from modules.layer.tokenformer.pattention import Pattention
from modules.layer.tokenformer.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
    AliBi,
)

class SelfAttention(nn.Module):
    """標準 Self-Attention, 適用於單 GPU, 無需 Flash Attention。"""

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        param_key_init_method,
        param_value_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number

        # 不使用模型並行
        self.hidden_size = neox_args.hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // neox_args.num_attention_heads
        self.num_attention_heads = neox_args.num_attention_heads

        self.pos_emb = neox_args.pos_emb
        self.sliding_window_width = neox_args.sliding_window_width

        self.num_kv_heads = self.num_attention_heads
        self.kv_hidden_size = self.hidden_size

        self.query = Pattention(
            args=neox_args,
            input_channels=self.hidden_size,
            output_channels=self.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )
        self.key = Pattention(
            args=neox_args,
            input_channels=self.hidden_size,
            output_channels=self.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )
        self.value = Pattention(
            args=neox_args,
            input_channels=self.hidden_size,
            output_channels=self.hidden_size,
            param_token_num=neox_args.qkv_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(neox_args.num_attention_heads, 1, 0)

        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(self.hidden_size_per_attention_head * neox_args.rotary_pct)
            dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
            self.rotary_emb = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
        else:
            self.rotary_emb = None

        self.rope_fusion = neox_args.rope_fusion
        self.dropout_p = neox_args.attention_dropout
        self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output
        self.proj = Pattention(
            args=neox_args,
            input_channels=self.hidden_size,
            output_channels=self.hidden_size,
            param_token_num=neox_args.proj_slot_num,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
        )

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        標準 Scaled Dot-Product Attention。
        """
        # print ("query", query.shape)
        # print ("key", key.shape)
        # print ("value", value.shape)
        d_k = query.size(-1)  # 取得 key 維度
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # QK^T / sqrt(d_k)
        # print ("scores", scores.shape)
        # print ("mask", mask.shape)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
                # print ("mask", mask.shape)
            # [bs, 1, seq, seq] → [bs, heads, seq, seq]
            if mask.size(1) == 1 and query.size(1) > 1:
                mask = mask.expand(-1, query.size(1), -1, -1)

            # print ("scores", scores.shape)
            # print ("mask", mask.shape)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # 遮罩填充為負無窮大

        # print ("scores", scores.shape)
        attn_weights = F.softmax(scores, dim=-1)  # Softmax
        # print ("attn_weights", attn_weights.shape)
        attn_weights = self.attention_dropout(attn_weights)  # Dropout
        # print ("attn_weights", attn_weights.shape)
        # print ("answer", torch.matmul(attn_weights, value).shape)
        return torch.matmul(attn_weights, value), attn_weights  # 計算輸出

    def forward(self, hidden_states, attention_mask, layer_past=None):
        """
        前向傳播，計算 Self-Attention。
        """
        # hidden_states: [b, s, h]
        b, sq, h = hidden_states.size()
        hidden_states = hidden_states.transpose(0, 1)  # → [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> 3 [sq, b, np, hn]

        query_layer = self.query(hidden_states).view(sq, b, self.num_attention_heads, self.hidden_size_per_attention_head)
        key_layer = self.key(hidden_states).view(sq, b, self.num_attention_heads, self.hidden_size_per_attention_head)
        value_layer = self.value(hidden_states).view(sq, b, self.num_attention_heads, self.hidden_size_per_attention_head)

        # print('hidden state',hidden_states[0])
        # print('before query_layer',query_layer[0])
        # print('before key_layer',key_layer[0])

        # print ("query_layer", query_layer.shape)
        # print ("key_layer", key_layer.shape)
        # print ("value_layer", value_layer.shape)
        
        if self.rotary_emb is not None:
            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            # print ("cos", cos.shape)
            # print ("sin", sin.shape)
            # print ("offset", offset)
            # print ("query_layer", query_layer.shape)
            # print ("key_layer", key_layer.shape)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=offset)

        
        # print('before query_layer',query_layer[0])
        # print('before key_layer',key_layer[0])

        if layer_past is not None and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        # print('after query_layer',query_layer[0])
        # print('after key_layer',key_layer[0])

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        # query_layer: [seq_len, b, nhead, head_dim] → [b, nhead, seq_len, head_dim]
        query = query_layer.permute(1, 2, 0, 3)
        key = key_layer.permute(1, 2, 0, 3)
        value = value_layer.permute(1, 2, 0, 3)

        context_layer, attn_weights = self.scaled_dot_product_attention(query, key, value, attention_mask)

        # [b, np, sq, hn]
        # print ("context_layer", context_layer.shape)
        # print ("attn_weights", attn_weights.shape)
        # =====================
        context_layer = context_layer.transpose(1, 2)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # print ("context_layer", context_layer.shape)
        # =====================
        context_layer = context_layer.view(sq, b, self.hidden_size)
        # print ("context_layer", context_layer.shape)
        context_layer = context_layer.transpose(0, 1)
        # print ("context_layer", context_layer.shape)
        output = self.proj(context_layer)

        if self.use_cache:
            output = [output, present]
    
        
        # print('query',query[0])
        # print('value',value[0])
        # print('key',key[0])
        # print('context_layer',context_layer[0])
        # print('output',output[0])
        return output