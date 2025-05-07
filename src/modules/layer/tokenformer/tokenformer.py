import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layer.tokenformer.pattention import Pattention
from modules.layer.tokenformer.SelfAttention import SelfAttention
from modules.layer.tokenformer.simple_layer_norm import SimpleLayerNorm

class TokenformerLayer(nn.Module):
    """
    單 GPU 版本的 TokenformerLayer，不依賴模型並行或 Apex。
    Input/Output: [batch_size, seq_length, hidden_size]
    """
    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=True,
        use_cache=False,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.use_cache = use_cache

        self.hidden_size = neox_args.hidden_size
        self.hidden_dropout = neox_args.hidden_dropout

        self.input_layernorm = SimpleLayerNorm(self.hidden_size, eps=1e-5)
        self.attention = SelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            param_key_init_method=init_method,
            param_value_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            rotary=rotary,
            use_cache=use_cache,
        )

        self.post_attention_layernorm = SimpleLayerNorm(self.hidden_size, eps=1e-5)

        self.mlp = Pattention(
            args=neox_args,
            input_channels=self.hidden_size,
            output_channels=self.hidden_size,
            param_token_num=neox_args.ffn_slot_num,
            param_key_init_method=init_method,
            param_value_init_method=output_layer_init_method,
        )

        self.layer_past = None

    def forward(self, x, attention_mask, hh=None, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        if hh is not None:
            x = torch.cat([hh, x], dim=1)
        
        residual = x
        # print ("x.shape", x.shape)
        # print ("attention_mask.shape", attention_mask.shape)
        normed_input = self.input_layernorm(x)
        # print ("normed_input.shape", normed_input.shape)
        # if layer_past is not None:
            # print ("layer_past.shape", layer_past.shape)
        # print('normed input', normed_input[0])
        # print('mask',attention_mask[0])
        attention_output = self.attention(normed_input, attention_mask, layer_past=layer_past)
        # print('attention output', attention_output[0])
        # print ("attention_output.shape", attention_output.shape)
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents

        # Dropout + 殘差
        attention_output = F.dropout(attention_output, p=self.hidden_dropout, training=self.training) + residual

        # LayerNorm + MLP
        normed_attn = self.post_attention_layernorm(attention_output)
        mlp_output = self.mlp(normed_attn)

        # Dropout + 殘差
        output = F.dropout(mlp_output, p=self.hidden_dropout, training=self.training) + attention_output

        new_hh = output[:, 1, :] if hh is not None else None
        
        return output, new_hh