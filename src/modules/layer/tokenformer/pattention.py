import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from modules.layer.tokenformer.init_function import get_init_methods

class Pattention(nn.Module):
    """Pattention Layer.
    """

    def __init__(
        self,
        args,
        input_channels,
        output_channels,
        param_token_num,
        param_key_init_method,
        param_value_init_method,
    ):
        super().__init__()

        if args.transfer and args.n_new_tokens > 0:
            self.param_token_num = param_token_num + args.n_new_tokens
        else:
            self.param_token_num = param_token_num
            
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = 'l2_norm_gelu'
        
        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim)))
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim)))
        
        param_key_init_method(self.key_param_tokens)
        param_value_init_method(self.value_param_tokens)
    
    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = norm_outputs
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
            outputs = norm_outputs
        elif normalize_type == 'l2_norm_gelu':
            eps = 1e-8  # or smaller like 1e-12
            norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
            norm = torch.clamp(norm, min=eps)
            
            norm_outputs = inputs / norm * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
            # print ("inputs", inputs[0])
            # print ("norm_outputs", norm_outputs[0])
            # print ("nonlinear_outputs", nonlinear_outputs[0])
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None):

        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 
        # just for gelu nonlinear, set torch.zeros for softmax
        attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # just for gelu nonlinear, set -inf for softmax
                attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        # print ("query", query[0])
        # print ("key", key[0])
        # print ("value", value[0])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # just for gelu nonlinear, set attn_weight += attn_bias for softmax
        attn_weight *= attn_bias
        # modified softmax
        # print ("attn_weight attn_weight *= attn_bias", attn_weight[0])
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        # print ("attn_weight", attn_weight[0])
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # print ("attn_weight", attn_weight[0])
        output = attn_weight @ value
        # print ("output", output[0])
        return output
