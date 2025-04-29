import torch
import torch.nn as nn
import pytest
from modules.layer.tokenformer.SelfAttention import SelfAttention
from modules.layer.tokenformer.init_function import get_init_methods
# 測試類別
class TestSelfAttention:

    @pytest.fixture
    def mock_neox_args(self):
        """模擬一個 Neox 設定參數"""
        class Args:
            precision = "fp32"
            hidden_size = 64
            num_attention_heads = 1
            apply_query_key_layer_scaling = False
            attention_softmax_in_fp32 = False
            rotary_pct = 1.0
            rotary_emb_base = 10000
            seq_length = 128
            params_dtype = torch.float32
            rotary_save_freqs_buffer = False
            attention_dropout = 0.1
            use_mup = False
            qkv_slot_num = 3
            proj_slot_num = 3
            sliding_window_width = None
            pos_emb = "none"
            init_method = "small_init"
            output_layer_init_method = "wang_init"
            rnn_hidden_dim = 64
            mup_init_scale = 1
            layer_number = 1
            rope_fusion = False
        
        return Args()

    @pytest.fixture
    def self_attention_layer(self, mock_neox_args):
        """建立 SelfAttention 物件"""
        attention_mask_func = lambda x, y: x  # 假的遮罩函數
        init_method, output_layer_init_method = get_init_methods(mock_neox_args)
        param_key_init_method = init_method
        param_value_init_method = output_layer_init_method
        return SelfAttention(
            neox_args=mock_neox_args,
            attention_mask_func=attention_mask_func,
            param_key_init_method=param_key_init_method,
            param_value_init_method=param_value_init_method,
            layer_number=1,
            rotary=True,
            use_cache=False,
            parallel_output=False
        )

    def test_self_attention_output_shape(self, self_attention_layer, mock_neox_args):
        """測試輸入輸出維度是否一致"""
        batch_size = 2
        seq_length = 8
        hidden_size = mock_neox_args.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = None
        print ("hidden_states.shape", hidden_states.shape)
        output = self_attention_layer(hidden_states, attention_mask)
        
        assert output.shape == (batch_size, seq_length, hidden_size), \
            f"輸出維度錯誤: {output.shape}，應該為 {(batch_size, seq_length, hidden_size)}"

    def test_self_attention_rotary_embedding(self, mock_neox_args):
        """測試旋轉嵌入是否能夠正確應用"""
        mock_neox_args.rotary_pct = 1.0
        init_method, output_layer_init_method = get_init_methods(mock_neox_args)
        self_attention_layer = SelfAttention(
            neox_args=mock_neox_args,
            attention_mask_func=lambda x, y: x,
            param_key_init_method=init_method,
            param_value_init_method=output_layer_init_method,
            layer_number=1,
            rotary=True,  # 啟用旋轉嵌入
            use_cache=False,
            parallel_output=False
        )

        batch_size = 2
        seq_length = 10
        hidden_size = mock_neox_args.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = None

        output = self_attention_layer(hidden_states, attention_mask)
        assert output.shape == (batch_size, seq_length, hidden_size), \
            "啟用旋轉嵌入後輸出形狀不匹配"

    def test_self_attention_device(self, self_attention_layer):
        """測試模型是否能夠使用 GPU"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self_attention_layer.to(device)
            assert next(self_attention_layer.parameters()).is_cuda, "模型應該被轉移到 GPU 上"

    def test_self_attention_forward_pass(self, self_attention_layer):
        """測試前向傳播是否可以順利執行"""
        batch_size = 4
        seq_length = 16
        hidden_size = self_attention_layer.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = None

        try:
            output = self_attention_layer(hidden_states, attention_mask)
        except Exception as e:
            pytest.fail(f"前向傳播失敗: {str(e)}")