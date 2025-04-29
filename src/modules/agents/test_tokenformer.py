import torch
import pytest
from modules.layer.tokenformer.tokenformer import TokenformerLayer
from modules.layer.tokenformer.init_function import get_init_methods

class TestTokenformerLayer:

    @pytest.fixture
    def mock_neox_args(self):
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
            pos_emb = "False"
            init_method = "small_init"
            output_layer_init_method = "wang_init"
            rnn_hidden_dim = 64
            mup_init_scale = 1
            num_layers = 1
            rope_fusion = False
            ffn_slot_num = 3
            hidden_dropout = 0.1
            bias_dropout_fusion = False
            mlp_type = "tokenformer"
        return Args()

    @pytest.fixture
    def tokenformer_layer(self, mock_neox_args):
        attention_mask_func = lambda x, y: x
        init_method, output_layer_init_method = get_init_methods(mock_neox_args)
        return TokenformerLayer(
            neox_args=mock_neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=1,
            rotary=True,
            use_cache=True
        )

    def test_tokenformer_output_shape(self, tokenformer_layer, mock_neox_args):
        batch_size = 2
        seq_length = 10
        hidden_size = mock_neox_args.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        prev_hh = torch.randn(batch_size, 1, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length, seq_length)

        output, new_hh = tokenformer_layer(hidden_states, attention_mask, hh=prev_hh)

        assert output.shape == (batch_size, seq_length + 1, hidden_size)
        assert new_hh.shape == (batch_size, 1, hidden_size)

    def test_tokenformer_device(self, tokenformer_layer):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tokenformer_layer.to(device)
            assert next(tokenformer_layer.parameters()).is_cuda

    def test_tokenformer_forward_pass(self, tokenformer_layer):
        batch_size = 4
        seq_length = 16
        hidden_size = tokenformer_layer.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        hh = torch.randn(batch_size, 1, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length, seq_length)

        try:
            output, new_hh = tokenformer_layer(hidden_states, attention_mask, hh=hh)
            assert output.shape == (batch_size, seq_length + 1, hidden_size)
            assert new_hh.shape == (batch_size, 1, hidden_size)
        except Exception as e:
            pytest.fail(f"前向傳播失敗: {str(e)}")

    def test_tokenformer_zero_mask(self, tokenformer_layer):
        batch_size = 2
        seq_length = 8
        hidden_size = tokenformer_layer.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        hh = torch.randn(batch_size, 1, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length, seq_length)

        output, new_hh = tokenformer_layer(hidden_states, attention_mask, hh=hh)
        assert output.shape == (batch_size, seq_length + 1, hidden_size)
        assert new_hh.shape == (batch_size, 1, hidden_size)

    def test_tokenformer_single_token(self, tokenformer_layer):
        batch_size = 2
        seq_length = 1
        hidden_size = tokenformer_layer.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        hh = torch.randn(batch_size, 1, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length, seq_length)

        output, new_hh = tokenformer_layer(hidden_states, attention_mask, hh=hh)
        assert output.shape == (batch_size, seq_length + 1, hidden_size)
        assert new_hh.shape == (batch_size, 1, hidden_size)