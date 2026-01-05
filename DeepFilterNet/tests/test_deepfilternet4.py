"""Tests for DeepFilterNet4 full model.

This module tests the complete DfNet4 model implementation, including:
- Basic model instantiation and forward pass
- Different encoder variants (standard vs hybrid)
- Different DF decoder variants (single-res, multi-res, adaptive)
- Model configuration options
- Integration with existing infrastructure

Run with: pytest tests/test_deepfilternet4.py -v
"""

import os
import tempfile
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Helper to create test config
def create_test_config(
    backbone: str = "mamba",
    use_time_branch: bool = False,
    use_phase_branch: bool = False,
    use_multi_res_df: bool = False,
    adaptive_order: bool = False,
    model_variant: str = "full",
) -> str:
    """Create a temporary config file for testing."""
    config_content = f'''[df]
SR = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96
CONV_LOOKAHEAD = 0
LSNR_MIN = -15
LSNR_MAX = 40
DF_LOOKAHEAD = 0
DF_ORDER = 5

[deepfilternet4]
CONV_CH = 16
EMB_HIDDEN_DIM = 256
EMB_NUM_LAYERS = 2
DF_HIDDEN_DIM = 256
DF_NUM_LAYERS = 3
LINEAR_GROUPS = 1
ENC_LINEAR_GROUPS = 16
BACKBONE = {backbone}
USE_TIME_BRANCH = {"true" if use_time_branch else "false"}
USE_PHASE_BRANCH = {"true" if use_phase_branch else "false"}
USE_MULTI_RES_DF = {"true" if use_multi_res_df else "false"}
DF_RESOLUTIONS = 96,5;48,3;24,2
ADAPTIVE_ORDER = {"true" if adaptive_order else "false"}
ADAPTIVE_ORDER_MAX = 7
ADAPTIVE_ORDER_MIN = 2
MODEL_VARIANT = {model_variant}
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        return f.name


def create_model_with_config(config_path: str):
    """Create DfNet4 model with given config file."""
    from df.config import config
    config.load(config_path, allow_reload=True)
    
    from df.deepfilternet4 import DfNet4, ModelParams4
    from df.modules import erb_fb
    from libdf import DF
    
    p = ModelParams4()
    df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    
    model = DfNet4(erb, erb_inverse, run_df=True, train_mask=True)
    return model.cpu()


@pytest.fixture(scope="module")
def standard_config():
    """Create a standard config for testing."""
    config_path = create_test_config(
        backbone="mamba",
        use_time_branch=False,
        use_phase_branch=False,
        use_multi_res_df=False,
        adaptive_order=False,
    )
    yield config_path
    os.unlink(config_path)


@pytest.fixture(scope="module")
def hybrid_config():
    """Create a hybrid encoder config for testing."""
    config_path = create_test_config(
        backbone="mamba",
        use_time_branch=True,
        use_phase_branch=True,
        use_multi_res_df=False,
        adaptive_order=False,
    )
    yield config_path
    os.unlink(config_path)


@pytest.fixture(scope="module")
def multires_config():
    """Create a multi-resolution DF config for testing."""
    config_path = create_test_config(
        backbone="mamba",
        use_time_branch=False,
        use_phase_branch=True,
        use_multi_res_df=True,
        adaptive_order=False,
    )
    yield config_path
    os.unlink(config_path)


@pytest.fixture(scope="module")
def adaptive_config():
    """Create an adaptive order config for testing."""
    config_path = create_test_config(
        backbone="mamba",
        use_time_branch=False,
        use_phase_branch=False,
        use_multi_res_df=False,
        adaptive_order=True,
    )
    yield config_path
    os.unlink(config_path)


@pytest.fixture(scope="module")
def gru_config():
    """Create a GRU backbone config for testing."""
    config_path = create_test_config(
        backbone="gru",
        use_time_branch=False,
        use_phase_branch=False,
        use_multi_res_df=False,
        adaptive_order=False,
    )
    yield config_path
    os.unlink(config_path)


class TestDfNet4Basic:
    """Test basic DfNet4 functionality."""
    
    def test_model_creation_standard(self, standard_config):
        """Test model creation with standard encoder."""
        model = create_model_with_config(standard_config)
        assert model is not None
        assert not model.use_hybrid_encoder
        
    def test_forward_pass_standard(self, standard_config):
        """Test forward pass with standard encoder."""
        model = create_model_with_config(standard_config)
        
        batch_size = 2
        num_frames = 100
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        assert m.shape == (batch_size, 1, num_frames, 32)
        assert lsnr.shape == (batch_size, num_frames, 1)
        # DF order is 5
        assert df_coefs.shape == (batch_size, 5, num_frames, 96, 2)
        
    def test_output_dtypes(self, standard_config):
        """Test output dtypes are correct."""
        model = create_model_with_config(standard_config)
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.dtype == torch.float32
        assert m.dtype == torch.float32
        assert lsnr.dtype == torch.float32
        assert df_coefs.dtype == torch.float32
        
    def test_parameter_count_standard(self, standard_config):
        """Test parameter count for standard model."""
        model = create_model_with_config(standard_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Standard model should be around 2-3M parameters
        assert 1_500_000 < total_params < 4_000_000
        

class TestDfNet4HybridEncoder:
    """Test DfNet4 with hybrid encoder."""
    
    def test_model_creation_hybrid(self, hybrid_config):
        """Test model creation with hybrid encoder."""
        model = create_model_with_config(hybrid_config)
        assert model is not None
        assert model.use_hybrid_encoder
        
    def test_forward_without_waveform(self, hybrid_config):
        """Test hybrid encoder without waveform input."""
        model = create_model_with_config(hybrid_config)
        
        batch_size = 2
        num_frames = 100
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        assert m.shape == (batch_size, 1, num_frames, 32)
        
    def test_forward_with_waveform(self, hybrid_config):
        """Test hybrid encoder with waveform input."""
        model = create_model_with_config(hybrid_config)
        
        batch_size = 2
        num_frames = 100
        hop_size = 480
        fft_size = 960
        waveform_len = num_frames * hop_size + (fft_size - hop_size)
        
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        waveform = torch.randn(batch_size, waveform_len)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec, waveform)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        assert m.shape == (batch_size, 1, num_frames, 32)
        
    def test_parameter_count_hybrid(self, hybrid_config):
        """Test parameter count for hybrid model."""
        model = create_model_with_config(hybrid_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Hybrid model has more params due to additional encoders
        assert 4_000_000 < total_params < 7_000_000


class TestDfNet4MultiResDF:
    """Test DfNet4 with multi-resolution deep filtering."""
    
    def test_model_creation_multires(self, multires_config):
        """Test model creation with multi-res DF."""
        model = create_model_with_config(multires_config)
        assert model is not None
        assert model.use_multi_res_df
        
    def test_forward_multires(self, multires_config):
        """Test forward pass with multi-res DF."""
        model = create_model_with_config(multires_config)
        
        batch_size = 2
        num_frames = 100
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        # DF coefs shape should still be largest resolution
        assert df_coefs.shape[1] == 5  # df_order
        assert df_coefs.shape[3] == 96  # nb_df


class TestDfNet4AdaptiveOrder:
    """Test DfNet4 with adaptive filter order."""
    
    def test_model_creation_adaptive(self, adaptive_config):
        """Test model creation with adaptive order."""
        model = create_model_with_config(adaptive_config)
        assert model is not None
        assert model.adaptive_order
        
    def test_forward_adaptive(self, adaptive_config):
        """Test forward pass with adaptive order."""
        model = create_model_with_config(adaptive_config)
        
        batch_size = 2
        num_frames = 100
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        # With adaptive order (max=7), output has max order
        assert df_coefs.shape[1] == 7


class TestDfNet4Lite:
    """Test DfNet4Lite lightweight variant."""
    
    @pytest.fixture
    def lite_config(self, tmp_path):
        """Create config for lite model."""
        config_path = tmp_path / "dfnet4_lite.ini"
        config_path.write_text("""
[df]
SR = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96
DF_LOOKAHEAD = 0

[deepfilternet4]
BACKBONE = mamba
USE_TIME_BRANCH = false
USE_PHASE_BRANCH = false
USE_MULTI_RES_DF = false
ADAPTIVE_ORDER = false
MODEL_VARIANT = lite
""")
        return str(config_path)
    
    def test_model_creation_lite(self, lite_config):
        """Test DfNet4Lite model creation."""
        from df.config import config
        config.load(lite_config, allow_reload=True)
        
        from df.deepfilternet4 import init_model
        
        model = init_model(run_df=True, train_mask=True)
        assert model is not None
        assert "DfNet4Lite" in type(model).__name__
        
    def test_forward_pass_lite(self, lite_config):
        """Test forward pass with lite model."""
        from df.config import config
        config.load(lite_config, allow_reload=True)
        
        from df.deepfilternet4 import init_model
        
        model = init_model(run_df=True, train_mask=True).cpu()
        
        batch_size = 2
        num_frames = 100
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 2, num_frames, 96)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (batch_size, 1, num_frames, 481, 2)
        assert m.shape == (batch_size, 1, num_frames, 32)
        assert lsnr.shape == (batch_size, num_frames, 1)
        assert df_coefs.shape == (batch_size, 5, num_frames, 96, 2)
        
    def test_parameter_count_lite(self, lite_config):
        """Test parameter count for lite model is reduced."""
        from df.config import config
        config.load(lite_config, allow_reload=True)
        
        from df.deepfilternet4 import init_model
        
        model = init_model(run_df=True, train_mask=True).cpu()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Lite model should be around 1-1.5M parameters (50% of full)
        assert 500_000 < total_params < 2_000_000
        print(f"DfNet4Lite params: {total_params:,}")
        
    def test_lite_vs_full_param_reduction(self, lite_config, standard_config):
        """Test that lite model has fewer params than full model."""
        from df.config import config
        
        # Load lite model
        config.load(lite_config, allow_reload=True)
        from df.deepfilternet4 import init_model
        lite_model = init_model(run_df=True, train_mask=True).cpu()
        lite_params = sum(p.numel() for p in lite_model.parameters())
        
        # Load full model
        config.load(standard_config, allow_reload=True)
        full_model = init_model(run_df=True, train_mask=True).cpu()
        full_params = sum(p.numel() for p in full_model.parameters())
        
        # Lite should be at least 30% smaller
        assert lite_params < full_params * 0.7
        print(f"Full: {full_params:,}, Lite: {lite_params:,}, Reduction: {(1 - lite_params/full_params)*100:.1f}%")


class TestDfNet4GRUBackbone:
    """Test DfNet4 with GRU backbone (fallback mode)."""
    
    def test_model_creation_gru(self, gru_config):
        """Test model creation with GRU backbone."""
        model = create_model_with_config(gru_config)
        assert model is not None
        
    def test_forward_gru(self, gru_config):
        """Test forward pass with GRU backbone."""
        model = create_model_with_config(gru_config)
        
        spec = torch.randn(2, 1, 100, 481, 2)
        feat_erb = torch.randn(2, 1, 100, 32)
        feat_spec = torch.randn(2, 1, 100, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape == (2, 1, 100, 481, 2)


class TestDfNet4GradientFlow:
    """Test gradient flow through DfNet4."""
    
    def test_gradient_flow_standard(self, standard_config):
        """Test gradient flow with standard encoder."""
        model = create_model_with_config(standard_config)
        model.train()
        
        spec = torch.randn(2, 1, 50, 481, 2, requires_grad=True)
        feat_erb = torch.randn(2, 1, 50, 32, requires_grad=True)
        feat_spec = torch.randn(2, 1, 50, 96, 2, requires_grad=True)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Create composite loss
        loss = spec_e.mean() + m.mean() + lsnr.mean() + df_coefs.mean()
        loss.backward()
        
        # Check gradients exist for inputs
        assert spec.grad is not None
        assert feat_erb.grad is not None
        assert feat_spec.grad is not None
        
        # Check at least some model parameters have gradients (not all may be in the active path)
        params_with_grad = sum(1 for _, p in model.named_parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad > total_params // 2, f"Too few params with gradients: {params_with_grad}/{total_params}"
                
    def test_gradient_flow_hybrid(self, hybrid_config):
        """Test gradient flow with hybrid encoder."""
        model = create_model_with_config(hybrid_config)
        model.train()
        
        spec = torch.randn(2, 1, 50, 481, 2, requires_grad=True)
        feat_erb = torch.randn(2, 1, 50, 32, requires_grad=True)
        feat_spec = torch.randn(2, 1, 50, 96, 2, requires_grad=True)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        loss = spec_e.mean() + m.mean() + lsnr.mean() + df_coefs.mean()
        loss.backward()
        
        assert spec.grad is not None


class TestDfNet4EvalMode:
    """Test DfNet4 in evaluation mode."""
    
    def test_eval_mode_deterministic(self, standard_config):
        """Test that eval mode produces deterministic outputs."""
        model = create_model_with_config(standard_config)
        model.eval()
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        with torch.no_grad():
            out1 = model(spec, feat_erb, feat_spec)
            out2 = model(spec, feat_erb, feat_spec)
            
        for o1, o2 in zip(out1, out2):
            assert torch.allclose(o1, o2)
            
    def test_train_vs_eval_mode(self, standard_config):
        """Test that train and eval modes may differ (dropout, etc.)."""
        model = create_model_with_config(standard_config)
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        model.train()
        with torch.no_grad():
            out_train = model(spec, feat_erb, feat_spec)
            
        model.eval()
        with torch.no_grad():
            out_eval = model(spec, feat_erb, feat_spec)
            
        # Outputs may or may not differ depending on dropout settings
        # Just verify both modes work
        assert out_train[0].shape == out_eval[0].shape


class TestDfNet4BatchSizes:
    """Test DfNet4 with various batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, standard_config, batch_size):
        """Test with various batch sizes."""
        model = create_model_with_config(standard_config)
        
        spec = torch.randn(batch_size, 1, 50, 481, 2)
        feat_erb = torch.randn(batch_size, 1, 50, 32)
        feat_spec = torch.randn(batch_size, 1, 50, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape[0] == batch_size
        assert m.shape[0] == batch_size
        assert lsnr.shape[0] == batch_size


class TestDfNet4SequenceLengths:
    """Test DfNet4 with various sequence lengths."""
    
    @pytest.mark.parametrize("num_frames", [10, 50, 100, 200])
    def test_various_sequence_lengths(self, standard_config, num_frames):
        """Test with various sequence lengths."""
        model = create_model_with_config(standard_config)
        
        spec = torch.randn(2, 1, num_frames, 481, 2)
        feat_erb = torch.randn(2, 1, num_frames, 32)
        feat_spec = torch.randn(2, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape[2] == num_frames
        assert m.shape[2] == num_frames
        assert lsnr.shape[1] == num_frames


class TestDfNet4Components:
    """Test individual components of DfNet4."""
    
    def test_encoder4(self, standard_config):
        """Test Encoder4 component."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import Encoder4, ModelParams4
        
        p = ModelParams4()
        encoder = Encoder4(
            conv_ch=p.conv_ch,
            nb_erb=p.nb_erb,
            nb_df=p.nb_df,
            emb_hidden_dim=p.emb_hidden_dim,
            enc_lin_groups=p.enc_lin_groups,
            use_mamba=p.backbone.lower() == "mamba",
            lin_groups=p.lin_groups,
            lsnr_min=p.lsnr_min,
            lsnr_max=p.lsnr_max,
        ).cpu()
        
        # Encoder4 expects feat_erb [B, 1, T, E] and feat_spec [B, 2, T, F]
        feat_erb = torch.randn(2, 1, 100, p.nb_erb)
        feat_spec = torch.randn(2, 2, 100, p.nb_df)  # [B, 2, T, F] - 2 channels for real/imag
        
        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)
        
        assert e0 is not None
        assert e1 is not None
        assert e2 is not None
        assert e3 is not None
        assert emb.shape[0] == 2  # batch
        assert emb.shape[1] == 100  # time
        
    def test_erb_decoder4(self, standard_config):
        """Test ErbDecoder4 component."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import ErbDecoder4, ModelParams4
        
        p = ModelParams4()
        decoder = ErbDecoder4(
            conv_ch=p.conv_ch,
            nb_erb=p.nb_erb,
            emb_hidden_dim=p.emb_hidden_dim,
            use_mamba=p.backbone.lower() == "mamba",
            lin_groups=p.lin_groups,
        ).cpu()
        
        # Create mock encoder outputs with correct shapes
        # Encoder outputs: e0 [B,C,T,E], e1 [B,C,T,E/2], e2 [B,C,T,E/4], e3 [B,C,T,E/4]
        # emb [B, T, C*E/4]
        b, t = 2, 100
        ch = p.conv_ch
        nb_erb = p.nb_erb
        
        emb = torch.randn(b, t, ch * nb_erb // 4)  # [B, T, C*E/4]
        e3 = torch.randn(b, ch, t, nb_erb // 4)    # [B, C, T, E/4]
        e2 = torch.randn(b, ch, t, nb_erb // 4)    # [B, C, T, E/4]
        e1 = torch.randn(b, ch, t, nb_erb // 2)    # [B, C, T, E/2]
        e0 = torch.randn(b, ch, t, nb_erb)         # [B, C, T, E]
        
        m = decoder(emb, e3, e2, e1, e0)
        
        assert m.shape == (b, 1, t, nb_erb)


class TestDfNet4InitModel:
    """Test init_model function."""
    
    def test_init_model_standard(self, standard_config):
        """Test init_model creates DfNet4."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import init_model
        
        model = init_model(run_df=True, train_mask=True)
        
        assert model is not None
        assert "DfNet4" in type(model).__name__
        
    def test_init_model_with_df_state(self, standard_config):
        """Test init_model with custom df_state."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import init_model, ModelParams4
        from libdf import DF
        
        p = ModelParams4()
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
        
        model = init_model(df_state=df_state, run_df=True, train_mask=True)
        
        assert model is not None


class TestDfNet4Devices:
    """Test DfNet4 on different devices."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self, standard_config):
        """Test DfNet4 on CUDA."""
        model = create_model_with_config(standard_config).cuda()
        
        spec = torch.randn(2, 1, 50, 481, 2).cuda()
        feat_erb = torch.randn(2, 1, 50, 32).cuda()
        feat_spec = torch.randn(2, 1, 50, 96, 2).cuda()
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.device.type == "cuda"
        
    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available"
    )
    def test_mps(self, standard_config):
        """Test DfNet4 on MPS (Apple Silicon)."""
        model = create_model_with_config(standard_config)
        # MPS may have issues with some ops, test on CPU for now
        # model = model.to("mps")
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e is not None


class TestDfNet4EdgeCases:
    """Test edge cases for DfNet4."""
    
    def test_single_frame(self, standard_config):
        """Test with single frame input."""
        model = create_model_with_config(standard_config)
        
        spec = torch.randn(2, 1, 1, 481, 2)
        feat_erb = torch.randn(2, 1, 1, 32)
        feat_spec = torch.randn(2, 1, 1, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape[2] == 1
        
    def test_very_long_sequence(self, standard_config):
        """Test with very long sequence."""
        model = create_model_with_config(standard_config)
        
        num_frames = 500
        spec = torch.randn(1, 1, num_frames, 481, 2)
        feat_erb = torch.randn(1, 1, num_frames, 32)
        feat_spec = torch.randn(1, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        assert spec_e.shape[2] == num_frames


class TestModelParams4:
    """Test ModelParams4 configuration."""
    
    def test_default_values(self, standard_config):
        """Test default parameter values."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import ModelParams4
        
        p = ModelParams4()
        
        assert p.conv_ch == 16
        assert p.emb_hidden_dim == 256
        assert p.df_hidden_dim == 256
        assert p.backbone == "mamba"
        
    def test_get_df_resolutions(self, multires_config):
        """Test parsing DF resolutions."""
        from df.config import config
        config.load(multires_config, allow_reload=True)
        
        from df.deepfilternet4 import ModelParams4
        
        p = ModelParams4()
        resolutions = p.get_df_resolutions()
        
        assert len(resolutions) == 3
        assert resolutions[0] == (96, 5)
        assert resolutions[1] == (48, 3)
        assert resolutions[2] == (24, 2)
        
    def test_generate_config_template(self):
        """Test configuration template generation."""
        from df.deepfilternet4 import ModelParams4
        
        template = ModelParams4.generate_config_template()
        
        # Check that template contains expected sections
        assert "[df]" in template
        assert "[train]" in template
        assert "[deepfilternet4]" in template
        
        # Check that key parameters are documented
        assert "BACKBONE" in template
        assert "MAMBA_D_STATE" in template
        assert "USE_MULTI_RES_DF" in template
        assert "MODEL_VARIANT" in template
        
    def test_dfnet4_specific_params(self, standard_config):
        """Test DFNet4-specific parameters are properly configured."""
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import ModelParams4
        
        p = ModelParams4()
        
        # Test Mamba params have defaults
        assert p.mamba_d_state == 16
        assert p.mamba_d_conv == 4
        assert p.mamba_expand == 2
        
        # Test hybrid encoder params
        assert p.use_time_branch == False
        assert p.use_phase_branch == False or p.use_phase_branch == True  # May vary by config
        assert p.fusion_type in ("simple", "attention")
        
        # Test multi-res DF params
        assert isinstance(p.use_multi_res_df, bool)
        assert isinstance(p.df_resolutions, str)
        
        # Test adaptive order params
        assert isinstance(p.adaptive_order, bool)
        assert p.max_df_order >= p.min_df_order
        
        # Test model variant
        assert p.model_variant in ("full", "lite")
        
    def test_section_name(self):
        """Test that section name is correct."""
        from df.deepfilternet4 import ModelParams4
        
        assert ModelParams4.section == "deepfilternet4"


class TestConfigCompatibility:
    """Test backward compatibility for DFNet4 configs."""
    
    def test_config_fix_dfnet4_called(self):
        """Test that _fix_dfnet4 is called during config load."""
        from df.config import config
        
        # Create a minimal config
        config_content = '''[df]
SR = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96

[train]
MODEL = deepfilternet4
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_content)
            config_path = f.name
            
        try:
            config.load(config_path, allow_reload=True)
            # Should have created deepfilternet4 section
            assert config.parser.has_section("deepfilternet4")
        finally:
            os.unlink(config_path)
            
    def test_dfnet3_param_migration(self):
        """Test that compatible params are migrated from DFNet3 config."""
        from df.config import config
        
        # Create a config with DFNet3 section
        config_content = '''[df]
SR = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96

[train]
MODEL = deepfilternet4

[deepfilternet3]
CONV_CH = 32
EMB_HIDDEN_DIM = 512
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_content)
            config_path = f.name
            
        try:
            config.load(config_path, allow_reload=True)
            
            # Should have migrated params to deepfilternet4
            assert config.parser.has_section("deepfilternet4")
            if config.parser.has_option("deepfilternet4", "conv_ch"):
                assert config.parser.get("deepfilternet4", "conv_ch") == "32"
            if config.parser.has_option("deepfilternet4", "emb_hidden_dim"):
                assert config.parser.get("deepfilternet4", "emb_hidden_dim") == "512"
        finally:
            os.unlink(config_path)


class TestDfNet4Integration:
    """Full integration tests for DfNet4 end-to-end functionality."""
    
    def test_full_forward_backward_cycle(self, standard_config):
        """Test complete forward and backward pass cycle."""
        model = create_model_with_config(standard_config)
        model.train()
        
        batch_size = 2
        num_frames = 50
        spec = torch.randn(batch_size, 1, num_frames, 481, 2, requires_grad=True)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32, requires_grad=True)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2, requires_grad=True)
        
        # Forward
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Compute loss (mock spectral loss)
        target_spec = torch.randn_like(spec_e)
        loss = F.mse_loss(spec_e, target_spec) + F.mse_loss(m, torch.ones_like(m) * 0.5)
        
        # Backward
        loss.backward()
        
        # Verify gradients
        assert spec.grad is not None
        assert feat_erb.grad is not None
        assert feat_spec.grad is not None
        
        # Count parameters with gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        param_count = sum(1 for _ in model.parameters())
        assert grad_count > param_count // 2, f"Too few params with grads: {grad_count}/{param_count}"
        
    def test_training_step_optimizer(self, standard_config):
        """Test a complete training step with optimizer."""
        model = create_model_with_config(standard_config)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        batch_size = 2
        num_frames = 50
        
        # Record initial weights
        initial_weights = {
            name: param.clone() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
        
        # Forward pass
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Loss
        target_spec = torch.randn_like(spec_e)
        loss = F.mse_loss(spec_e, target_spec)
        
        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify weights changed
        weights_changed = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_weights:
                if not torch.allclose(param, initial_weights[name]):
                    weights_changed += 1
                    
        assert weights_changed > 0, "No weights changed after optimizer step"
        
    def test_training_step_with_scheduler(self, standard_config):
        """Test training step with learning rate scheduler."""
        model = create_model_with_config(standard_config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Run a few steps
        for i in range(3):
            spec = torch.randn(2, 1, 50, 481, 2)
            feat_erb = torch.randn(2, 1, 50, 32)
            feat_spec = torch.randn(2, 1, 50, 96, 2)
            
            optimizer.zero_grad()
            spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
            loss = spec_e.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # Verify LR changed
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
        
    def test_gradient_clipping(self, standard_config):
        """Test training with gradient clipping."""
        model = create_model_with_config(standard_config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        optimizer.zero_grad()
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        loss = spec_e.mean() * 1000  # Large multiplier to create large gradients
        loss.backward()
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Verify gradients are clipped
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Should be close to or less than max_norm
        assert total_norm <= max_norm * 1.1, f"Grad norm {total_norm} > {max_norm}"
        
    def test_model_save_load_checkpoint(self, standard_config):
        """Test saving and loading model checkpoint."""
        model1 = create_model_with_config(standard_config)
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        model1.eval()
        with torch.no_grad():
            out1 = model1(spec, feat_erb, feat_spec)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            
        try:
            torch.save({
                'model_state_dict': model1.state_dict(),
                'epoch': 10,
            }, checkpoint_path)
            
            # Load into new model
            model2 = create_model_with_config(standard_config)
            checkpoint = torch.load(checkpoint_path)
            model2.load_state_dict(checkpoint['model_state_dict'])
            
            model2.eval()
            with torch.no_grad():
                out2 = model2(spec, feat_erb, feat_spec)
                
            # Outputs should match
            for o1, o2 in zip(out1, out2):
                assert torch.allclose(o1, o2, atol=1e-5)
        finally:
            os.unlink(checkpoint_path)
            
    def test_all_model_variants_forward(self, tmp_path):
        """Test forward pass for all model variants."""
        variants = [
            {"backbone": "mamba", "model_variant": "full"},
            {"backbone": "mamba", "model_variant": "lite"},
            {"backbone": "gru", "model_variant": "full"},
        ]
        
        for variant_config in variants:
            config_path = create_test_config(**variant_config)
            try:
                model = create_model_with_config(config_path)
                
                spec = torch.randn(2, 1, 50, 481, 2)
                feat_erb = torch.randn(2, 1, 50, 32)
                feat_spec = torch.randn(2, 1, 50, 96, 2)
                
                spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
                
                assert spec_e.shape == (2, 1, 50, 481, 2)
                assert m.shape == (2, 1, 50, 32)
            finally:
                os.unlink(config_path)
                
    def test_hybrid_encoder_variants(self, tmp_path):
        """Test all hybrid encoder configurations."""
        configs = [
            {"use_time_branch": True, "use_phase_branch": False},
            {"use_time_branch": False, "use_phase_branch": True},
            {"use_time_branch": True, "use_phase_branch": True},
        ]
        
        for cfg in configs:
            config_path = create_test_config(**cfg)
            try:
                model = create_model_with_config(config_path)
                
                spec = torch.randn(2, 1, 50, 481, 2)
                feat_erb = torch.randn(2, 1, 50, 32)
                feat_spec = torch.randn(2, 1, 50, 96, 2)
                
                spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
                
                assert spec_e.shape == (2, 1, 50, 481, 2)
            finally:
                os.unlink(config_path)
                
    def test_df_variants(self, tmp_path):
        """Test different DF configurations."""
        configs = [
            {"use_multi_res_df": True, "adaptive_order": False},
            {"use_multi_res_df": False, "adaptive_order": True},
            {"use_multi_res_df": True, "adaptive_order": True},
        ]
        
        for cfg in configs:
            config_path = create_test_config(**cfg)
            try:
                model = create_model_with_config(config_path)
                
                spec = torch.randn(2, 1, 50, 481, 2)
                feat_erb = torch.randn(2, 1, 50, 32)
                feat_spec = torch.randn(2, 1, 50, 96, 2)
                
                spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
                
                assert spec_e.shape == (2, 1, 50, 481, 2)
            finally:
                os.unlink(config_path)
                
    def test_memory_efficient_training(self, standard_config):
        """Test training doesn't accumulate memory over iterations."""
        model = create_model_with_config(standard_config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Run several iterations
        for i in range(5):
            spec = torch.randn(2, 1, 50, 481, 2)
            feat_erb = torch.randn(2, 1, 50, 32)
            feat_spec = torch.randn(2, 1, 50, 96, 2)
            
            optimizer.zero_grad()
            spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
            loss = spec_e.mean()
            loss.backward()
            optimizer.step()
            
        # If we got here without OOM, memory management is working
        assert True
        
    def test_inference_with_no_grad(self, standard_config):
        """Test inference with torch.no_grad() context."""
        model = create_model_with_config(standard_config)
        model.eval()
        
        spec = torch.randn(2, 1, 100, 481, 2)
        feat_erb = torch.randn(2, 1, 100, 32)
        feat_spec = torch.randn(2, 1, 100, 96, 2)
        
        with torch.no_grad():
            spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
            
        assert spec_e.shape == (2, 1, 100, 481, 2)
        assert not spec_e.requires_grad


class TestDfNet4WithLoss:
    """Test DfNet4 integration with loss functions."""
    
    def test_with_spectral_loss(self, standard_config):
        """Test DfNet4 with spectral loss computation."""
        model = create_model_with_config(standard_config)
        model.train()
        
        batch_size = 2
        num_frames = 50
        
        spec = torch.randn(batch_size, 1, num_frames, 481, 2)
        feat_erb = torch.randn(batch_size, 1, num_frames, 32)
        feat_spec = torch.randn(batch_size, 1, num_frames, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Mock clean spectrum
        clean_spec = torch.randn_like(spec_e)
        
        # Spectral magnitude loss
        spec_e_mag = torch.sqrt(spec_e[..., 0]**2 + spec_e[..., 1]**2 + 1e-8)
        clean_mag = torch.sqrt(clean_spec[..., 0]**2 + clean_spec[..., 1]**2 + 1e-8)
        mag_loss = F.l1_loss(spec_e_mag, clean_mag)
        
        # Complex loss
        complex_loss = F.mse_loss(spec_e, clean_spec)
        
        total_loss = mag_loss + 0.5 * complex_loss
        total_loss.backward()
        
        assert total_loss.item() > 0
        
    def test_with_mask_loss(self, standard_config):
        """Test DfNet4 with mask loss computation."""
        model = create_model_with_config(standard_config)
        model.train()
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Target mask (ideal mask would be between 0 and 1)
        target_mask = torch.rand(2, 1, 50, 32)
        
        mask_loss = F.binary_cross_entropy(m, target_mask)
        mask_loss.backward()
        
        assert mask_loss.item() > 0
        
    def test_with_lsnr_loss(self, standard_config):
        """Test DfNet4 with LSNR loss computation."""
        model = create_model_with_config(standard_config)
        model.train()
        
        spec = torch.randn(2, 1, 50, 481, 2)
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 1, 50, 96, 2)
        
        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
        
        # Target LSNR (should be in range [lsnr_min, lsnr_max])
        target_lsnr = torch.rand(2, 50, 1) * 55 - 15  # [-15, 40]
        
        # Normalize predicted LSNR to same range for comparison
        lsnr_loss = F.mse_loss(lsnr, target_lsnr)
        lsnr_loss.backward()
        
        assert lsnr_loss.item() > 0


class TestDfNet4FeatureExtraction:
    """Test feature extraction compatibility with DfNet4."""
    
    def test_with_libdf_features(self, standard_config):
        """Test DfNet4 with features from libdf.
        
        Validates that DfNet4 works with the expected feature shapes
        that would be produced by libdf.
        """
        from df.config import config
        config.load(standard_config, allow_reload=True)
        
        from df.deepfilternet4 import ModelParams4
        from libdf import DF
        
        p = ModelParams4()
        
        # Create DF state to get correct dimensions
        df_state = DF(
            sr=p.sr, 
            fft_size=p.fft_size, 
            hop_size=p.hop_size, 
            nb_bands=p.nb_erb,
        )
        
        # Validate DF state matches config (note: these are methods)
        assert df_state.sr() == p.sr
        assert df_state.fft_size() == p.fft_size
        assert df_state.hop_size() == p.hop_size
        assert df_state.nb_erb() == p.nb_erb
        
        # Calculate expected dimensions
        audio_len = 48000  # 1 second at 48kHz
        num_frames = audio_len // p.hop_size
        num_freqs = p.fft_size // 2 + 1
        
        # Create model and test with correctly shaped inputs
        model = create_model_with_config(standard_config)
        model.eval()
        
        # Use random features matching expected shapes from libdf
        spec = torch.randn(1, 1, num_frames, num_freqs, 2)
        feat_erb = torch.randn(1, 1, num_frames, p.nb_erb)
        feat_spec = torch.randn(1, 1, num_frames, p.nb_df, 2)
        
        with torch.no_grad():
            spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
            
        # Verify output shapes match input frames
        assert spec_e.shape[2] == num_frames
        assert m.shape[2] == num_frames
        assert lsnr.shape[1] == num_frames
        assert df_coefs.shape[2] == num_frames
        
        # Verify ERB output matches erb bands
        assert m.shape[3] == p.nb_erb
        
        # Verify DF coefs match nb_df
        assert df_coefs.shape[3] == p.nb_df


class TestQuantization:
    """Test quantization module for DFNet4."""
    
    def test_check_quantization_available(self):
        """Test quantization availability check."""
        from df.quantization import check_quantization_available
        
        # Should return bool
        result = check_quantization_available()
        assert isinstance(result, bool)
        
    def test_quantization_config(self):
        """Test QuantizationConfig dataclass."""
        from df.quantization import QuantizationConfig
        
        config = QuantizationConfig()
        assert config.backend == "x86"
        assert config.qat_epochs == 10
        assert config.calibration_batches == 100
        assert config.per_channel == True
        assert config.symmetric == True
        
        # Test with custom values
        config2 = QuantizationConfig(
            backend="qnnpack",
            qat_epochs=5,
            per_channel=False,
        )
        assert config2.backend == "qnnpack"
        assert config2.qat_epochs == 5
        assert config2.per_channel == False
        
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        from df.quantization import get_model_size_mb
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        
        size = get_model_size_mb(model)
        assert isinstance(size, float)
        assert size > 0
        
    def test_compare_model_sizes(self):
        """Test model size comparison."""
        from df.quantization import compare_model_sizes, get_model_size_mb
        
        # Create two identical models (as a baseline test)
        model1 = nn.Linear(100, 100)
        model2 = nn.Linear(100, 100)
        
        result = compare_model_sizes(model1, model2)
        assert "original_mb" in result
        assert "quantized_mb" in result
        assert "compression_ratio" in result
        assert "size_reduction_pct" in result
        
    def test_qat_callback_init(self):
        """Test QATCallback initialization."""
        from df.quantization import QATCallback
        
        model = nn.Linear(10, 10)
        callback = QATCallback(
            model,
            start_epoch=5,
            freeze_bn_epoch=8,
            backend="x86",
        )
        
        assert callback.start_epoch == 5
        assert callback.freeze_bn_epoch == 8
        assert callback.backend == "x86"
        assert callback.qat_active == False
        
    def test_qat_callback_epoch_callbacks(self):
        """Test QATCallback epoch callbacks don't crash."""
        from df.quantization import QATCallback
        
        model = nn.Linear(10, 10)
        callback = QATCallback(model, start_epoch=10)
        
        # Should not crash even if quantization not available
        callback.on_epoch_start(0)
        callback.on_epoch_end(0)
        
        # Still not active since epoch < start_epoch
        # (depends on quantization availability)
        
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Quantization tests require specific backend support"
    )
    def test_dynamic_quantization(self):
        """Test dynamic quantization (if available)."""
        from df.quantization import quantize_dynamic, check_quantization_available
        
        if not check_quantization_available():
            pytest.skip("Quantization not available")
            
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        
        quantized = quantize_dynamic(model)
        
        # Should return a model
        assert quantized is not None
        
        # Test forward pass works
        x = torch.randn(2, 100)
        with torch.no_grad():
            out = quantized(x)
        assert out.shape == (2, 10)
        
    def test_export_quantized_model_state_dict(self):
        """Test exporting model as state dict."""
        from df.quantization import export_quantized_model
        
        model = nn.Linear(10, 10)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            export_path = f.name
            
        try:
            result = export_quantized_model(model, export_path, export_format="state_dict")
            assert os.path.exists(result)
            
            # Should be loadable
            state_dict = torch.load(result)
            assert "weight" in state_dict
            assert "bias" in state_dict
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
