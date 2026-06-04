import dataclasses

import torch
import torch.nn.functional as F
from flowmatch_vae.config import Config, MultiScaleEncoderConfig, MultiScaleDecoderConfig, MoEConfig
from flowmatch_vae.models.vae import FlowMatchVAE
from flowmatch_vae.models.conv_encoder import (
    SwiGLUConv, RMSNorm2d, AttnPool2x2, Upsample2x, MultiScalePrior, MultiScaleConvEncoder,
)
from flowmatch_vae.models.decoder import FlowDecoder


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

def test_rmsnorm2d():
    norm = RMSNorm2d(64)
    x = torch.randn(2, 64, 16, 16)
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(dim=1, keepdim=True).sqrt()
    assert (rms - 1.0).abs().max() < 0.1


def test_swiglu_conv_same_size():
    conv = SwiGLUConv(64, 64, kernel_size=3, dilation=2)
    x = torch.randn(2, 64, 16, 16)
    assert conv(x).shape == (2, 64, 16, 16)


def test_swiglu_conv_channel_change():
    conv = SwiGLUConv(3, 128, kernel_size=3, dilation=1)
    x = torch.randn(2, 3, 64, 64)
    assert conv(x).shape == (2, 128, 64, 64)


def test_swiglu_conv_pointwise():
    conv = SwiGLUConv(64, 64, kernel_size=1, dilation=1)
    x = torch.randn(2, 64, 4, 4)
    assert conv(x).shape == (2, 64, 4, 4)


def test_attn_pool_halves_spatial():
    pool = AttnPool2x2(64)
    x = torch.randn(2, 64, 16, 16)
    assert pool(x).shape == (2, 64, 8, 8)


def test_attn_pool_weights_sum_to_one():
    pool = AttnPool2x2(64)
    x = torch.randn(2, 64, 4, 4)
    B, C, H, W = x.shape
    x_bhwc = x.permute(0, 2, 3, 1)
    x_blocks = x_bhwc.view(B, H // 2, 2, W // 2, 2, C)
    x_blocks = x_blocks.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // 2, W // 2, 4, C)
    x_norm = x_blocks / (x_blocks.norm(dim=-1, keepdim=True) + 1e-6)
    P = B * (H // 2) * (W // 2) * 4
    logits = pool.alpha * pool.phi(x_norm.reshape(P, C)) + pool.phi.bias
    weights = F.softmax(logits, dim=-1)
    assert weights.shape == (P, 4)
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5


def test_attn_pool_grad_flows():
    pool = AttnPool2x2(32)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None


def test_upsample2x():
    up = Upsample2x(64)
    x = torch.randn(2, 64, 4, 4)
    out = up(x)
    assert out.shape == (2, 64, 8, 8)


def test_upsample2x_grad():
    up = Upsample2x(32)
    x = torch.randn(1, 32, 4, 4, requires_grad=True)
    out = up(x)
    out.sum().backward()
    assert x.grad is not None


def test_prior_predicts_scales():
    prior = MultiScalePrior(256, latent_spatial_size=8)
    z = torch.randn(2, 8, 8, 256)
    tokens = prior(z)
    # Default layout: scales 0(32x32), 1(16x16), 2(8x8=latent), 3(4x4), 4(2x2), 5(1x1)
    # Prior predicts all except scale 2 (latent)
    assert 0 in tokens and tokens[0].shape == (2, 1024, 256)  # 32x32
    assert 1 in tokens and tokens[1].shape == (2, 256, 256)   # 16x16
    assert 3 in tokens and tokens[3].shape == (2, 16, 256)    # 4x4
    assert 4 in tokens and tokens[4].shape == (2, 4, 256)     # 2x2
    assert 5 in tokens and tokens[5].shape == (2, 1, 256)     # 1x1
    assert 2 not in tokens  # scale 2 = z itself, not predicted


def test_encoder_returns_scale_dict():
    cfg = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    enc = MultiScaleConvEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    mu, logvar, tokens = enc(x)
    assert mu.shape == (2, 8, 8, 256)
    assert logvar.shape == (2, 8, 8, 256)
    assert isinstance(tokens, dict)
    assert 0 in tokens and tokens[0].shape == (2, 1024, 256)
    assert 2 in tokens and tokens[2].shape == (2, 64, 256)
    assert 5 in tokens and tokens[5].shape == (2, 1, 256)


def test_encoder_grad_flows():
    cfg = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    enc = MultiScaleConvEncoder(cfg)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    mu, logvar, tokens = enc(x)
    mu.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

def test_decoder_unet_shape():
    cfg = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
    )
    dec = FlowDecoder(cfg)
    x_t = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    z = torch.randn(2, 8, 8, 256)
    out = dec(x_t, t, z, scale_tokens=None)
    assert out.shape == (2, 3, 64, 64)


def test_decoder_with_scale_tokens():
    cfg = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
    )
    dec = FlowDecoder(cfg)
    x_t = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    z = torch.randn(2, 8, 8, 256)
    scale_tokens = {
        1: torch.randn(2, 256, 256),
        2: z.reshape(2, -1, 256),
        3: torch.randn(2, 16, 256),
        4: torch.randn(2, 4, 256),
        5: torch.randn(2, 1, 256),
    }
    out = dec(x_t, t, z, scale_tokens=scale_tokens)
    assert out.shape == (2, 3, 64, 64)


# ---------------------------------------------------------------------------
# VAE integration tests
# ---------------------------------------------------------------------------

def _small_config():
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    cfg.decoder = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
    )
    return cfg


def test_vae_loss_runs():
    model = FlowMatchVAE(_small_config())
    x = torch.randn(2, 3, 64, 64)
    losses = model.compute_loss(x)
    assert "loss" in losses
    assert "fm_loss" in losses
    assert "kl_loss" in losses
    assert "prior_loss" in losses
    assert losses["fm_loss"].shape == ()
    assert losses["kl_loss"].shape == ()
    assert losses["prior_loss"].shape == ()


def test_vae_sample_shape():
    model = FlowMatchVAE(_small_config())
    model.eval()
    with torch.no_grad():
        x_recon = model.sample(num_samples=2, num_steps=4, device="cpu")
    assert x_recon.shape == (2, 3, 64, 64)


def test_vae_encode_shape():
    model = FlowMatchVAE(_small_config())
    x = torch.randn(2, 3, 64, 64)
    mu, logvar, tokens = model.encode(x)
    assert mu.shape == (2, 8, 8, 256)
    assert logvar.shape == (2, 8, 8, 256)
    z = model.reparameterize(mu, logvar)
    assert z.shape == (2, 8, 8, 256)


def test_vae_reconstruct():
    model = FlowMatchVAE(_small_config())
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        x_recon = model.reconstruct(x, num_steps=4)
    assert x_recon.shape == (2, 3, 64, 64)


def test_end_to_end_overfit_single_batch():
    """Overfit on a single batch to verify the full training loop."""
    cfg = _small_config()
    model = FlowMatchVAE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    x = torch.randn(4, 3, 64, 64)
    for _ in range(800):
        losses = model.compute_loss(x)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    assert losses["fm_loss"].item() < 2.0, f"FM loss should decrease, got {losses['fm_loss'].item()}"


# ---------------------------------------------------------------------------
# Checkpoint save / load tests
# ---------------------------------------------------------------------------

def _moe_config():
    """Small config with MoE enabled (conv + DiT experts)."""
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    cfg.decoder = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
        moe=MoEConfig(num_experts=2),
    )
    return cfg


def test_checkpoint_roundtrip_no_moe():
    """Save and load a checkpoint without MoE — outputs must match."""
    cfg = _small_config()
    model = FlowMatchVAE(cfg)
    model.eval()

    state = model.state_dict()

    # Load into a fresh model
    model2 = FlowMatchVAE(cfg)
    model2.load_state_dict(state)
    model2.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        mu1, lv1, _ = model.encode(x)
        mu2, lv2, _ = model2.encode(x)

    assert torch.allclose(mu1, mu2, atol=1e-5), "mu mismatch after load_state_dict"
    assert torch.allclose(lv1, lv2, atol=1e-5), "logvar mismatch after load_state_dict"


def test_checkpoint_roundtrip_with_moe():
    """Save and load a checkpoint with MoE enabled — outputs must match."""
    cfg = _moe_config()
    model = FlowMatchVAE(cfg)
    model.eval()

    state = model.state_dict()

    model2 = FlowMatchVAE(cfg)
    model2.load_state_dict(state)
    model2.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        mu1, lv1, _ = model.encode(x)
        mu2, lv2, _ = model2.encode(x)

    assert torch.allclose(mu1, mu2, atol=1e-5), "mu mismatch after load_state_dict (MoE)"
    assert torch.allclose(lv1, lv2, atol=1e-5), "logvar mismatch after load_state_dict (MoE)"


def test_checkpoint_roundtrip_full_forward_with_moe():
    """Full forward pass (compute_loss) with MoE — outputs must match after save/load."""
    cfg = _moe_config()
    model = FlowMatchVAE(cfg)
    model.eval()

    state = model.state_dict()

    model2 = FlowMatchVAE(cfg)
    model2.load_state_dict(state)
    model2.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        # Use deterministic z for comparison (disable reparameterization randomness)
        torch.manual_seed(42)
        out1 = model(x)
        torch.manual_seed(42)
        out2 = model2(x)

    assert torch.allclose(out1["fm_loss"], out2["fm_loss"], atol=1e-5), \
        f"fm_loss mismatch: {out1['fm_loss'].item()} vs {out2['fm_loss'].item()}"
    assert torch.allclose(out1["kl_loss"], out2["kl_loss"], atol=1e-5), \
        f"kl_loss mismatch: {out1['kl_loss'].item()} vs {out2['kl_loss'].item()}"


def test_checkpoint_file_roundtrip_with_moe():
    """Full file-based checkpoint save/load cycle (matches train.py -> sample.py path)."""
    import tempfile
    from flowmatch_vae.sample import load_model

    cfg = _moe_config()
    model = FlowMatchVAE(cfg)
    model.eval()

    # Save exactly like train.py / train_dist.py
    ckpt = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        path = f.name

    # Load exactly like sample.py
    model2, cfg_loaded = load_model(path, device="cpu")

    # Verify MoE config was reconstructed correctly
    assert isinstance(cfg_loaded.decoder.moe, MoEConfig), \
        f"Expected MoEConfig, got {type(cfg_loaded.decoder.moe)}"
    assert cfg_loaded.decoder.moe.num_experts == 2
    assert cfg_loaded.decoder.moe.routing_mode == "soft"

    # Verify outputs match
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        mu1, lv1, _ = model.encode(x)
        mu2, lv2, _ = model2.encode(x)

    assert torch.allclose(mu1, mu2, atol=1e-5), "mu mismatch after file roundtrip"
    assert torch.allclose(lv1, lv2, atol=1e-5), "logvar mismatch after file roundtrip"

    import os
    os.unlink(path)


def test_shared_params_in_state_dict():
    """Shared encoder/decoder parameters should have identical values after load."""
    cfg = _small_config()
    model = FlowMatchVAE(cfg)
    model.eval()

    sd = model.state_dict()

    model2 = FlowMatchVAE(cfg)
    model2.load_state_dict(sd)

    # Verify shared conv blocks hold the same parameter object
    assert model2.encoder.conv_blocks[0] is model2.decoder.shared_convs[0]
    assert model2.encoder.conv_blocks[0] is model2.decoder.up_samples[0].refine
    # Verify shared pool
    assert model2.encoder.pool is model2.decoder.shared_pool


def test_checkpoint_preserves_mhc_config():
    """mhc config section must be correctly restored from checkpoint."""
    import tempfile
    from flowmatch_vae.sample import load_model

    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(num_conv_blocks=1, dilations=(1,))
    cfg.decoder = MultiScaleDecoderConfig(blocks_down=(1,1,1,1,1), blocks_up=(1,1,1,1))
    cfg.mhc.expansion_rate = 8
    cfg.mhc.sinkhorn_iters = 10

    ckpt = {
        "epoch": 1,
        "model_state_dict": FlowMatchVAE(cfg).state_dict(),
        "config": dataclasses.asdict(cfg),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        path = f.name

    _, cfg_loaded = load_model(path, device="cpu")
    assert cfg_loaded.mhc.expansion_rate == 8, \
        f"Expected mhc.expansion_rate=8, got {cfg_loaded.mhc.expansion_rate}"
    assert cfg_loaded.mhc.sinkhorn_iters == 10, \
        f"Expected mhc.sinkhorn_iters=10, got {cfg_loaded.mhc.sinkhorn_iters}"

    import os
    os.unlink(path)
