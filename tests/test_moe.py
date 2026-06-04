"""Tests for Mixture of Experts layers."""

import pytest
import torch
import torch.nn as nn

from flowmatch_vae.config import Config, MultiScaleEncoderConfig, MultiScaleDecoderConfig, MoEConfig
from flowmatch_vae.models.moe import TopProbGate, MoEConvLayer, MoEDiTLayer
from flowmatch_vae.models.decoder import FlowDecoder
from flowmatch_vae.models.vae import FlowMatchVAE


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

def test_gate_soft_routing_sums_to_one():
    """Soft routing: weights should sum to 1 per sample."""
    gate = TopProbGate(
        feature_dim=64, gate_hidden_dim=32, num_experts=4,
        include_zero_expert=True, routing_mode="soft",
    )
    pooled = torch.randn(3, 64)   # already pooled (B, C)
    time_emb = torch.randn(3, 64)
    weights, mask = gate(pooled, time_emb)
    assert weights.shape == (3, 5)  # 4 real + 1 zero
    assert mask is None
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5


def test_gate_top_prob_threshold():
    """top_prob routing: only experts above threshold are selected."""
    gate = TopProbGate(
        feature_dim=64, gate_hidden_dim=32, num_experts=4,
        include_zero_expert=False, routing_mode="top_prob",
        prob_threshold=0.05,
    )
    pooled = torch.randn(5, 64)
    time_emb = torch.randn(5, 64)
    weights, mask = gate(pooled, time_emb)
    assert weights.shape == (5, 4)
    assert mask is not None
    assert mask.shape == (5, 4)
    # Weights for unmasked experts should be > 0
    assert (weights[mask] > 0).all()
    # Weights for masked-out experts should be 0
    assert (weights[~mask] == 0).all()
    # Should still sum to 1
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5


def test_gate_top_k_routing():
    """top_k routing: exactly k experts should be selected."""
    gate = TopProbGate(
        feature_dim=32, gate_hidden_dim=16, num_experts=4,
        include_zero_expert=False, routing_mode="top_k", top_k=2,
    )
    pooled = torch.randn(4, 32)
    time_emb = torch.randn(4, 32)
    weights, mask = gate(pooled, time_emb)
    assert weights.shape == (4, 4)
    assert mask is not None
    # Exactly 2 experts selected per sample
    assert (mask.sum(dim=-1) == 2).all()
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# MoEConvLayer tests
# ---------------------------------------------------------------------------

def test_moe_conv_output_shape():
    """MoE conv layer preserves spatial shape."""
    moe = MoEConvLayer(
        in_channels=64, out_channels=64, kernel_size=3, dilation=1,
        num_experts=4, include_zero_expert=True,
        routing_mode="soft",
    )
    x = torch.randn(2, 64, 8, 8)
    t = torch.randn(2, 64)
    out = moe(x, t)
    assert out.shape == (2, 64, 8, 8)


def test_moe_conv_gradient_flows():
    """Gradients flow through MoE conv layer."""
    moe = MoEConvLayer(
        in_channels=32, out_channels=32, kernel_size=3, dilation=1,
        num_experts=2, include_zero_expert=False,
        routing_mode="soft",
    )
    x = torch.randn(1, 32, 4, 4, requires_grad=True)
    t = torch.randn(1, 32)
    out = moe(x, t)
    out.sum().backward()
    assert x.grad is not None


def test_moe_conv_sparse_routing():
    """MoE conv with sparse routing still produces correct output."""
    moe = MoEConvLayer(
        in_channels=32, out_channels=32, kernel_size=3, dilation=1,
        num_experts=2, include_zero_expert=True,
        routing_mode="top_k", top_k=1,
    )
    x = torch.randn(3, 32, 4, 4)
    t = torch.randn(3, 32)
    out = moe(x, t)
    assert out.shape == (3, 32, 4, 4)


# ---------------------------------------------------------------------------
# MoEDiTLayer tests (require CUDA due to mHC Sinkhorn)
# ---------------------------------------------------------------------------

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@cuda_required
def test_moe_dit_output_shape():
    """MoE DiT layer preserves spatial shape."""
    moe = MoEDiTLayer(
        dim=64, num_heads=4, window_size=4, shift_size=0,
        num_experts=2, include_zero_expert=True,
        routing_mode="soft", z_dim=64,
        mhc_expansion=2, mhc_sinkhorn_iters=5,
    ).to("cuda")
    x = torch.randn(2, 8, 8, 64, device="cuda")
    cond = torch.randn(2, 64, device="cuda")
    z_tokens = torch.randn(2, 16, 64, device="cuda")
    out = moe(x, cond, z_tokens)
    assert out.shape == (2, 8, 8, 64)


def test_zero_expert_contributes_zero():
    """When gate routes only to zero expert, output should be near-zero."""
    moe = MoEConvLayer(
        in_channels=32, out_channels=32, kernel_size=3, dilation=1,
        num_experts=2, include_zero_expert=True,
        routing_mode="soft",
    )
    x = torch.randn(1, 32, 4, 4)
    t = torch.randn(1, 32)

    # Override gate to route entirely to zero expert (last index = 2)
    with torch.no_grad():
        # Set gate MLP last layer to produce very large negative logits
        # for real experts, so softmax favors the zero expert (logit=0).
        moe.gate.mlp[-1].weight.zero_()
        moe.gate.mlp[-1].bias.fill_(-10.0)  # all real experts get -10 logit

    out = moe(x, t)
    # With all real experts at -10 logit and zero expert at 0, softmax heavily
    # favors zero expert whose output is zeros. Output should be very close to 0.
    assert out.abs().max().item() < 0.1, (
        f"Output should be near-zero when routing to zero expert, got max abs {out.abs().max().item()}"
    )


@cuda_required
def test_moe_dit_gradient_flows():
    """Gradients flow through MoE DiT layer."""
    moe = MoEDiTLayer(
        dim=64, num_heads=4, window_size=4, shift_size=0,
        num_experts=2, include_zero_expert=False,
        routing_mode="soft", z_dim=64,
        mhc_expansion=2, mhc_sinkhorn_iters=5,
    ).to("cuda")
    x = torch.randn(1, 4, 4, 64, device="cuda", requires_grad=True)
    cond = torch.randn(1, 64, device="cuda")
    z_tokens = torch.randn(1, 8, 64, device="cuda")
    out = moe(x, cond, z_tokens)
    out.sum().backward()
    assert x.grad is not None


@cuda_required
def test_moe_dit_sparse_routing():
    """MoE DiT with top_k routing."""
    moe = MoEDiTLayer(
        dim=64, num_heads=4, window_size=4, shift_size=0,
        num_experts=3, include_zero_expert=True,
        routing_mode="top_k", top_k=2, z_dim=64,
        mhc_expansion=2, mhc_sinkhorn_iters=5,
    ).to("cuda")
    x = torch.randn(2, 4, 4, 64, device="cuda")
    cond = torch.randn(2, 64, device="cuda")
    z_tokens = torch.randn(2, 8, 64, device="cuda")
    out = moe(x, cond, z_tokens)
    assert out.shape == (2, 4, 4, 64)


# ---------------------------------------------------------------------------
# Full decoder with MoE
# ---------------------------------------------------------------------------

@cuda_required
def test_moe_decoder_output_shape():
    """Full decoder with MoE config produces correct output shape."""
    moe_cfg = MoEConfig(
        num_experts=2,
        include_zero_expert=True,
        gate_hidden_dim=32,
        routing_mode="soft",
        moe_conv=True,
        moe_dit=True,
    )
    dec_cfg = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
        moe=moe_cfg,
    )
    enc_cfg = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    # Build encoder to get shared modules
    from flowmatch_vae.models.conv_encoder import MultiScaleConvEncoder
    encoder = MultiScaleConvEncoder(enc_cfg)

    dec = FlowDecoder(
        dec_cfg, mhc_cfg=None,
        vae_scale_map=[1, 2, 3, 4, 5],
        latent_vae_scale=2,
        shared_convs=encoder.conv_blocks,
        shared_pool=encoder.pool,
        encoder_dilations=enc_cfg.dilations,
    ).to("cuda")

    x_t = torch.randn(2, 3, 64, 64, device="cuda")
    t = torch.rand(2, device="cuda")
    z = torch.randn(2, 8, 8, 256, device="cuda")
    v = dec(x_t, t, z)
    assert v.shape == (2, 3, 64, 64), f"Expected (2,3,64,64), got {v.shape}"


@cuda_required
def test_moe_decoder_gradient_flows():
    """Gradients flow through full MoE decoder."""
    moe_cfg = MoEConfig(
        num_experts=2,
        include_zero_expert=False,
        gate_hidden_dim=32,
        routing_mode="soft",
        moe_conv=True,
        moe_dit=True,
    )
    dec_cfg = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
        moe=moe_cfg,
    )
    enc_cfg = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    from flowmatch_vae.models.conv_encoder import MultiScaleConvEncoder
    encoder = MultiScaleConvEncoder(enc_cfg)

    dec = FlowDecoder(
        dec_cfg, mhc_cfg=None,
        vae_scale_map=[1, 2, 3, 4, 5],
        latent_vae_scale=2,
        shared_convs=encoder.conv_blocks,
        shared_pool=encoder.pool,
        encoder_dilations=enc_cfg.dilations,
    ).to("cuda")

    x_t = torch.randn(1, 3, 64, 64, device="cuda")
    t = torch.rand(1, device="cuda")
    z = torch.randn(1, 8, 8, 256, device="cuda")
    v = dec(x_t, t, z)
    v.sum().backward()
    # down_blocks are kept for backward compat but not used when MoE is active;
    # skip them when checking gradient flow.
    active_params = [
        (name, p) for name, p in dec.named_parameters()
        if p.requires_grad and not name.startswith("down_blocks.")
    ]
    for name, p in active_params:
        assert p.grad is not None, f"No gradient for {name}"


@cuda_required
def test_moe_vae_integration():
    """VAE with MoE decoder runs end-to-end."""
    moe_cfg = MoEConfig(
        num_experts=2,
        include_zero_expert=True,
        gate_hidden_dim=32,
        routing_mode="soft",
        moe_conv=True,
        moe_dit=True,
    )
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )
    cfg.decoder = MultiScaleDecoderConfig(
        blocks_down=(1, 1, 1, 1, 1),
        blocks_up=(1, 1, 1, 1),
        moe=moe_cfg,
    )
    model = FlowMatchVAE(cfg).to("cuda")
    x = torch.randn(2, 3, 64, 64, device="cuda")
    losses = model.compute_loss(x)
    assert "loss" in losses
    assert losses["loss"].requires_grad
