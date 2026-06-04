"""Mixture of Experts layers for the decoder down-path.

Provides three components:
- **TopProbGate**: Routing gate with soft / top_prob / top_k modes.
- **MoEConvLayer**: MoE wrapper around SwiGLUConv experts (BCHW format).
- **MoEDiTLayer**: MoE wrapper around CrossAttnAdaLNSwinBlock experts (BHWC format).

Each layer optionally includes a *zero expert* — a virtual expert whose output
is always zeros, contributing nothing to the weighted sum.  This acts as an
implicit "no-op" that the gate can learn to route to when the residual path
alone is sufficient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.models.conv_encoder import SwiGLUConv
from flowmatch_vae.models.swin import CrossAttnAdaLNSwinBlock


class TopProbGate(nn.Module):
    """Routing gate with three routing modes.

    Computes per-sample expert weights from spatial features + time embedding.

    Args:
        feature_dim: Channel dimension C of the spatial features.
        gate_hidden_dim: Hidden dimension of the gate MLP.
        num_experts: Number of *real* experts (excluding zero expert).
        include_zero_expert: If True, add a virtual zero expert.
        routing_mode: "soft" | "top_prob" | "top_k".
        prob_threshold: Threshold for top_prob routing.
        top_k: Number of experts for top_k routing.
    """

    def __init__(
        self,
        feature_dim: int,
        gate_hidden_dim: int = 64,
        num_experts: int = 4,
        include_zero_expert: bool = True,
        routing_mode: str = "soft",
        prob_threshold: float = 0.05,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_real_experts = num_experts
        self.include_zero_expert = include_zero_expert
        self.num_experts = num_experts + (1 if include_zero_expert else 0)
        self.routing_mode = routing_mode
        self.prob_threshold = prob_threshold
        self.top_k = top_k

        # MLP: concat(globally-pooled features, time_emb) -> weights
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, num_experts),  # only real experts
        )
        # Zeros init on last layer for near-uniform initial routing
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, pooled: torch.Tensor, time_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute routing weights.

        Args:
            pooled: (B, C) globally-pooled feature vector.
            time_emb: (B, C) time embedding.

        Returns:
            weights: (B, num_experts) routing weights (sum to 1 per sample).
            mask: (B, num_experts) boolean mask (True = selected), or None for
                  soft routing.
        """

        # Concat features + time -> MLP -> logits for real experts
        gate_input = torch.cat([pooled, time_emb], dim=-1)  # (B, 2C)
        logits = self.mlp(gate_input)  # (B, num_real_experts)

        # Add zero expert logit (always 0 -> uniform with others after zeros init)
        if self.include_zero_expert:
            zero_logit = torch.zeros(
                logits.shape[0], 1, device=logits.device, dtype=logits.dtype
            )
            logits = torch.cat([logits, zero_logit], dim=-1)  # (B, num_experts)

        if self.routing_mode == "soft":
            weights = F.softmax(logits, dim=-1)
            return weights, None

        elif self.routing_mode == "top_prob":
            probs = F.softmax(logits, dim=-1)
            mask = probs > self.prob_threshold

            # Fallback: if no expert exceeds threshold, pick argmax
            any_selected = mask.any(dim=-1)  # (B,)
            if not any_selected.all():
                argmax = probs.argmax(dim=-1)  # (B,)
                fallback_mask = torch.zeros_like(mask)
                fallback_mask.scatter_(1, argmax.unsqueeze(1), True)
                mask = torch.where(
                    any_selected.unsqueeze(1), mask, fallback_mask
                )

            # Renormalize masked probabilities
            masked_probs = probs * mask.float()
            weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            return weights, mask

        elif self.routing_mode == "top_k":
            k = min(self.top_k, self.num_experts)
            probs = F.softmax(logits, dim=-1)
            _, top_indices = probs.topk(k, dim=-1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
            masked_probs = probs * mask.float()
            weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            return weights, mask

        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")


class MoEConvLayer(nn.Module):
    """Mixture-of-Experts convolution layer.

    Each expert is a SwiGLUConv. The last expert index is the zero expert
    (output = zeros, no learnable parameters).

    Input/output: (B, C, H, W).

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Spatial kernel size.
        dilation: Dilation rate.
        num_experts: Number of real experts.
        include_zero_expert: If True, add a virtual zero expert.
        gate_hidden_dim: Hidden dimension for the routing gate.
        routing_mode: "soft" | "top_prob" | "top_k".
        prob_threshold: Threshold for top_prob routing.
        top_k: k for top_k routing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        num_experts: int = 4,
        include_zero_expert: bool = True,
        gate_hidden_dim: int = 64,
        routing_mode: str = "soft",
        prob_threshold: float = 0.05,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_real_experts = num_experts
        self.include_zero_expert = include_zero_expert
        self.num_experts = num_experts + (1 if include_zero_expert else 0)
        self.zero_expert_idx = num_experts if include_zero_expert else -1

        # Real experts
        self.experts = nn.ModuleList([
            SwiGLUConv(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
            for _ in range(num_experts)
        ])

        # Gate
        self.gate = TopProbGate(
            feature_dim=in_channels,
            gate_hidden_dim=gate_hidden_dim,
            num_experts=num_experts,
            include_zero_expert=include_zero_expert,
            routing_mode=routing_mode,
            prob_threshold=prob_threshold,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), time_emb: (B, C) -> (B, C, H, W)."""
        B, C, H, W = x.shape
        pooled = x.mean(dim=[2, 3])  # (B, C)
        weights, mask = self.gate(pooled, time_emb)  # (B, num_experts), mask or None

        output = torch.zeros_like(x)

        for i in range(self.num_real_experts):
            # Skip expert if sparse routing and no sample needs it
            if mask is not None and not mask[:, i].any():
                continue

            expert_out = self.experts[i](x)  # (B, C, H, W)
            w = weights[:, i].view(B, 1, 1, 1)  # broadcast

            if mask is not None:
                # Zero out contribution for samples that don't select this expert
                sample_mask = mask[:, i].view(B, 1, 1, 1).float()
                output = output + w * expert_out * sample_mask
            else:
                output = output + w * expert_out

        # Zero expert contributes nothing (output=zeros), so we skip it.
        # Its weight is implicitly absorbed by the softmax normalization.

        return output


class MoEDiTLayer(nn.Module):
    """Mixture-of-Experts DiT block layer.

    Each expert is a CrossAttnAdaLNSwinBlock. The last expert index is the
    zero expert (output = zeros, no learnable parameters).

    Input/output: (B, H, W, C).

    Args:
        dim: Token dimension C.
        num_heads: Number of attention heads.
        window_size: Window size for self-attention.
        shift_size: Shift size for windowed attention.
        num_experts: Number of real experts.
        include_zero_expert: If True, add a virtual zero expert.
        gate_hidden_dim: Hidden dimension for the routing gate.
        routing_mode: "soft" | "top_prob" | "top_k".
        prob_threshold: Threshold for top_prob routing.
        top_k: k for top_k routing.
        z_dim: Dimension of cross-attention source tokens.
        mhc_expansion: mHC stream count.
        mhc_sinkhorn_iters: mHC Sinkhorn iterations.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        num_experts: int = 4,
        include_zero_expert: bool = True,
        gate_hidden_dim: int = 64,
        routing_mode: str = "soft",
        prob_threshold: float = 0.05,
        top_k: int = 2,
        z_dim: int | None = None,
        mhc_expansion: int = 4,
        mhc_sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.num_real_experts = num_experts
        self.include_zero_expert = include_zero_expert
        self.num_experts = num_experts + (1 if include_zero_expert else 0)
        self.zero_expert_idx = num_experts if include_zero_expert else -1
        self.dim = dim

        # Real experts
        self.experts = nn.ModuleList([
            CrossAttnAdaLNSwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                z_dim=z_dim,
                mhc_expansion=mhc_expansion,
                mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            )
            for _ in range(num_experts)
        ])

        # Gate (features in BHWC -> pooled via mean over H,W)
        self.gate = TopProbGate(
            feature_dim=dim,
            gate_hidden_dim=gate_hidden_dim,
            num_experts=num_experts,
            include_zero_expert=include_zero_expert,
            routing_mode=routing_mode,
            prob_threshold=prob_threshold,
            top_k=top_k,
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, z_tokens: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, H, W, C), cond: (B, C), z_tokens: (B, N_z, C) -> (B, H, W, C)."""
        B, H, W, C = x.shape
        pooled = x.mean(dim=[1, 2])  # (B, C)
        weights, mask = self.gate(pooled, cond)  # (B, num_experts), mask or None

        output = torch.zeros_like(x)

        for i in range(self.num_real_experts):
            # Skip expert if sparse routing and no sample needs it
            if mask is not None and not mask[:, i].any():
                continue

            expert_out = self.experts[i](x, cond, z_tokens)  # (B, H, W, C)
            w = weights[:, i].view(B, 1, 1, 1)  # broadcast over H, W, C

            if mask is not None:
                sample_mask = mask[:, i].view(B, 1, 1, 1).float()
                output = output + w * expert_out * sample_mask
            else:
                output = output + w * expert_out

        # Zero expert contributes nothing

        return output
