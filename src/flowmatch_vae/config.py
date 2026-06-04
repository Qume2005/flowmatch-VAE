from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """Swin Transformer encoder configuration."""
    in_channels: int = 3
    patch_size: int = 4
    embed_dim: int = 128
    depths: tuple[int, ...] = (2, 6)
    num_heads: tuple[int, ...] = (4, 8)
    window_size: int = 4
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


@dataclass
class MultiScaleEncoderConfig:
    """Multi-scale SwiGLU convolution encoder configuration.

    Progressive 2x2 pooling from 64x64 down to 1x1 through 6 stages.
    Each stage has SwiGLUConv layers followed by AttnPool2x2.
    All scale features are fused via self-attention with 2D RoPE.
    """
    in_channels: int = 3
    embed_dim: int = 256
    # Number of SwiGLU conv layers per stage (6 stages: 64->32->16->8->4->2->1)
    layers_per_stage: tuple[int, ...] = (3, 2, 2, 2, 1, 1)
    # Dilation rates per stage (cycles through if more layers than rates)
    dilations_per_stage: tuple[tuple[int, ...], ...] = (
        (1, 2, 3, 4),  # 64x64: multi-scale dilated conv
        (1,),           # 32x32
        (1,),           # 16x16
        (1,),           # 8x8
        (1,),           # 4x4
        (1,),           # 2x2
    )
    # Kernel size per stage (3 for spatial conv, 1 for pointwise)
    kernel_sizes_per_stage: tuple[int, ...] = (3, 3, 3, 3, 1, 1)
    # Number of attention heads for fusion
    fusion_heads: int = 8
    # Which scale index to extract latent from (2 = 8x8, the third pool output)
    latent_scale_idx: int = 2


@dataclass
class DecoderConfig:
    """OT-CFM velocity network configuration."""
    out_channels: int = 3
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 12
    num_heads: int = 8
    window_size: int = 4
    mlp_ratio: float = 4.0
    latent_spatial_size: int = 8
    latent_dim: int = 256


@dataclass
class MultiScaleDecoderConfig:
    """U-Net multi-scale OT-CFM velocity network configuration.

    Decoder operates at 5 levels (16x16 down to 1x1) with DiT blocks
    at each level doing cross-attention to VAE encoder features.
    """
    out_channels: int = 3
    patch_size: int = 4
    embed_dim: int = 256
    # Blocks per level in down path (5 levels: 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1)
    blocks_down: tuple[int, ...] = (2, 2, 1, 1, 1)
    # Blocks per level in up path (4 levels: 2x2 -> 4x4 -> 8x8 -> 16x16)
    blocks_up: tuple[int, ...] = (1, 1, 2, 2)
    num_heads: int = 8
    window_size: int = 4
    latent_spatial_size: int = 8
    latent_dim: int = 256


@dataclass
class mHCConfig:
    """Manifold-constrained Hyper-Connections configuration.

    Based on "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880).
    """
    expansion_rate: int = 4          # n — number of parallel residual streams
    sinkhorn_iters: int = 20         # Sinkhorn-Knopp iterations for doubly stochastic projection
    gating_init: float = 0.01        # Initial value for learnable gating alphas


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 512
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.01
    kl_weight: float = 0.001
    kl_warmup_epochs: int = 50
    num_sample_steps: int = 8
    image_size: int = 64
    data_path: str = "./data"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    sample_interval: int = 5
    save_interval: int = 20
    prior_weight: float = 0.1


@dataclass
class Config:
    encoder: MultiScaleEncoderConfig = field(default_factory=MultiScaleEncoderConfig)
    decoder: MultiScaleDecoderConfig = field(default_factory=MultiScaleDecoderConfig)
    mhc: mHCConfig = field(default_factory=mHCConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
