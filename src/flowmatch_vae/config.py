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
class DecoderConfig:
    """OT-CFM velocity network configuration."""
    out_channels: int = 3
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    window_size: int = 4
    mlp_ratio: float = 4.0
    latent_spatial_size: int = 8
    latent_dim: int = 256


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.01
    kl_weight: float = 0.1
    kl_warmup_epochs: int = 10
    num_sample_steps: int = 8
    image_size: int = 64
    data_path: str = "./data"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    sample_interval: int = 5
    save_interval: int = 20


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
