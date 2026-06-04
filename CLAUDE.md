# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variational Autoencoder with a **Swin Transformer encoder** and **OT-CFM (Optimal Transport Conditional Flow Matching) decoder** for 64×64 image generation on CelebA. The latent space is spatial: `z ∈ (B, 8, 8, 256)` — a 2D feature map preserving spatial structure.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_vae.py -v

# Single-GPU training
python -m flowmatch_vae.train

# Distributed training (8 GPUs, Ray + PyTorch DDP)
python -m flowmatch_vae.train_dist

# Sample from prior
python -m flowmatch_vae.sample checkpoints/checkpoint_epoch200.pt --mode sample --num-samples 16

# Reconstruct an image
python -m flowmatch_vae.sample checkpoints/checkpoint_epoch200.pt --mode reconstruct --image test.jpg
```

## Architecture

### Data Flow

```
Image (B,3,64,64)
  → PatchEmbed(4×4) → (B,16,16,128) → Stage1[2×SwinBlock] → PatchMerge → (B,8,8,256)
  → Stage2[6×SwinBlock] → mu,logvar (B,8,8,256)
  → reparameterize → z (B,8,8,256)
  → noise x0 ~ N(0,1), sample t ~ U(0,1), x_t = (1-t)*x0 + t*x
  → PatchEmbed(4×4) + nearest-upsample z → 12×CrossAttnAdaLNSwinBlock(x_t, z, t_emb)
  → LayerNorm → Linear → pixel_shuffle → v_pred (B,3,64,64)
  → FM loss: MSE(v_pred, x-x0) + KL divergence loss
```

### Key Design Decisions

- **OT-CFM loss**: Straight-line interpolation between noise and data; velocity target is `x1 - x0`. Generation uses 8-step Euler integration from noise.
- **Cross-attention z injection**: Decoder uses dedicated cross-attention layers (x tokens attend to z tokens) rather than simple addition.
- **AdaLN conditioning**: Time `t` is injected via adaptive LayerNorm (DiT-style) — the time embedding predicts scale/shift/gate parameters for each sub-layer.
- **Pixel shuffle output**: Decoder reconstructs via `F.pixel_shuffle` rather than transposed convolutions.

### Source Layout (`src/flowmatch_vae/`)

| File | Role |
|------|------|
| `config.py` | Three dataclass configs (`EncoderConfig`, `DecoderConfig`, `TrainConfig`) composed into `Config` |
| `models/swin.py` | Core Swin building blocks — `SwinBlock` (encoder), `AdaLNSwinBlock`, `CrossAttnAdaLNSwinBlock` (decoder), `WindowAttention`, `PatchEmbed`, `PatchMerge` |
| `models/encoder.py` | `SwinEncoder` — two-stage Swin-T, outputs `mu` and `logvar` |
| `models/decoder.py` | `FlowDecoder` — OT-CFM velocity network with sinusoidal time embedding + 12 cross-attention blocks |
| `models/vae.py` | `FlowMatchVAE` — combines encoder/decoder, implements `compute_loss()`, `sample()`, `reconstruct()` |
| `data/celeba.py` | CelebA loading with in-memory caching (`cache_dataset()` pre-loads into `TensorDataset` for zero-IO training) |
| `train.py` | Single-GPU training loop |
| `train_dist.py` | 8-GPU distributed training via Ray Actors + PyTorch native DDP (NCCL) |

### Swin Block Variants (in `swin.py`)

- **`SwinBlock`**: Standard W-MSA / SW-MSA for the encoder. Alternating shifted windows.
- **`AdaLNSwinBlock`**: DiT-style adaptive LayerNorm modulation. Condition vector → 6 params (scale/shift × 2 sub-layers + 2 gates). Zero-initialized output projection.
- **`CrossAttnAdaLNSwinBlock`**: Decoder block. Self-attention (windowed) + cross-attention (full, x attends to z) + FFN, all AdaLN-conditioned on time. Condition → 9 params (3 groups of scale/shift/gate).

## Distributed Training

`train_dist.py` uses Ray for worker orchestration and PyTorch DDP for gradient sync. Key details:
- Each `TrainingWorker` is a Ray actor with 1 GPU, running DDP on NCCL backend.
- The full dataset is cached in memory via `cache_dataset()` before training starts.
- Ctrl+C triggers graceful shutdown: writes `/tmp/flowmatch_vae_stop`, workers checkpoint and run a sampling benchmark before exiting.

## Checkpoint Format

Checkpoints store: `model_state_dict`, `optimizer_state_dict`, `config` (dataclass), `epoch`. When loading, the code handles both dataclass and raw-dict config formats for `weights_only=True` compatibility.
