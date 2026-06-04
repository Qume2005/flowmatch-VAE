# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variational Autoencoder with a **Swin Transformer encoder** and **OT-CFM (Optimal Transport Conditional Flow Matching) decoder** for 64×64 image generation on CelebA. The latent space is spatial: `z ∈ (B, 8, 8, 256)` — a 2D feature map preserving spatial structure.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run all tests (ROS plugins interfere — disable autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v

# Run a single test file
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py -v

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
  → RMSNorm → Linear → pixel_shuffle → v_pred (B,3,64,64)
  → FM loss: MSE(v_pred, x-x0) + KL divergence loss
```

### Key Design Decisions

- **OT-CFM loss**: Straight-line interpolation between noise and data; velocity target is `x1 - x0`. Generation uses 8-step Euler integration from noise.
- **Kimi Linear Attention (KDA)**: Bidirectional linear attention replacing softmax. L2-normalised Q/K, channel-wise forget gate (alpha), delta-rule learning rate (beta), depthwise conv on Q/K/V, sigmoid output gating. O(n·d²) instead of O(n²·d). See arXiv:2510.26692.
- **mHC residual connections**: Expanded n-stream residual (n=4) with Sinkhorn-Knopp doubly-stochastic H_res, sigmoid-constrained H_pre/H_post. The stream expands C→4C internally, contracted back by averaging. See arXiv:2512.24880.
- **SwiGLU FFN**: `w_down(SiLU(w_gate(x)) * w_up(x))` with hidden dim rounded to 256-multiples.
- **RMSNorm everywhere**: No LayerNorm, no BatchNorm. AdaLN blocks use `elementwise_affine=False` since scale/shift come from the conditioning MLP.
- **AdaLN conditioning**: Time `t` injected via adaptive RMSNorm modulation (DiT-style) — the time embedding predicts scale/shift/gate parameters for each sub-layer.
- **Pixel shuffle output**: Decoder reconstructs via `F.pixel_shuffle` rather than transposed convolutions.
- **Muon + SGD optimizer**: Muon (Newton-Schulz orthogonalised momentum) for 2D weight matrices, SGD (momentum=0.9) for biases/norms/embeddings. See arXiv:2502.16982.

### Source Layout (`src/flowmatch_vae/`)

| File | Role |
|------|------|
| `config.py` | Dataclass configs: `EncoderConfig`, `DecoderConfig`, `mHCConfig`, `TrainConfig` composed into `Config` |
| `models/swin.py` | Core blocks — `RMSNorm`, `KimiLinearAttention`, `KimiLinearCrossAttention`, `SwiGLUFFN`, `mHCConnection` (read/write API), `SwinBlock`, `AdaLNSwinBlock`, `CrossAttnAdaLNSwinBlock`, `PatchEmbed`, `PatchMerge` |
| `models/encoder.py` | `SwinEncoder` — two-stage Swin-T, outputs `mu` and `logvar`. Accepts `mhc_cfg` kwarg. |
| `models/decoder.py` | `FlowDecoder` — OT-CFM velocity network with sinusoidal time embedding + 12 cross-attention blocks. Accepts `mhc_cfg` kwarg. |
| `models/vae.py` | `FlowMatchVAE` — combines encoder/decoder, implements `compute_loss()`, `sample()`, `reconstruct()` |
| `optimizers.py` | `Muon` optimizer (Newton-Schulz, Nesterov momentum, bf16 iteration) + `split_param_groups()` helper |
| `data/celeba.py` | CelebA loading with in-memory caching (`cache_dataset()` pre-loads into `TensorDataset` for zero-IO training) |
| `train.py` | Single-GPU training loop with Muon + SGD |
| `train_dist.py` | 8-GPU distributed training via Ray Actors + PyTorch native DDP (NCCL) |

### Block Variants (in `swin.py`)

- **`SwinBlock`**: Encoder block. KimiLinear windowed attn + SwiGLU FFN, both with mHC residual connections. Expands stream to nC internally, contracts back to C.
- **`AdaLNSwinBlock`**: Decoder block with AdaLN modulation. Condition → 6 params (s1,sh1,g1, s2,sh2,g2). RMSNorm with `elementwise_affine=False`.
- **`CrossAttnAdaLNSwinBlock`**: Full decoder block. Self-attn (windowed KimiLinear) + cross-attn (KimiLinearCrossAttention, x→z) + SwiGLU FFN. Condition → 9 params (3× scale/shift/gate). Three independent mHC connections.

### mHC Connection API

`mHCConnection` uses a two-phase read/write pattern:
1. `layer_input, ctx = mhc.read(x_stream)` — reads from nC-stream, returns C-dim layer input + context dict
2. Run sublayer on `layer_input`
3. `x_stream = mhc.write(x_stream, sublayer_out, ctx)` — updates the nC-stream

Internal: mappings (H_pre/H_res/H_post) are computed once from RMSNorm'd stream in `read()` and reused in `write()`.

## Distributed Training

`train_dist.py` uses Ray for worker orchestration and PyTorch DDP for gradient sync. Key details:
- Each `TrainingWorker` is a Ray actor with 1 GPU, running DDP on NCCL backend.
- The full dataset is cached in memory via `cache_dataset()` before training starts.
- Ctrl+C triggers graceful shutdown: writes `/tmp/flowmatch_vae_stop`, workers checkpoint and run a sampling benchmark before exiting.

## Checkpoint Format

Checkpoints store: `model_state_dict`, `muon_state_dict`, `sgd_state_dict`, `config` (dataclass), `epoch`. When loading, the code handles both dataclass and raw-dict config formats for `weights_only=True` compatibility.
