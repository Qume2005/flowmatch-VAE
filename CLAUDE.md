# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variational Autoencoder with a **multi-scale SwiGLU conv encoder** and **U-Net multi-scale OT-CFM decoder** for 64×64 image generation on CelebA. The latent space is spatial: `z ∈ (B, 8, 8, 256)` — a 2D feature map preserving spatial structure.

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
=== ENCODER ===
Image (B,3,64,64)
  -> Stem Conv1x1(3->256) + RMSNorm2d
  -> Stage0: 3x SwiGLUConv(d in {1,2,3,4}, k=3) + AttnPool2x2 -> 32x32
  -> Stage1: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 16x16
  -> Stage2: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 8x8
  -> Stage3: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 4x4
  -> Stage4: 1x SwiGLUConv(k=1) + AttnPool2x2 -> 2x2
  -> Stage5: 1x SwiGLUConv(k=1) + AttnPool2x2 -> 1x1
  -> Collect all scales (1365 tokens) + scale_embed + 2D RoPE
  -> FusionAttention -> per-scale tokens dict + mu, logvar (B,8,8,256)

=== DECODER (U-Net) ===
x_t (B,3,64,64) + z (B,8,8,256) + per_scale_tokens from encoder

Down path (AttnPool2x2 between levels):
  Level 0 (16x16): DiT_block x 2 [self-attn + cross-attn -> VAE scale 1]
  Level 1 (8x8):   DiT_block x 2 [self-attn + cross-attn -> VAE scale 2 = z]
  Level 2 (4x4):   DiT_block x 1 [self-attn + cross-attn -> VAE scale 3]
  Level 3 (2x2):   DiT_block x 1 [self-attn + cross-attn -> VAE scale 4]
  Level 4 (1x1):   DiT_block x 1 [self-attn + cross-attn -> VAE scale 5]

Up path (Upsample2x between levels, skip connections via addition):
  Level 3 (2x2):   DiT_block x 1 + skip from down
  Level 2 (4x4):   DiT_block x 1 + skip from down
  Level 1 (8x8):   DiT_block x 2 + skip from down
  Level 0 (16x16): DiT_block x 2 + skip from down

Output: RMSNorm -> Linear -> pixel_shuffle -> v_pred (B,3,64,64)

=== GENERATION ===
z ~ N(0,I) -> MultiScalePrior predicts scales {1,3,4,5} -> decoder uses predicted tokens
```

### Key Design Decisions

- **OT-CFM loss**: Straight-line interpolation between noise and data; velocity target is `x1 - x0`. Generation uses 8-step Euler integration from noise.
- **Multi-Scale SwiGLU Conv Encoder**: Progressive 2x2 attention-pooling from 64x64 to 1x1. Each stage has multi-channel SwiGLU-gated depthwise convolutions with dilation. All 6 scales fused via linear self-attention with 2D RoPE. Encoder returns per-scale tokens dict for decoder cross-attention.
- **U-Net Multi-Scale Decoder**: Down path uses AttnPool2x2, up path uses Upsample2x (nearest + SwiGLUConv). Skip connections via addition. Each level cross-attends to the VAE encoder's corresponding scale features. Goes coarse-to-fine naturally.
- **MultiScalePrior**: Predicts multi-scale features from z for generation. ConvTranspose (upsample larger scales) + Conv (pool smaller scales) + SwiGLUConv. Trained with L2 auxiliary loss. Training randomly picks encoder features or prior predictions (50/50) to bridge train/test gap.
- **AttnPool2x2**: Learned 2x2 pooling via softmax attention (inspired by mHC H_pre). alpha*phi(x_norm) + bias -> softmax -> weighted sum. Guarantees weights sum to 1.
- **1/7 kernel constraint**: Dilation rates chosen so effective RF <= sqrt(H*W)/7. At 64x64: dilations (1,2,3,4) -> RF (3,5,7,9). At smaller scales: dilation=1 or pointwise (kernel=1).
- **2D RoPE**: Split head dimension in half: first half encodes y-position, second half encodes x-position. Applied in FusionAttention for position-aware cross-scale attention.
- **z IS scale 2**: The VAE bottleneck z (8x8) is naturally one of the encoder's multi-scale levels. No separate z cross-attention needed — decoder level 1 directly queries z.
- **Kimi Linear Attention (KDA)**: Bidirectional linear attention replacing softmax. O(n·d²) complexity enables cross-attention to 1024 tokens (32x32 scale) efficiently.
- **mHC residual connections**: Expanded n-stream residual (n=4) with Sinkhorn-Knopp doubly-stochastic H_res. See arXiv:2512.24880.
- **SwiGLU FFN**: `w_down(SiLU(w_gate(x)) * w_up(x))`. SwiGLU-gated depthwise convolutions used throughout encoder and Upsample2x.
- **RMSNorm everywhere**: No LayerNorm, no BatchNorm. AdaLN blocks use `elementwise_affine=False` since scale/shift come from the conditioning MLP.
- **AdaLN conditioning**: Time `t` injected via adaptive RMSNorm modulation (DiT-style).
- **Muon + SGD optimizer**: Muon (Newton-Schulz orthogonalised momentum) for 2D weight matrices, SGD (momentum=0.9) for biases/norms/embeddings.

### Source Layout (`src/flowmatch_vae/`)

| File | Role |
|------|------|
| `config.py` | Dataclass configs: `MultiScaleEncoderConfig`, `MultiScaleDecoderConfig`, `mHCConfig`, `TrainConfig` composed into `Config` |
| `models/conv_encoder.py` | Multi-scale encoder + prior: `SwiGLUConv`, `AttnPool2x2`, `Upsample2x`, `FusionAttention`, `MultiScalePrior`, `MultiScaleConvEncoder` |
| `models/swin.py` | Core blocks — `RMSNorm`, `KimiLinearAttention`, `KimiLinearCrossAttention`, `SwiGLUFFN`, `mHCConnection`, `CrossAttnAdaLNSwinBlock`, `PatchEmbed` |
| `models/encoder.py` | Legacy `SwinEncoder` (kept for backward compatibility) |
| `models/decoder.py` | `FlowDecoder` — U-Net multi-scale OT-CFM velocity network with down/up paths and per-level cross-attention |
| `models/vae.py` | `FlowMatchVAE` — encoder + decoder + prior, `compute_loss()` (FM + KL + prior), `sample()`, `reconstruct()` |
| `optimizers.py` | `Muon` optimizer + `split_param_groups()` |
| `data/celeba.py` | CelebA with in-memory caching |
| `train.py` | Single-GPU training loop |
| `train_dist.py` | 8-GPU distributed training via Ray + DDP |

### Block Variants

- **`CrossAttnAdaLNSwinBlock`** (swin.py): DiT block used at every decoder level. Self-attn + cross-attn + SwiGLU FFN, all with mHC residual. AdaLN time conditioning → 9 params.
- **`MultiScaleConvEncoder`** (conv_encoder.py): 6 stages SwiGLUConv + AttnPool2x2, FusionAttention fusion, returns `(mu, logvar, per_scale_tokens)`.
- **`MultiScalePrior`** (conv_encoder.py): z → predicts scales {1,3,4,5}. ConvTranspose up + Conv down + SwiGLUConv.
- **`Upsample2x`** (conv_encoder.py): Nearest upsample 2× + SwiGLUConv refinement.

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
