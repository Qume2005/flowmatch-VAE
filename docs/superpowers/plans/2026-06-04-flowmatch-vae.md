# Flow Matching VAE 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Swin Transformer 编码器 + OT-CFM 解码器的 VAE，在 CelebA 64×64 上训练和采样。

**Architecture:** Swin Encoder (4×4 patch embed → Stage1 dim=128 → PatchMerge → Stage2 dim=256 → μ/logvar) + OT-CFM Velocity Net (adaLN Swin blocks, 条件化于 z 和 t)。训练用 MSE on velocity，推理用欧拉积分。

**Tech Stack:** PyTorch, torchvision, einops, numpy

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/flowmatch_vae/__init__.py` | 包初始化 |
| `src/flowmatch_vae/config.py` | 所有超参数 dataclass |
| `src/flowmatch_vae/models/swin.py` | Swin 基础组件：WindowAttention, SwinBlock, AdaLNSwinBlock, PatchEmbed, PatchMerge |
| `src/flowmatch_vae/models/encoder.py` | SwinEncoder → μ, logvar |
| `src/flowmatch_vae/models/decoder.py` | FlowDecoder (OT-CFM 速度场网络) |
| `src/flowmatch_vae/models/vae.py` | FlowMatchVAE 组合 encoder + decoder + loss |
| `src/flowmatch_vae/data/celeba.py` | CelebA 64×64 数据加载 |
| `src/flowmatch_vae/train.py` | 训练循环 |
| `src/flowmatch_vae/sample.py` | 采样/推理脚本 |
| `tests/test_swin.py` | Swin 组件测试 |
| `tests/test_encoder.py` | Encoder 测试 |
| `tests/test_decoder.py` | Decoder 测试 |
| `tests/test_vae.py` | VAE + loss 测试 |

---

### Task 1: 项目脚手架

**Files:**
- Modify: `pyproject.toml`
- Create: `src/flowmatch_vae/__init__.py`, `src/flowmatch_vae/models/__init__.py`, `src/flowmatch_vae/data/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: 初始化 git**

```bash
cd /home/larkume/projects/flowmatch-VAE
git init
```

- [ ] **Step 2: 更新 pyproject.toml**

```toml
[project]
name = "flowmatch-vae"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchvision",
    "einops",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: 创建目录和 __init__.py**

```bash
mkdir -p src/flowmatch_vae/models src/flowmatch_vae/data tests
touch src/flowmatch_vae/__init__.py
touch src/flowmatch_vae/models/__init__.py
touch src/flowmatch_vae/data/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: 安装依赖**

```bash
cd /home/larkume/projects/flowmatch-VAE
pip install -e ".[dev]"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: project scaffolding with dependencies"
```

---

### Task 2: 配置模块

**Files:**
- Create: `src/flowmatch_vae/config.py`

- [ ] **Step 1: 实现 config.py**

```python
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
    latent_spatial_size: int = 8  # z 的空间尺寸 (8×8)
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
    sample_interval: int = 5  # 每 N epoch 采样可视化
    save_interval: int = 20  # 每 N epoch 存 checkpoint


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
```

- [ ] **Step 2: 验证 import**

```bash
python -c "from flowmatch_vae.config import Config; c = Config(); print(c)"
```

Expected: 打印 Config 对象，无报错。

- [ ] **Step 3: Commit**

```bash
git add src/flowmatch_vae/config.py
git commit -m "feat: add configuration dataclasses"
```

---

### Task 3: Swin 基础组件

这是最核心也最复杂的部分。包含窗口注意力、SwinBlock、AdaLNSwinBlock、PatchEmbed、PatchMerge。

**Files:**
- Create: `src/flowmatch_vae/models/swin.py`
- Create: `tests/test_swin.py`

- [ ] **Step 1: 写测试 test_swin.py**

```python
import pytest
import torch

from flowmatch_vae.models.swin import (
    PatchEmbed,
    PatchMerge,
    WindowAttention,
    SwinBlock,
    AdaLNSwinBlock,
    window_partition,
    window_reverse,
)


class TestWindowPartitionReverse:
    def test_roundtrip(self):
        x = torch.randn(2, 8, 8, 64)
        windows = window_partition(x, window_size=4)
        assert windows.shape == (8, 4, 4, 64)  # 2 * (8/4)*(8/4) = 8 windows
        restored = window_reverse(windows, window_size=4, H=8, W=8)
        assert torch.allclose(x, restored)

    def test_shape(self):
        x = torch.randn(1, 16, 16, 128)
        windows = window_partition(x, window_size=4)
        assert windows.shape == (16, 4, 4, 128)  # 1 * 4 * 4


class TestWindowAttention:
    def test_shape(self):
        dim, num_heads, ws = 128, 4, 4
        attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=ws)
        x = torch.randn(8, ws * ws, dim)  # 8 windows, 16 tokens each
        out = attn(x)
        assert out.shape == (8, ws * ws, dim)


class TestPatchEmbed:
    def test_shape(self):
        pe = PatchEmbed(in_channels=3, patch_size=4, embed_dim=128)
        x = torch.randn(2, 3, 64, 64)
        out = pe(x)
        assert out.shape == (2, 16, 16, 128)


class TestPatchMerge:
    def test_shape(self):
        pm = PatchMerge(dim=128)
        x = torch.randn(2, 16, 16, 128)
        out = pm(x)
        assert out.shape == (2, 8, 8, 256)  # spatial /2, dim *2


class TestSwinBlock:
    def test_shape(self):
        block = SwinBlock(dim=128, num_heads=4, window_size=4, shift_size=0)
        x = torch.randn(2, 16, 16, 128)
        out = block(x)
        assert out.shape == (2, 16, 16, 128)

    def test_shifted(self):
        block = SwinBlock(dim=128, num_heads=4, window_size=4, shift_size=2)
        x = torch.randn(2, 16, 16, 128)
        out = block(x)
        assert out.shape == (2, 16, 16, 128)


class TestAdaLNSwinBlock:
    def test_shape(self):
        block = AdaLNSwinBlock(dim=256, num_heads=8, window_size=4, shift_size=0)
        x = torch.randn(2, 16, 16, 256)
        cond = torch.randn(2, 256)  # conditioning signal (time embed)
        out = block(x, cond)
        assert out.shape == (2, 16, 16, 256)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/larkume/projects/flowmatch-VAE
python -m pytest tests/test_swin.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 swin.py**

```python
"""Swin Transformer building blocks.

Components:
- window_partition / window_reverse: 窗口分区与还原
- WindowAttention: 窗口内多头注意力 (含相对位置偏置)
- SwinBlock: 标准 Swin Transformer Block (W-MSA / SW-MSA + FFN)
- AdaLNSwinBlock: 带 adaLN 调制的 Swin Block (用于 decoder)
- PatchEmbed: Patch 嵌入 (Conv2d)
- PatchMerge: Patch 合并 (spatial /2, channel *2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nH*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B*nH*nW, ws, ws, C) -> (B, H, W, C)"""
    nH, nW = H // window_size, W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """窗口内多头自注意力，含可学习相对位置偏置。"""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表: (2*ws-1)*(2*ws-1) 个条目, 每个头一个
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1, num_heads))
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 预计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=0)  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, ws*ws)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_pos_index = rel_coords.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("relative_position_index", rel_pos_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*nW, ws*ws, C)"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, nH, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 加入相对位置偏置
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1)  # (nH, N, N)
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Shifted-window attention mask (预计算, 避免动态 padding)
# ---------------------------------------------------------------------------

def _compute_shift_mask(window_size: int, shift_size: int, H: int, W: int) -> torch.Tensor | None:
    """计算 shifted window attention 的 attention mask。
    返回 (nW, ws*ws, ws*ws) 或 None (shift_size == 0)。
    """
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  # (nW, ws*ws, ws*ws)


# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """标准 Swin Transformer Block: LN -> W-MSA/SW-MSA -> Res -> LN -> FFN -> Res"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        ws = self.window_size

        shortcut = x
        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, ws)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, ws * ws, C)

        # Attention mask for shifted windows
        attn_mask = _compute_shift_mask(ws, self.shift_size, H, W)
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device).unsqueeze(1)  # (nW, 1, ws*ws, ws*ws)
            nW = attn_mask.shape[0]
            attn_mask = attn_mask.expand(B, -1, -1, -1).reshape(B * nW, 1, ws * ws, ws * ws)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows)  # (nW*B, ws*ws, C)
        # 注意: WindowAttention 没有直接接受 mask，我们在 attention 计算中需要加上 mask
        # 但为简化，我们在 forward 中直接应用
        if attn_mask is not None:
            # 重新计算带 mask 的 attention
            B_ = x_windows.shape[0]
            N = x_windows.shape[1]
            qkv = self.attn.qkv(x_windows).reshape(B_, N, 3, self.attn.num_heads, C // self.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.attn.scale
            bias = self.attn.relative_position_bias_table[self.attn.relative_position_index.view(-1)]
            bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
            attn = attn + bias
            attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            attn_windows = self.attn.proj(attn_windows)

        # Merge windows
        shifted_x = window_reverse(attn_windows, ws, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# AdaLN Swin Block (用于 decoder, 条件化于时间嵌入)
# ---------------------------------------------------------------------------

class AdaLNSwinBlock(nn.Module):
    """Swin Block with adaptive LayerNorm conditioning.

    cond -> MLP -> (scale, shift, gate) 用于调制每个 sub-layer。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # adaLN modulation: cond_dim -> 6 * dim (scale1, shift1, scale2, shift2, gate1, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C), cond: (B, C)"""
        B, H, W, C = x.shape
        ws = self.window_size

        # AdaLN modulation
        params = self.adaLN_modulation(cond)  # (B, 6*C)
        s1, sh1, s2, sh2, g1, g2 = params.chunk(6, dim=-1)
        s1, sh1 = s1.unsqueeze(1).unsqueeze(1), sh1.unsqueeze(1).unsqueeze(1)
        s2, sh2 = s2.unsqueeze(1).unsqueeze(1), sh2.unsqueeze(1).unsqueeze(1)
        g1, g2 = g1.unsqueeze(1).unsqueeze(1), g2.unsqueeze(1).unsqueeze(1)

        # Attention branch
        x_norm = self.norm1(x) * (1 + s1) + sh1

        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm

        x_windows = window_partition(shifted_x, ws).view(-1, ws * ws, C)

        attn_mask = _compute_shift_mask(ws, self.shift_size, H, W)
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device).unsqueeze(1).expand(B, -1, -1, -1)
            nW = attn_mask.shape[1]
            attn_mask = attn_mask.reshape(B * nW, 1, ws * ws, ws * ws)
            # 带 mask 的 attention
            B_ = x_windows.shape[0]
            N = x_windows.shape[1]
            qkv = self.attn.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.attn.scale
            bias = self.attn.relative_position_bias_table[self.attn.relative_position_index.view(-1)]
            bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
            attn = attn + bias + attn_mask
            attn = attn.softmax(dim=-1)
            attn_out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            attn_out = self.attn.proj(attn_out)
        else:
            attn_out = self.attn(x_windows)

        shifted_x = window_reverse(attn_out, ws, H, W)
        if self.shift_size > 0:
            x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted_x

        x = x + g1 * x_attn

        # FFN branch
        x = x + g2 * self.mlp(self.norm2(x) * (1 + s2) + sh2)
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """将图片分割为不重叠 patch 并线性投影。"""

    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, H/ps, W/ps, embed_dim)"""
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.permute(0, 2, 3, 1)  # (B, H/ps, W/ps, embed_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Patch Merging
# ---------------------------------------------------------------------------

class PatchMerge(nn.Module):
    """合并 2×2 邻域 patch, 空间减半, 通道翻倍。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H/2, W/2, 2*C)"""
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous()  # (B, H/2, W/2, 2, 2, C) -> rearrange
        x = x.view(B, H // 2, W // 2, 4 * C)  # concat 2×2 neighbors
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth (drop paths) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob
```

- [ ] **Step 4: 运行测试**

```bash
cd /home/larkume/projects/flowmatch-VAE
python -m pytest tests/test_swin.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/swin.py tests/test_swin.py
git commit -m "feat: Swin Transformer building blocks"
```

---

### Task 4: Swin Encoder

**Files:**
- Create: `src/flowmatch_vae/models/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: 写测试 test_encoder.py**

```python
import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.encoder import SwinEncoder


def test_encoder_output_shape():
    cfg = Config().encoder
    encoder = SwinEncoder(cfg)
    x = torch.randn(4, 3, 64, 64)
    mu, logvar = encoder(x)
    # 64 / patch_size(4) = 16, 再经过 PatchMerge /2 = 8
    assert mu.shape == (4, 8, 8, 256), f"Expected (4,8,8,256), got {mu.shape}"
    assert logvar.shape == (4, 8, 8, 256)


def test_encoder_gradient_flows():
    cfg = Config().encoder
    encoder = SwinEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    mu, logvar = encoder(x)
    loss = mu.sum() + logvar.sum()
    loss.backward()
    assert all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_encoder.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 encoder.py**

```python
"""Swin Transformer Encoder: 图片 -> μ, logvar 潜在空间。"""

from __future__ import annotations

import torch
import torch.nn as nn

from flowmatch_vae.config import EncoderConfig
from flowmatch_vae.models.swin import PatchEmbed, PatchMerge, SwinBlock


class SwinEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=cfg.in_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
        )

        # Stochastic depth rate 线性递增
        depths = cfg.depths
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_blocks)]

        # 构建 stages
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        dim = cfg.embed_dim

        for i, depth in enumerate(depths):
            stage = nn.Sequential(*[
                SwinBlock(
                    dim=dim,
                    num_heads=cfg.num_heads[i],
                    window_size=cfg.window_size,
                    shift_size=0 if (j % 2 == 0) else cfg.window_size // 2,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_path=dpr[sum(depths[:i]) + j],
                )
                for j in range(depth)
            ])
            self.stages.append(stage)

            # 除最后一个 stage 外，每 stage 后接 PatchMerge
            if i < len(depths) - 1:
                self.downsample.append(PatchMerge(dim))
                dim *= 2
            else:
                self.downsample.append(None)

        self.out_dim = dim

        # μ 和 logvar 投影头
        self.mu_head = nn.Linear(dim, dim)
        self.logvar_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) -> μ: (B, 8, 8, C), logvar: (B, 8, 8, C)"""
        x = self.patch_embed(x)  # (B, 16, 16, embed_dim)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.downsample[i] is not None:
                x = self.downsample[i](x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_encoder.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/encoder.py tests/test_encoder.py
git commit -m "feat: Swin Transformer encoder with μ/logvar heads"
```

---

### Task 5: OT-CFM 速度场网络 (Decoder)

**Files:**
- Create: `src/flowmatch_vae/models/decoder.py`
- Create: `tests/test_decoder.py`

- [ ] **Step 1: 写测试 test_decoder.py**

```python
import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.decoder import FlowDecoder


def test_decoder_output_shape():
    cfg = Config().decoder
    decoder = FlowDecoder(cfg)
    x_t = torch.randn(4, 3, 64, 64)       # 噪声图
    t = torch.rand(4)                       # 时间
    z = torch.randn(4, 8, 8, 256)           # 潜在向量
    v = decoder(x_t, t, z)
    assert v.shape == (4, 3, 64, 64), f"Expected (4,3,64,64), got {v.shape}"


def test_decoder_gradient_flows():
    cfg = Config().decoder
    decoder = FlowDecoder(cfg)
    x_t = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    z = torch.randn(2, 8, 8, 256)
    v = decoder(x_t, t, z)
    v.sum().backward()
    assert all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 decoder.py**

```python
"""OT-CFM 速度场网络：条件 Swin 架构，预测 velocity field。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from flowmatch_vae.config import DecoderConfig
from flowmatch_vae.models.swin import PatchEmbed, AdaLNSwinBlock


class SinusoidalTimeEmbedding(nn.Module):
    """将标量时间 t 编码为向量。"""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class FlowDecoder(nn.Module):
    """OT-CFM velocity network: v_θ(x_t, t, z) -> velocity field。

    架构:
    - x_t 通过 PatchEmbed 编码为 tokens
    - z 上采样后与 x_t tokens 相加
    - t 通过 sinusoidal embedding + MLP -> adaLN conditioning
    - N 个 AdaLNSwinBlock 处理
    - Linear 输出头 -> velocity
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # x_t 的 patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=cfg.out_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
        )

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(cfg.embed_dim)

        # z 的通道投影 (latent_dim -> embed_dim) + 上采样
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.z_upsample_factor = cfg.patch_size // cfg.latent_spatial_size * (cfg.image_size if hasattr(cfg, 'image_size') else 64) // cfg.patch_size // cfg.latent_spatial_size
        # 64 / 4 = 16 tokens; z is 8×8; upsample 2×
        self.z_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # AdaLN Swin blocks
        self.blocks = nn.ModuleList([
            AdaLNSwinBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                shift_size=0 if (i % 2 == 0) else cfg.window_size // 2,
                mlp_ratio=cfg.mlp_ratio,
            )
            for i in range(cfg.depth)
        ])

        # 输出头: tokens -> velocity pixels
        self.out_norm = nn.LayerNorm(cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.patch_size * cfg.patch_size * cfg.out_channels)

        self.patch_size = cfg.patch_size

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 3, 64, 64) 噪声图
        t:   (B,) 时间
        z:   (B, 8, 8, latent_dim) 潜在向量 (空间式)
        返回: (B, 3, 64, 64) 速度场
        """
        B = x_t.shape[0]

        # Patch embed x_t -> (B, 16, 16, embed_dim)
        h = self.patch_embed(x_t)

        # z 投影 + 上采样 -> (B, 16, 16, embed_dim)
        z_proj = self.z_proj(z)  # (B, 8, 8, embed_dim)
        # 上采样: (B, 8, 8, embed_dim) -> permute to (B, embed_dim, 8, 8) -> upsample -> permute back
        z_proj = z_proj.permute(0, 3, 1, 2)  # (B, embed_dim, 8, 8)
        z_proj = self.z_upsample(z_proj)       # (B, embed_dim, 16, 16)
        z_proj = z_proj.permute(0, 2, 3, 1)   # (B, 16, 16, embed_dim)

        # z 注入: 相加
        h = h + z_proj

        # 时间嵌入
        t_emb = self.time_embed(t)  # (B, embed_dim)

        # AdaLN Swin blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # 输出头
        h = self.out_norm(h)
        h = self.out_proj(h)  # (B, 16, 16, patch_size^2 * out_channels)

        # Reshape to image
        ps = self.patch_size
        h = h.permute(0, 3, 1, 2)  # (B, ps^2*C, 16, 16)
        h = nn.functional.pixel_shuffle(h, ps)  # (B, C, 64, 64)

        return h
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/decoder.py tests/test_decoder.py
git commit -m "feat: OT-CFM velocity network (FlowDecoder)"
```

---

### Task 6: VAE 模型组合

**Files:**
- Create: `src/flowmatch_vae/models/vae.py`
- Create: `tests/test_vae.py`

- [ ] **Step 1: 写测试 test_vae.py**

```python
import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE


def test_vae_loss_runs():
    cfg = Config()
    model = FlowMatchVAE(cfg)
    x = torch.randn(4, 3, 64, 64)
    losses = model.compute_loss(x)
    assert "loss" in losses
    assert "fm_loss" in losses
    assert "kl_loss" in losses
    assert losses["fm_loss"].shape == ()
    assert losses["kl_loss"].shape == ()


def test_vae_sample_shape():
    cfg = Config()
    model = FlowMatchVAE(cfg)
    model.eval()
    with torch.no_grad():
        x_recon = model.sample(num_samples=2, num_steps=4, device="cpu")
    assert x_recon.shape == (2, 3, 64, 64)


def test_vae_encode_decode():
    cfg = Config()
    model = FlowMatchVAE(cfg)
    x = torch.randn(2, 3, 64, 64)
    # 编码
    mu, logvar = model.encode(x)
    assert mu.shape == (2, 8, 8, 256)
    # 重参数化
    z = model.reparameterize(mu, logvar)
    assert z.shape == (2, 8, 8, 256)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_vae.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 vae.py**

```python
"""Flow Matching VAE: Swin Encoder + OT-CFM Decoder。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.config import Config
from flowmatch_vae.models.encoder import SwinEncoder
from flowmatch_vae.models.decoder import FlowDecoder


class FlowMatchVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SwinEncoder(cfg.encoder)
        self.decoder = FlowDecoder(cfg.decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) -> μ, logvar"""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化采样 z ~ N(μ, σ²)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """计算 OT-CFM + KL 联合 loss。

        Args:
            x: (B, 3, 64, 64) 原图, 值域 [-1, 1]
        Returns:
            dict with "loss", "fm_loss", "kl_loss"
        """
        B = x.shape[0]

        # 1. Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. OT-CFM loss
        x1 = x  # 目标: 原图
        x0 = torch.randn_like(x1)  # 源: 噪声
        t = torch.rand(B, 1, 1, 1, device=x.device)  # 均匀采样时间
        x_t = (1 - t) * x0 + t * x1  # OT 插值
        v_target = x1 - x0  # 目标速度

        # Decoder 预测速度
        v_pred = self.decoder(x_t, t.squeeze().view(B), z)
        fm_loss = F.mse_loss(v_pred, v_target)

        # 4. 总 loss (含 KL warmup, 实际 warmup 在 trainer 里控制)
        loss = fm_loss + self.cfg.train.kl_weight * kl_loss

        return {"loss": loss, "fm_loss": fm_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        num_steps: int = 8,
        z: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """从先验采样或给定 z 解码生成图片。

        Args:
            num_samples: 生成数量
            num_steps: 欧拉积分步数
            z: 可选, (B, 8, 8, 256) 潜在向量。None 则从先验采样。
            device: 设备
        Returns:
            (num_samples, 3, 64, 64) 生成图片
        """
        if z is None:
            z = torch.randn(num_samples, 8, 8, 256, device=device)
        else:
            num_samples = z.shape[0]

        x = torch.randn(num_samples, 3, 64, 64, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)
            v = self.decoder(x, t, z)
            x = x + v * dt

        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
        """编码后重建图片。"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.sample(num_steps=num_steps, z=z, device=x.device)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_vae.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/vae.py tests/test_vae.py
git commit -m "feat: FlowMatchVAE with OT-CFM loss and sampling"
```

---

### Task 7: CelebA 数据加载

**Files:**
- Create: `src/flowmatch_vae/data/celeba.py`

- [ ] **Step 1: 实现 celeba.py**

```python
"""CelebA 数据集加载, 预处理为 64×64, 归一化到 [-1, 1]。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


def get_transforms(image_size: int = 64) -> transforms.Compose:
    """CelebA 标准预处理: center crop 178×178 → resize → normalize [-1, 1]。"""
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])


def get_dataloader(
    data_path: str = "./data",
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 4,
    split: str = "train",
) -> DataLoader:
    transform = get_transforms(image_size)
    dataset = CelebA(
        root=data_path,
        split=split,
        target_type=[],
        transform=transform,
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
```

- [ ] **Step 2: 验证 import**

```bash
python -c "from flowmatch_vae.data.celeba import get_dataloader; print('OK')"
```

Expected: 打印 OK

- [ ] **Step 3: Commit**

```bash
git add src/flowmatch_vae/data/celeba.py
git commit -m "feat: CelebA data loading with 64×64 preprocessing"
```

---

### Task 8: 训练循环

**Files:**
- Create: `src/flowmatch_vae/train.py`

- [ ] **Step 1: 实现 train.py**

```python
"""训练 Flow Matching VAE。"""

from __future__ import annotations

import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flowmatch_vae.config import Config
from flowmatch_vae.data.celeba import get_dataloader
from flowmatch_vae.models.vae import FlowMatchVAE


def train(cfg: Config | None = None):
    cfg = cfg or Config()
    tc = cfg.train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    os.makedirs(tc.save_dir, exist_ok=True)
    os.makedirs(tc.log_dir, exist_ok=True)

    # 数据
    loader = get_dataloader(
        data_path=tc.data_path,
        batch_size=tc.batch_size,
        image_size=tc.image_size,
    )
    print(f"Dataset size: {len(loader.dataset)}")

    # 模型
    model = FlowMatchVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)

    # 训练循环
    for epoch in range(1, tc.epochs + 1):
        model.train()
        total_loss = 0.0
        total_fm = 0.0
        total_kl = 0.0
        n_batches = 0

        # KL warmup
        kl_weight = min(1.0, epoch / max(tc.kl_warmup_epochs, 1)) * tc.kl_weight

        epoch_start = time.time()
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            # 前向 + loss (手动控制 kl_weight)
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)

            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            B = images.shape[0]
            x1 = images
            x0 = torch.randn_like(x1)
            t = torch.rand(B, 1, 1, 1, device=device)
            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            v_pred = model.decoder(x_t, t.view(B), z)
            fm_loss = torch.nn.functional.mse_loss(v_pred, v_target)

            loss = fm_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_fm += fm_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                    f"loss={loss.item():.4f} fm={fm_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} kl_w={kl_weight:.4f}"
                )

        scheduler.step()
        elapsed = time.time() - epoch_start
        avg_loss = total_loss / n_batches
        avg_fm = total_fm / n_batches
        avg_kl = total_kl / n_batches

        print(
            f"Epoch {epoch}/{tc.epochs} | "
            f"loss={avg_loss:.4f} fm={avg_fm:.4f} kl={avg_kl:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # 保存 checkpoint
        if epoch % tc.save_interval == 0 or epoch == tc.epochs:
            path = os.path.join(tc.save_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, path)
            print(f"Saved checkpoint: {path}")

        # 采样可视化
        if epoch % tc.sample_interval == 0:
            _save_samples(model, cfg, epoch, device)

    print("Training complete!")


@torch.no_grad()
def _save_samples(model: FlowMatchVAE, cfg: Config, epoch: int, device: str):
    """保存采样图片用于可视化。"""
    from torchvision.utils import save_image

    model.eval()
    samples = model.sample(num_samples=16, num_steps=cfg.train.num_sample_steps, device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

    path = os.path.join(cfg.train.log_dir, f"samples_epoch{epoch}.png")
    save_image(samples, path, nrow=4)
    print(f"Saved samples: {path}")
    model.train()


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: 验证 import**

```bash
python -c "from flowmatch_vae.train import train; print('OK')"
```

Expected: 打印 OK

- [ ] **Step 3: Commit**

```bash
git add src/flowmatch_vae/train.py
git commit -m "feat: training loop with KL warmup and sampling"
```

---

### Task 9: 采样/推理脚本

**Files:**
- Create: `src/flowmatch_vae/sample.py`

- [ ] **Step 1: 实现 sample.py**

```python
"""从训练好的 FlowMatchVAE 采样或重建图片。"""

from __future__ import annotations

import argparse
import os

import torch
from torchvision.utils import save_image

from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[FlowMatchVAE, Config]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FlowMatchVAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def sample(
    checkpoint_path: str,
    num_samples: int = 16,
    num_steps: int = 8,
    output_path: str = "samples.png",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(checkpoint_path, device)
    images = model.sample(num_samples=num_samples, num_steps=num_steps, device=device)
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, output_path, nrow=4)
    print(f"Saved {num_samples} samples to {output_path}")


@torch.no_grad()
def reconstruct(
    checkpoint_path: str,
    image_path: str,
    num_steps: int = 8,
    output_path: str = "reconstruction.png",
    device: str = "auto",
):
    from torchvision import transforms
    from PIL import Image

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(checkpoint_path, device)
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    recon = model.reconstruct(x, num_steps=num_steps)
    comparison = torch.cat([x, recon], dim=0)
    comparison = (comparison.clamp(-1, 1) + 1) / 2
    save_image(comparison, output_path, nrow=2)
    print(f"Saved reconstruction to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowMatch VAE Sampling")
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument("--mode", choices=["sample", "reconstruct"], default="sample")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--image", type=str, help="Input image for reconstruction")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    if args.mode == "sample":
        sample(args.checkpoint, args.num_samples, args.num_steps, args.output)
    else:
        reconstruct(args.checkpoint, args.image, args.num_steps, args.output)
```

- [ ] **Step 2: 验证 import**

```bash
python -c "from flowmatch_vae.sample import sample; print('OK')"
```

Expected: 打印 OK

- [ ] **Step 3: Commit**

```bash
git add src/flowmatch_vae/sample.py
git commit -m "feat: sampling and reconstruction script"
```

---

### Task 10: 端到端测试

**Files:**
- Modify: `tests/test_vae.py` (追加)

- [ ] **Step 1: 在 test_vae.py 末尾追加端到端测试**

```python
def test_end_to_end_overfit_single_batch():
    """在单个 batch 上过拟合，验证训练流程完整。"""
    cfg = Config()
    # 加速: 小模型
    cfg.encoder.depths = (1, 1)
    cfg.decoder.depth = 1
    model = FlowMatchVAE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 3, 64, 64)
    for _ in range(20):
        losses = model.compute_loss(x)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    assert losses["fm_loss"].item() < 1.0, f"FM loss should decrease, got {losses['fm_loss'].item()}"
```

- [ ] **Step 2: 运行全部测试**

```bash
python -m pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_vae.py
git commit -m "test: end-to-end overfit test for training pipeline"
```

---

### Task 11: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README.md**

```markdown
# Flow Matching VAE

Swin Transformer encoder + OT-CFM decoder 的变分自编码器。

## 架构

- **Encoder**: Swin-T, 输入 64×64 图片, 输出 8×8×256 潜在空间 (μ, logvar)
- **Decoder**: OT-CFM 条件速度场网络, 给定 z 和噪声, 欧拉积分生成图片

## 安装

```bash
pip install -e ".[dev]"
```

## 训练

```bash
python -m flowmatch_vae.train
```

## 采样

```bash
# 从先验采样
python -m flowmatch_vae.sample checkpoints/checkpoint_epoch200.pt --mode sample --num-samples 16

# 重建图片
python -m flowmatch_vae.sample checkpoints/checkpoint_epoch200.pt --mode reconstruct --image test.jpg
```

## 测试

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```
