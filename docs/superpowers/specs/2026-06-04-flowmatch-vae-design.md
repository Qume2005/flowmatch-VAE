# Flow Matching VAE 设计规格

## 概述

用 Swin Transformer 编码器将图片嵌入到空间式语义潜在向量，再用 OT-CFM (Optimal Transport Conditional Flow Matching) 解码器还原图片。

- **数据集**: CelebA, 64×64
- **潜在空间**: 8×8×256 空间式 feature map
- **推理步数**: 4-8 步欧拉积分

## 架构

```
图片 x (64×64×3)
    ↓
[Swin Encoder] → μ, logvar (8×8×256)
    ↓ reparameterize
z ~ N(μ, σ²)  (8×8×256)
    ↓
[OT-CFM Velocity Net] ← condition: z, time: t
    ↓
v_θ(x_t, t, z) → 速度场 (64×64×3)
    ↓ 欧拉积分
x̂ (64×64×3) 重建图片
```

## Swin Encoder

- **Patch Embedding**: 4×4 patch → 16×16 tokens, dim=128
- **Stage 1**: 16×16, 2 个 Swin Transformer block, dim=128, head数=4
- **Patch Merging**: 16×16 → 8×8, dim=256
- **Stage 2**: 8×8, 6 个 Swin Transformer block, dim=256, head数=8
- **输出头**: 两个独立 Linear 层分别输出 μ 和 logvar（8×8×256）
- **窗口大小**: 4（8×8 feature map 上窗口覆盖半个空间维度）

### Swin Transformer Block

标准实现：
1. W-MSA (Window Multi-Head Self-Attention) + residual + LN
2. Shifted W-MSA + residual + LN（交替使用）
3. FFN (MLP) + residual + LN

## OT-CFM 速度场网络

条件 Swin 架构，预测速度场 v_θ(x_t, t, z)。

### 输入处理

- **x_t 输入**: 64×64×3 → 4×4 patch embed → 16×16 tokens, dim=256
- **z 注入**: 8×8×256 → repeat 2×2 spatial → 16×16×256，与 x_t tokens 相加
- **时间注入**: t → Sinusoidal positional embedding → MLP → adaLN 参数 (scale, shift)

### 网络结构

- 6 个 Swin block, dim=256, head数=8, 窗口大小=4
- adaLN 调制: 在每个 block 的 attention 和 FFN 前施加 scale & shift
- 无 patch merging（保持 16×16 分辨率）

### 输出头

- Linear(256, 4×4×3) → reshape → 64×64×3 速度场

## Loss

### Flow Matching Loss

```python
# OT-CFM
x0 = torch.randn_like(x1)  # 噪声
t = torch.rand(B, 1, 1, 1)  # 时间采样
xt = (1 - t) * x0 + t * x1  # 线性插值
v_target = x1 - x0           # 目标速度
v_pred = model(xt, t, z)     # 网络预测
L_FM = F.mse_loss(v_pred, v_target)
```

### KL Divergence

```python
L_KL = -0.5 * torch.mean(1 + logvar - mu**2 - (logvar).exp())
```

### 总 Loss

```
L = L_FM + β * L_KL
```

- β 典型值: 0.01 ~ 1.0
- β warmup: 前 N epoch 线性从 0 增到目标值

## 推理

欧拉积分，从 t=0 到 t=1:

```python
x = torch.randn(B, 3, 64, 64)  # 初始噪声
z = encoder.encode(image)        # 或从先验采样
steps = 8
dt = 1.0 / steps
for i in range(steps):
    t = torch.tensor(i / steps)
    v = velocity_net(x, t, z)
    x = x + v * dt
```

## 训练配置

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing |
| Batch size | 128 |
| Epochs | 100-200 |
| β (KL weight) | 0.1, warmup 10 epochs |
| 数据预处理 | Center crop 178×178 → resize 64×64, [-1,1] normalize |

## 项目结构

```
flowmatch-vae/
├── src/
│   └── flowmatch_vae/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── swin_encoder.py      # Swin Transformer 编码器
│       │   ├── flow_decoder.py      # OT-CFM 速度场网络
│       │   └── vae.py               # 组合 Encoder + Decoder
│       ├── data/
│       │   ├── __init__.py
│       │   └── celeba.py            # CelebA 数据加载
│       ├── train.py                 # 训练循环
│       ├── sample.py                # 采样/推理脚本
│       └── config.py                # 超参数配置
├── pyproject.toml
└── README.md
```

## 依赖

- PyTorch >= 2.0
- torchvision
- einops
- timm (可选，参考 Swin 实现但不直接依赖)
- numpy
