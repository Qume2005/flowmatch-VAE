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
