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
    mu, logvar = model.encode(x)
    assert mu.shape == (2, 8, 8, 256)
    z = model.reparameterize(mu, logvar)
    assert z.shape == (2, 8, 8, 256)


def test_end_to_end_overfit_single_batch():
    """在单个 batch 上过拟合，验证训练流程完整。"""
    cfg = Config()
    cfg.encoder.depths = (1, 1)
    cfg.decoder.depth = 1
    model = FlowMatchVAE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    x = torch.randn(4, 3, 64, 64)
    for _ in range(800):
        losses = model.compute_loss(x)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    assert losses["fm_loss"].item() < 1.0, f"FM loss should decrease, got {losses['fm_loss'].item()}"
