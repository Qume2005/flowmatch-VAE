import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.encoder import SwinEncoder


def test_encoder_output_shape():
    cfg = Config().encoder
    encoder = SwinEncoder(cfg)
    x = torch.randn(4, 3, 64, 64)
    mu, logvar = encoder(x)
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
