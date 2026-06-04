import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.decoder import FlowDecoder


def test_decoder_output_shape():
    cfg = Config().decoder
    decoder = FlowDecoder(cfg)
    x_t = torch.randn(4, 3, 64, 64)
    t = torch.rand(4)
    z = torch.randn(4, 8, 8, 256)
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
