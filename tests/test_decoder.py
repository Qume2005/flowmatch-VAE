import pytest
import torch
from flowmatch_vae.config import Config
from flowmatch_vae.models.decoder import FlowDecoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Decoder tests require CUDA")


def test_decoder_output_shape():
    cfg = Config()
    device = torch.device("cuda")
    decoder = FlowDecoder(cfg.decoder, mhc_cfg=cfg.mhc).to(device)
    x_t = torch.randn(2, 3, 64, 64, device=device)
    t = torch.rand(2, device=device)
    z = torch.randn(2, 8, 8, 256, device=device)
    v = decoder(x_t, t, z)
    assert v.shape == (2, 3, 64, 64), f"Expected (2,3,64,64), got {v.shape}"


def test_decoder_gradient_flows():
    cfg = Config()
    device = torch.device("cuda")
    decoder = FlowDecoder(cfg.decoder, mhc_cfg=cfg.mhc).to(device)
    x_t = torch.randn(1, 3, 64, 64, device=device)
    t = torch.rand(1, device=device)
    z = torch.randn(1, 8, 8, 256, device=device)
    v = decoder(x_t, t, z)
    v.sum().backward()
    assert all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
