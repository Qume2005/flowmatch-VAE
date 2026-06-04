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
        assert windows.shape == (8, 4, 4, 64)
        restored = window_reverse(windows, window_size=4, H=8, W=8)
        assert torch.allclose(x, restored)

    def test_shape(self):
        x = torch.randn(1, 16, 16, 128)
        windows = window_partition(x, window_size=4)
        assert windows.shape == (16, 4, 4, 128)


class TestWindowAttention:
    def test_shape(self):
        dim, num_heads, ws = 128, 4, 4
        attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=ws)
        x = torch.randn(8, ws * ws, dim)
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
        assert out.shape == (2, 8, 8, 256)


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
        cond = torch.randn(2, 256)
        out = block(x, cond)
        assert out.shape == (2, 16, 16, 256)
