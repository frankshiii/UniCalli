from types import SimpleNamespace

import pytest
import torch

import inference
from src.flux.model import Flux, FluxParams
from src.flux.sampling import get_noise
from src.flux.xflux_pipeline import XFluxSampler
from PIL import Image


def test_sampler_receives_offload(monkeypatch):
    monkeypatch.setattr(XFluxSampler, "_load_embedding", lambda *args: (None, None))
    sampler = XFluxSampler(
        clip=torch.nn.Identity(), t5=torch.nn.Identity(), ae=torch.nn.Identity(),
        ref_latent=None, model=torch.nn.Identity(), device=torch.device("cpu"),
        intern_vlm_path="unused", offload=True,
    )
    assert sampler.offload


def test_mps_checkpoint_loader_without_full_weights(tmp_path, monkeypatch):
    params = FluxParams(
        in_channels=16, vec_in_dim=8, context_in_dim=16,
        hidden_size=32, mlp_ratio=2, num_heads=4, depth=1,
        depth_single_blocks=1, axes_dim=[2, 2, 4], theta=10000,
        qkv_bias=True, guidance_embed=True,
    )
    original = Flux(params)
    original.init_module_embeddings(320, 896)
    original.to(torch.bfloat16)
    path = tmp_path / "checkpoint.bin"
    torch.save(original.state_dict(), path)
    monkeypatch.setitem(inference.configs, "test", SimpleNamespace(params=params))
    generator = inference.CalligraphyGenerator.__new__(inference.CalligraphyGenerator)
    generator.device = torch.device("mps")
    loaded = generator._load_model_from_checkpoint(str(path), "test", offload=True)
    assert not loaded.training
    assert not any(p.is_meta or p.requires_grad for p in loaded.parameters())
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    assert loaded.img_in.weight.dtype == torch.bfloat16
    torch.testing.assert_close(loaded.img_in.weight, original.img_in.weight)


def test_optimization_rejected_before_download(monkeypatch):
    monkeypatch.setattr(inference, "load_t5", lambda *args: pytest.fail("must not load weights"))
    with pytest.raises(ValueError, match="require CUDA"):
        inference.CalligraphyGenerator(device="cpu", use_deepspeed=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_column_noise_shape():
    actual = get_noise(1, 640, 128, torch.device("mps"), torch.bfloat16, 42)
    expected = get_noise(1, 640, 128, torch.device("cpu"), torch.bfloat16, 42)
    assert actual.shape == (1, 16, 80, 16)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_sampler_repeat_generation_with_offload(monkeypatch):
    """Run the actual sampler twice with small random weights and stub encoders."""
    class TextEncoder(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.register_buffer("value", torch.zeros(shape, dtype=torch.bfloat16))

        def forward(self, text):
            return self.value

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 16, kernel_size=8, stride=8)
            self.decoder = torch.nn.ConvTranspose2d(16, 3, kernel_size=8, stride=8)

        def encode(self, image):
            return self.encoder(image)

        def decode(self, latent):
            return self.decoder(latent)

    embedding = torch.nn.Embedding(10, 896)
    tokenizer = lambda *args, **kwargs: {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    monkeypatch.setattr(XFluxSampler, "_load_embedding", lambda *args: (embedding, tokenizer))
    params = FluxParams(
        in_channels=64, vec_in_dim=8, context_in_dim=16,
        hidden_size=32, mlp_ratio=2, num_heads=4, depth=1,
        depth_single_blocks=1, axes_dim=[2, 2, 4], theta=10000,
        qkv_bias=True, guidance_embed=True,
    )
    model = Flux(params)
    model.init_module_embeddings(320, 896)
    model = model.to(torch.bfloat16).eval().requires_grad_(False)
    sampler = XFluxSampler(
        TextEncoder((1, 8)), TextEncoder((1, 512, 16)), VAE(),
        None, model, torch.device("mps"), "unused", offload=True,
    )
    results = []
    for _ in range(2):
        image, _ = sampler(
            prompt="calligraphy", cond_text="生日快乐喵", width=128, height=640,
            controlnet_image=Image.new("RGB", (128, 640), "white"),
            num_steps=2, seed=42,
        )
        assert image.size == (128, 640)
        assert next(sampler.model.parameters()).device.type == "cpu"
        assert next(sampler.ae.encoder.parameters()).device.type == "cpu"
        assert next(sampler.ae.decoder.parameters()).device.type == "cpu"
        results.append(image.tobytes())
    assert results[0] == results[1]
