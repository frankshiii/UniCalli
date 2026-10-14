"""Device helpers shared by the inference entry points."""

import torch


def resolve_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    result = torch.device(device)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use an Apple Silicon Mac with an MPS-enabled PyTorch build.")
    return result


def validate_optimizations(device, use_deepspeed=False, use_4bit_quantization=False):
    if torch.device(device).type != "cuda" and (use_deepspeed or use_4bit_quantization):
        raise ValueError("DeepSpeed and the current 4-bit inference path require CUDA; disable them for MPS/CPU.")


def empty_device_cache(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def seeded_random(shape, device, dtype, seed, normal=True):
    """Use a CPU generator for MPS, preserving a reproducible local seed."""
    device = torch.device(device)
    random_device = torch.device("cpu") if device.type == "mps" else device
    generator = torch.Generator(device=random_device).manual_seed(seed)
    operation = torch.randn if normal else torch.rand
    return operation(*shape, device=random_device, dtype=dtype, generator=generator).to(device)


def assign_checkpoint(model, state_dict):
    """Materialize a meta model without allocating a second full parameter copy."""
    missing = set(model.state_dict()) - set(state_dict)
    if missing:
        raise ValueError(f"MPS loading requires a complete checkpoint; missing keys: {sorted(missing)}")
    model.load_state_dict(state_dict, strict=False, assign=True)
    return model
