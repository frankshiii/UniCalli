# Experimental Apple Silicon inference

This patch adds an MPS execution path. Small conditional UniCalli Transformer
tests run on Apple Silicon; **full pretrained-model generation has not yet been
validated**. This is not a claim that the model fits in 24 GB unified memory.

## Requirements

- Apple Silicon, macOS 14 or newer (the existing pipeline uses bfloat16).
- An MPS-enabled PyTorch build. Smoke tests were run with Python 3.12 and
  PyTorch 2.14.1 on an M5 Pro with 24 GB unified memory.
- Substantially more memory than the 24.3 GB diffusion checkpoint alone:
  text encoders, the VAE, embeddings and activations need additional memory.
  The minimum memory for full generation has not been measured.
- Enough disk space for the checkpoint, encoders, embeddings and environment.

CUDA's advertised 18 GB 4-bit requirement does not apply to this MPS path.
DeepSpeed and the current 4-bit path are rejected on MPS/CPU before model loading.
CPU offloading moves models between stages, but CPU and GPU share physical memory
on Apple Silicon. It does not eliminate the total parameter memory requirement.

## Installation and experimental inference

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'import torch; print(torch.backends.mps.is_available())'
```

Download the official model package following the main README and accept its
access conditions. The original academic/non-commercial restrictions still apply.

```sh
python inference.py --device mps \
  --checkpoint ./checkpoints/unicalli-base_cleaned.bin \
  --intern-vlm-path ./checkpoints/internvl_embedding \
  --text '生日快乐喵' --font 楷 --author 颜真卿 \
  --steps 25 --seed 42 --output output.png
```

The released model still requires exactly five characters and produces a column.
The Gradio entry point selects CUDA, then MPS, then CPU automatically; override
it with `UNICALLI_DEVICE=mps python app.py`. The 4-bit checkbox is disabled outside
CUDA. No CPU fallback for unsupported Metal operations is enabled implicitly.

## What changed

- MPS RoPE uses float32; CUDA/CPU retain the original float64 calculation.
- Seeded MPS noise is generated on CPU and transferred to Metal, without changing
  global RNG state. Cross-device full-image equivalence is not promised.
- MPS `.bin` checkpoint loading uses `mmap=True`, `weights_only=True`, and assigns
  parameters into a meta model rather than first allocating a second full copy.
  Incomplete checkpoints fail explicitly instead of leaving meta tensors unloaded.
  This path expects modern PyTorch ZIP-format tensor checkpoints. It does not
  convert or quantize the released model.
- The sampler receives the requested offload setting and moves the VAE encoder
  back to the device before encoding; cache clearing targets the selected backend.
- DeepSpeed is omitted on macOS; the Hugging Face Hub range is compatible with
  the existing Gradio 6.3.0 and Transformers 4.43.3 dependencies.

## Validation

```sh
pip install pytest
python -m pytest tests/test_runtime.py -q
```

Tests cover device selection, unsupported optimization options, checkpoint
assignment, RNG behavior, RoPE, attention, and real conditional double/single
Transformer blocks on MPS. A sampler integration test also runs encoding,
denoising and decoding twice with offloading, a small random Transformer and stub
encoders/VAE, checking that both runs finish and have identical output. All 17
tests passed on the machine described above. MPS tests skip when no Apple GPU is
exposed. These tests do not establish pretrained-model quality or memory fit.

Before calling the port production-ready, run the official checkpoint on a Mac
with sufficient memory, record peak memory and wall time, inspect generated text,
and repeat generation to check offload lifecycle behavior. CUDA end-to-end
regression testing and MPS quantization remain outstanding.
