# Performance and ROCm

## Recommended settings

| Setting | Recommended | Notes |
|---------|-------------|-------|
| TTS Mode | `local` | Built-in engine, no external server |
| Compile Codec | `true` | Faster decoding after one-time warmup |
| Parallel Workers | 20-60 | Higher throughput, higher VRAM usage |
| Render Mode | Batch (Fast) | Batched TTS calls |

## Example benchmark

Tested on AMD RX 7900 XTX (24 GB VRAM, ROCm 6.3):

| Configuration | Throughput |
|--------------|------------|
| Standard mode (sequential) | ~1x real-time |
| Batch mode, no codec compile | ~2x real-time |
| Batch mode + compile codec | ~3-6x real-time |

## ROCm (AMD GPU) notes

- AMD GPU acceleration is Linux-only (ROCm 6.3 expected).
- AMD on Windows runs in CPU mode.
- Threadspeak applies ROCm-specific compatibility/optimization behavior automatically where supported.
