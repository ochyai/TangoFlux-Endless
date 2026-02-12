# TangoFlux Endless: Real-time Audio Generation on Apple Silicon

Real-time, endless soundscape generation from text prompts on Apple Silicon.
Converts TangoFlux's FluxTransformer2DModel to CoreML for ANE/GPU acceleration, achieving **RTF 0.28x** (3.6x faster than real-time).

> Part of *計算機の自然 (Computational Nature)* — an installation exploring the boundary between computational and natural sound environments.

**Paper:** [TangoFlux Endless: Real-time Text-to-Audio Generation on Apple Silicon via CoreML-accelerated Flow Matching](paper/main.pdf) (9 pages, 30 references)

## Key Results

| Backend | 25 steps / 18s audio | RTF | Speedup |
|---------|---------------------|-----|---------|
| MPS (float32) | 8.05s ± 0.27s | 0.447x | baseline |
| **CoreML (ANE/GPU)** | **5.06s ± 0.40s** | **0.281x** | **1.59x** |

> RTF (Real-Time Factor) = generation time / audio duration. Below 1.0 means faster than real-time. Measured on M2 Ultra, 10 runs, 2 warmup discarded.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Process (PyQt6 GUI + Audio Playback)                      │
│                                                                  │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────────────────┐  │
│  │ Feeder   │  │ Audio Callback  │  │ Layer Mixer           │  │
│  │ Thread   │──│ (44.1kHz mono)  │──│ N-layer crossfade     │  │
│  │          │  │                 │  │ sin²/cos² envelope    │  │
│  └────┬─────┘  └─────────────────┘  └───────────────────────┘  │
│       │ .npy files via tempdir                                   │
│       ▼                                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Worker Subprocess (subprocess.Popen — GIL-free)          │  │
│  │                                                            │  │
│  │  T5 Encoder ──→ FluxTransformer ──→ VAE Decoder           │  │
│  │  (PyTorch MPS)  (CoreML ANE/GPU)   (PyTorch MPS)          │  │
│  │       │                                                    │  │
│  │       └──→ Embedding Index (all-MiniLM-L6-v2, 384-dim)   │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │  Clip Archive (embedding-indexed, max 200 clips)          │  │
│  │  index.json + {hash}.npy                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why subprocess isolation?

Python's GIL (Global Interpreter Lock) causes CPU-bound PyTorch inference and I/O-bound audio callbacks to contend for execution time. In a threaded model, this produces audible pop/click artifacts. By running the generator in a separate process via `subprocess.Popen`, GIL contention is eliminated entirely.

### Hybrid inference pipeline

```
T5 Encoder (PyTorch, MPS float32)
  → encoder_hidden_states + pooled_projection
  → numpy.astype(float16)
FluxTransformer (CoreML, ANE/GPU internal float16)
  → noise_pred
  → numpy.astype(float32) → torch.Tensor
Scheduler Step (PyTorch, CPU float32)
  → updated latents
  → loop × 25 steps
VAE Decode (PyTorch, MPS float32)
  → waveform (44.1kHz)
```

### N-layer staggered playback

Multiple audio clips play simultaneously with phase offsets, cross-fading via sin²/cos² envelopes:

```
Layer 1: |====XXXX====XXXX====|  (started at 2/3 position)
Layer 2: |    ====XXXX====XXXX|  (started at 1/3 position)
Layer 3: |        ====XXXX====|  (started at beginning)
         ↑ playback start

Envelope per clip:
  ┌─── sin²(t) fade-in ───┐── sustain ──┌─── cos²(t) fade-out ──┐
  0                        1             1                        0
```

Output gain: `mixed /= N_layers` prevents clipping without hard limiting.

## Technical Contributions

### 1. CoreML Conversion of Flux RoPE (Novel)

The original diffusers `FluxTransformer2DModel` uses two operations incompatible with CoreML:

**Problem 1: `einsum` with ellipsis notation**
```python
# Original — coremltools cannot trace ellipsis einsum
out = torch.einsum("...n,d->...nd", pos, omega)
```

**Solution:** Replace with equivalent basic tensor operations:
```python
out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
```

**Problem 2: Rank-6 tensor exceeds CoreML limit (rank 5)**
```python
# Original — 2x2 rotation matrix creates rank-6 via unsqueeze
stacked = torch.stack([cos, -sin, sin, cos], dim=-1)
out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # rank 5
return emb.unsqueeze(1)  # rank 6!
```

**Solution:** Separate cos/sin representation with direct rotation:
```python
# rope() returns (cos, sin) — each rank 3
def rope(pos, dim, theta):
    out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    return torch.cos(out).float(), torch.sin(out).float()

# apply_rope() uses direct rotation — max rank 5
def apply_rope(xq, xk, freqs_cis):
    cos_emb, sin_emb = freqs_cis
    xq_r, xq_i = xq[..., 0], xq[..., 1]
    xq_out = torch.stack([
        cos_emb * xq_r - sin_emb * xq_i,
        sin_emb * xq_r + cos_emb * xq_i
    ], dim=-1).flatten(-2)
```

See [`patches/`](patches/) for complete diffs. Verified: trace max diff = 0.00e+00.

### 2. Negative Result: float16 on MPS Produces Artifacts

Attempting float16 inference on Apple MPS backend produces audible electrical noise artifacts in the generated audio. Root cause analysis:

- `inference_flow()` in TangoFlux creates **hardcoded float32 intermediate tensors** (`torch.tensor(float("nan"))`, `torch.randn()`, `torch.zeros()`)
- These mix with float16 model weights, causing dtype mismatch
- The scheduler's iterative computation accumulates precision errors over 25 steps
- Result: audible high-frequency noise overlaid on the generated audio

**CoreML's internal float16** (via `compute_precision=ct.precision.FLOAT16`) does not have this problem because precision management is handled by the CoreML runtime, not by Python-level tensor operations.

### 3. Archive Fallback with Embedding Similarity

When the generation pipeline cannot keep up with playback (buffer underrun), the system falls back to previously generated clips using semantic similarity:

```python
# Encode current prompt with all-MiniLM-L6-v2 (384-dim)
current_embedding = encode("gentle rain with distant thunder")

# Cosine similarity search over archive
candidates = sorted(archive_index, key=cosine_sim, reverse=True)[:3]
fallback_clip = random.choice(candidates)  # top-3 random for variety
```

## Setup

```bash
# Python 3.10+ recommended
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Apply diffusers patches (required for CoreML)
DIFFUSERS_PATH=$(python -c "import diffusers; print(diffusers.__path__[0])")
patch -p1 -d "$DIFFUSERS_PATH" < patches/transformer_flux.patch
patch -p1 -d "$DIFFUSERS_PATH" < patches/attention_processor.patch

# Convert transformer to CoreML (one-time, ~40s)
python src/convert_coreml.py

# Run
python src/endless_play.py
```

**Without CoreML:** The system works without CoreML conversion (auto-fallback to MPS), but generation will be ~1.59x slower.

## Files

```
TangoFlux-Endless/
├── README.md                          # This file
├── TECHNICAL_REPORT.md                # Detailed technical analysis
├── LICENSE                            # MIT
├── requirements.txt
├── paper/
│   ├── main.tex                      # arXiv paper (LaTeX source)
│   ├── main.pdf                      # arXiv paper (compiled PDF)
│   └── references.bib                # BibTeX references (30 entries)
├── src/
│   ├── endless_play.py                # PyQt6 GUI + N-layer crossfade player
│   ├── generator_worker.py            # Subprocess worker (CoreML/MPS)
│   └── convert_coreml.py             # FluxTransformer → CoreML conversion
├── patches/
│   ├── README.md                      # Patch application guide
│   ├── transformer_flux.patch         # RoPE refactoring for CoreML
│   └── attention_processor.patch      # apply_rope() refactoring
└── benchmarks/
    ├── paper_benchmark.py             # Comprehensive benchmark suite
    ├── results/                       # Benchmark results (JSON)
    ├── optimize_bench.py              # MPS optimization analysis
    └── benchmark.py                   # Basic MPS vs CPU comparison
```

## Optimization Attempts Summary

| Optimization | Result | Status |
|-------------|--------|--------|
| CoreML transformer (ANE/GPU) | 1.59x speedup | **Deployed** |
| Steps 50 → 25 | 2.0x speedup | **Deployed** |
| subprocess isolation | GIL-free playback | **Deployed** |
| N-layer staggered start | Seamless transitions | **Deployed** |
| Archive fallback (embedding similarity) | No buffer underrun gaps | **Deployed** |
| float16 on MPS | Electrical noise artifacts | **Reverted** |
| No-CFG (guidance_scale=0) | Weak text adherence | **Reverted** |
| float16 + no-CFG combined | Compound quality degradation | **Reverted** |
| bfloat16 on MPS | Not supported (as of 2026-02) | Not implemented |
| torch.compile (inductor) | MPS backend immature | Not implemented |
| ONNX Runtime CoreML EP | Variable text length issues | Not implemented |
| Latent Consistency Model | No TangoFlux distillation exists | Not implemented |
| Metal FlashAttention | Deep diffusers coupling | Not implemented |

## Model

- **TangoFlux** ([declare-lab/TangoFlux](https://huggingface.co/declare-lab/TangoFlux)): 515M parameter text-to-audio model based on FluxTransformer2DModel
- **T5 Encoder**: google/flan-t5-large for text conditioning
- **AutoencoderOobleck**: VAE for latent-to-waveform decoding
- **all-MiniLM-L6-v2**: 384-dim sentence embeddings for archive similarity search

## Requirements

- macOS 15+ (for CoreML ANE support)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~4GB RAM for model loading

## Citation

```bibtex
@misc{ochiai2026tangoflux_endless,
  author = {Ochiai, Yoichi},
  title = {TangoFlux Endless: Real-time Text-to-Audio Generation on Apple Silicon via CoreML-accelerated Flow Matching},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ochyai/TangoFlux-Endless}
}
```

## License

MIT License. See [LICENSE](LICENSE).

---

*Yoichi Ochiai — Digital Nature Group, University of Tsukuba / Pixie Dust Technologies*
