# Technical Report: Real-time Endless Audio Generation on Apple Silicon

**TangoFlux Endless — CoreML-accelerated text-to-audio with seamless playback**

## Abstract

We present a system for real-time, continuous soundscape generation from text prompts on Apple Silicon. By converting the FluxTransformer2DModel (515M parameters) from TangoFlux to CoreML and executing it on Apple Neural Engine/GPU, we achieve a Real-Time Factor (RTF) of 0.29x — generating 18 seconds of audio in 5.2 seconds. The system combines subprocess-based GIL isolation, embedding-similarity archive fallback, and N-layer sin²/cos² crossfade mixing to produce seamless, endless audio output. We document the CoreML conversion process, including novel solutions for RoPE compatibility (einsum elimination, rank reduction), and report negative results on MPS float16 inference.

## 1. Introduction

Text-to-audio generation models have reached a quality level suitable for practical applications, but real-time deployment on consumer hardware remains challenging. TangoFlux, a 515M parameter model based on the Flux architecture, generates high-quality audio from text descriptions but requires ~8.6 seconds on Apple MPS (GPU) for 18 seconds of audio at 25 denoising steps. While this is technically faster than real-time (RTF 0.48x), the margin is insufficient for reliable continuous generation when accounting for text encoding, VAE decoding, and I/O overhead.

This report details our approach to building a production-ready endless audio generation system on Apple Silicon, with particular focus on:

1. CoreML conversion of the Flux transformer architecture, requiring non-trivial modifications to the Rotary Position Embedding (RoPE) implementation
2. Process-level isolation to eliminate Python GIL contention between inference and audio playback
3. Comprehensive benchmarking of optimization strategies, including documentation of approaches that failed

## 2. System Architecture

### 2.1 Design Constraints

The target platform is macOS on Apple Silicon (M1-M4), running Python 3.10 with PyTorch. The system must:

- Generate audio continuously without perceptible gaps or artifacts
- Allow real-time prompt changes during playback
- Degrade gracefully when generation cannot keep up with playback
- Support variable clip durations (5-30 seconds) and crossfade configurations

### 2.2 Process Model

```
Main Process (PyQt6 GUI)
├── Audio OutputStream (sounddevice callback, 44.1kHz mono, blocksize=2048)
├── Layer Mixer (N-layer crossfade, sin²/cos² envelope)
├── Feeder Thread (file watcher + archive fallback)
└── Worker Monitor Thread (stdout reader)

Worker Process (subprocess.Popen)
├── T5 Encoder (PyTorch, MPS/CPU)
├── FluxTransformer2DModel (CoreML, ANE/GPU)
├── AutoencoderOobleck VAE (PyTorch, MPS)
├── Embedding Model (all-MiniLM-L6-v2, CPU)
└── Archive Manager (index.json + .npy files)

IPC: .npy file exchange via temp directory
     command.json for control signals (update/stop)
```

**Rationale for subprocess isolation:** Python's GIL serializes CPU-bound work (PyTorch inference) and I/O-bound work (sounddevice audio callbacks). In our initial threaded implementation, GIL contention caused the audio callback to be delayed by up to 50ms during inference steps, producing audible click/pop artifacts. Moving inference to a separate process via `subprocess.Popen` eliminates GIL sharing entirely. The cost is IPC overhead (~1ms for .npy file read), which is negligible compared to the 2048-sample audio callback period (~46ms at 44.1kHz).

### 2.3 N-layer Staggered Playback

The mixer maintains N simultaneously playing audio clips (default N=3), each with a pre-computed sin²/cos² amplitude envelope:

```
Fade-in:   envelope[0:F]     = sin²(t),  t ∈ [0, π/2]
Sustain:   envelope[F:L-F]   = 1.0
Fade-out:  envelope[L-F:L]   = cos²(t),  t ∈ [0, π/2]
```

Initial clips are loaded with staggered positions to ensure immediate crossfade overlap:

```
Layer 1: position = (N-1)/N × clip_length    (most advanced)
Layer 2: position = (N-2)/N × clip_length
  ...
Layer N: position = 0                        (just starting)
```

When any layer reaches its trigger point (1/N of total length), the next clip is injected. The audio callback sums all active layers and divides by N:

```python
mixed = sum(layer.read(frames) for layer in active_layers)
mixed /= N  # prevents clipping: N layers × 0.75 peak = 0.75/N per layer
```

This produces a continuous, smoothly varying output without the harsh distortion caused by hard clipping (`np.clip(mixed, -1, 1)`).

### 2.4 Archive Fallback

When the generator cannot produce clips fast enough (e.g., during model loading, system load spikes, or prompt changes that require re-encoding), the system falls back to previously archived clips selected by semantic similarity.

The archive stores up to 200 clips with associated metadata:

```json
{
  "id": "e683bd2d13db",
  "filename": "e683bd2d13db.npy",
  "prompt": "gentle rain with distant thunder",
  "embedding": [0.0234, -0.0156, ...],  // 384-dim, L2-normalized
  "duration": 18,
  "timestamp": 1707782400.0
}
```

Retrieval uses cosine similarity between the current prompt embedding and archived embeddings, selecting randomly from the top-3 matches:

```python
similarity = np.dot(current_embedding, archive_embedding)  # both L2-normalized
candidates = sorted(archive, key=similarity, reverse=True)[:3]
selected = random.choice(candidates)
```

The top-3 random selection prevents repetitive playback of the single best match. Recently used clips are tracked and excluded to further increase variety.

## 3. CoreML Conversion

### 3.1 Motivation

The FluxTransformer2DModel is the computational bottleneck, consuming ~85% of total inference time. Apple's Neural Engine (ANE) is designed for transformer-style matrix multiplications and can significantly outperform the GPU (MPS) for compatible models.

### 3.2 Conversion Challenges

#### Challenge 1: einsum with Ellipsis Notation

The original RoPE implementation in diffusers uses `torch.einsum` with ellipsis notation:

```python
# diffusers/models/transformers/transformer_flux.py, line 46
out = torch.einsum("...n,d->...nd", pos, omega)
```

`coremltools` 8.x-9.x cannot convert einsum operations with ellipsis (`...`) subscripts. The error occurs during the MIL (Model Intermediate Language) conversion pass:

```
ERROR - converting 'einsum' op (located at: 'transformer/pos_embed/out.1')
```

**Solution:** Replace with equivalent basic tensor operations:

```python
# Equivalent: outer product via broadcasting
out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
```

This produces identical results (verified: max diff = 0.00e+00) while using only `unsqueeze` and `multiply`, both fully supported by CoreML.

#### Challenge 2: Rank-6 Tensor

CoreML supports tensors up to rank 5. The original RoPE creates a 2×2 rotation matrix representation that requires rank 6:

```python
# Original flow:
stacked = torch.stack([cos, -sin, sin, cos], dim=-1)  # rank N+1
out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # rank 5
# In EmbedND:
return emb.unsqueeze(1)  # rank 5 → rank 6!
```

The CoreML conversion fails:

```
Core ML only supports tensors with rank <= 5.
Layer "freqs_cis_cast_fp16", with type "expand_dims", outputs a rank 6 tensor.
```

**Solution:** Refactor the entire RoPE pipeline to use separated (cos, sin) representation:

1. `rope()` returns `(cos_tensor, sin_tensor)` — each rank 3
2. `EmbedND.forward()` returns `(cos_emb, sin_emb)` — each rank 4
3. `apply_rope()` performs direct rotation instead of matrix multiply

```python
# rope(): returns separated cos/sin
def rope(pos, dim, theta):
    out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    return torch.cos(out).float(), torch.sin(out).float()

# EmbedND: concatenates along feature dim, adds head dim
cos_emb = torch.cat(cos_parts, dim=-1).unsqueeze(1)  # rank 4
sin_emb = torch.cat(sin_parts, dim=-1).unsqueeze(1)  # rank 4

# apply_rope(): direct rotation via cos/sin
xq_out = torch.stack([
    cos_emb * xq_r - sin_emb * xq_i,
    sin_emb * xq_r + cos_emb * xq_i
], dim=-1).flatten(-2)  # max rank 5
```

The mathematical equivalence holds because the 2×2 rotation matrix `[[cos, -sin], [sin, cos]]` applied to vector `[r, i]` gives `[cos*r - sin*i, sin*r + cos*i]`, which is exactly what the direct formulation computes.

### 3.3 Conversion Configuration

```python
ct.convert(
    traced_model,
    inputs=[
        ct.TensorType("hidden_states",   shape=(2, 645, 64)),    # CFG batch
        ct.TensorType("timestep",         shape=(1,)),
        ct.TensorType("pooled_projections", shape=(2, 1024)),
        ct.TensorType("encoder_hidden_states", shape=(2, 65, 1024)),
        ct.TensorType("txt_ids",          shape=(2, 65, 3)),
        ct.TensorType("img_ids",          shape=(2, 645, 3)),
    ],
    outputs=[ct.TensorType(name="noise_pred")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
    convert_to="mlprogram",
)
```

Key decisions:

- **Fixed shapes**: Text is padded to `max_length=64` (+ 1 duration token = 65). This enables ANE graph optimization. Variable-length sequences would force CPU/GPU fallback.
- **Batch size = 2**: For classifier-free guidance (unconditional + conditional). The model always runs both branches.
- **FLOAT16 precision**: CoreML's internal float16 is stable (unlike MPS float16 — see Section 4). The runtime manages precision boundaries automatically.
- **macOS 15 target**: Required for latest ANE optimizations on M3/M4.

### 3.4 Hybrid Pipeline Integration

The CoreML model handles only the transformer denoising step. T5 encoding and VAE decoding remain in PyTorch:

```
T5 Encoder (MPS float32) → numpy float16 → CoreML Transformer → numpy float32 → PyTorch Scheduler (CPU) → loop × 25 → VAE (MPS float32)
```

Precision boundaries:
- MPS float32 → numpy float16: Input quantization for CoreML
- CoreML float16 → numpy float32: Output dequantization for scheduler
- Scheduler operates in float32 to prevent accumulated precision errors

## 4. Negative Results

### 4.1 float16 on MPS — Electrical Noise Artifacts

**Hypothesis:** Converting model parameters to float16 on MPS would reduce memory bandwidth and improve throughput.

**Implementation:** Applied `.half()` to the transformer, FC projection, and duration embedder. Patched `inference_flow()` to propagate `weight_dtype` to all intermediate tensor creation (replacing hardcoded float32 in `torch.randn()`, `torch.zeros()`, `torch.tensor(float("nan"))`, etc.).

**Result:** Audible high-frequency electrical noise overlaid on generated audio. The noise was consistent and clearly artificial — not a random artifact.

**Root cause:** The flow-matching scheduler performs iterative computation over 25 steps. At each step, small precision errors in the float16 noise prediction are integrated into the latent state. Over 25 steps, these errors accumulate beyond perceptual thresholds. The `torch.tensor(float("nan"))` used in the masked mean computation is particularly problematic — float16 NaN handling differs subtly from float32.

**Contrast with CoreML:** CoreML's internal float16 uses `compute_precision=FLOAT16`, but the runtime applies mixed-precision strategies (e.g., keeping certain accumulations in float32) that prevent error accumulation. This is not configurable from Python — it's an implementation detail of the CoreML runtime.

### 4.2 No-CFG (guidance_scale=0) — Weak Text Adherence

**Hypothesis:** Disabling classifier-free guidance would halve transformer forward passes per step (1× instead of 2×), providing a 2× speedup on the denoising loop.

**Result:** Generated audio was significantly less responsive to text prompts. Environmental sounds became generic and lacked the specific characteristics described in the prompt (e.g., "thunder" was absent from "rain with thunder").

**Analysis:** TangoFlux appears to rely heavily on CFG for prompt adherence. The unconditional branch provides a "baseline" that the conditional branch must deviate from — without this contrast, the model defaults to generic audio patterns. This is consistent with findings in text-to-image diffusion models, where CFG is similarly critical for prompt following.

### 4.3 float16 + No-CFG Combined

When both optimizations were applied simultaneously, quality degradation was severe — both float16 artifacts and weak text adherence compounded. Individual testing confirmed float16 as the primary source of audible artifacts.

### 4.4 Other Investigated Approaches

| Approach | Finding |
|----------|---------|
| **bfloat16 on MPS** | Not supported by MPS backend (as of macOS 15, Feb 2026) |
| **torch.compile (inductor)** | MPS Inductor backend immature. `aot_eager` provides 15-25% improvement but requires ~120s warmup — unsuitable for interactive use |
| **ONNX Runtime CoreML EP** | Cannot handle variable text sequence lengths. Fixed-shape workaround has ANE fallback issues |
| **Latent Consistency Model** | No distilled model exists for TangoFlux. Training one requires the original training data and infrastructure |
| **Metal FlashAttention** | Deep coupling with diffusers attention processor implementation. Patch surface too large for reliable maintenance |

## 5. Benchmark Results

### 5.1 End-to-End Generation (25 steps, 18s audio)

**CoreML (deployed configuration):**

| Metric | Value |
|--------|-------|
| Mean | 5.2s |
| Min | 4.7s |
| Max | 5.9s |
| Std | 0.3s |
| RTF | 0.29x |
| Samples | 26 clips |

**MPS baseline:**

| Metric | Value |
|--------|-------|
| Mean | 8.6s |
| Min | 8.0s |
| Max | 9.4s |
| RTF | 0.48x |

### 5.2 Transformer-Only (25 steps, single forward pass per step)

| Backend | Total | Per step | Speedup |
|---------|-------|----------|---------|
| CoreML (ANE/GPU) | 3.59s | 144ms | **1.94x** |
| MPS (float32) | 6.98s | 279ms | baseline |

The higher speedup in isolated transformer benchmarks (1.94x) versus end-to-end (1.65x) is expected — T5 encoding and VAE decoding are not accelerated by CoreML and contribute constant overhead.

### 5.3 Step Count vs. Quality Trade-off

| Steps | MPS Time | RTF | Quality Note |
|-------|----------|-----|-------------|
| 50 | 13.7s | 0.76x | Original default, diminishing returns |
| **25** | **7.5s** | **0.42x** | **Current default, good balance** |
| 15 | 4.9s | 0.27x | Slight quality loss, viable for preview |

### 5.4 Duration Independence

Audio duration does not affect transformer compute cost. TangoFlux uses a fixed `audio_seq_len = 645` regardless of requested duration (5s-30s). Duration only affects VAE output clipping. This is because the model generates a fixed-length latent representation and the VAE decoder produces the full-length waveform, which is then truncated.

| Duration | 25-step Time | Difference |
|----------|-------------|------------|
| 5s | 7.5s | baseline |
| 10s | 7.5s | ±0.1s |
| 30s | 7.5s | ±0.1s |

### 5.5 CoreML Conversion Metrics

| Metric | Value |
|--------|-------|
| Conversion time | 41.3s |
| Model size (.mlpackage) | ~1GB |
| Numerical diff (MPS vs CoreML) | max 0.0126 |
| Trace accuracy (orig vs traced) | 0.00e+00 |

The max diff of 0.0126 is due to float16 quantization inside CoreML and is below perceptual thresholds for audio generation.

## 6. Model Architecture Reference

| Component | Model | Parameters | Backend |
|-----------|-------|------------|---------|
| Text Encoder | google/flan-t5-large | 248M | MPS |
| Transformer | FluxTransformer2DModel | 515M | **CoreML** |
| VAE | AutoencoderOobleck | 46M | MPS |
| Embeddings | all-MiniLM-L6-v2 | 22M | CPU |
| **Total** | | **831M** | |

TangoFlux config:
- `num_layers`: 1 (joint attention blocks)
- `num_single_layers`: 32 (single attention blocks)
- `attention_head_dim`: 128
- `num_attention_heads`: 8
- `joint_attention_dim`: 1024
- `in_channels`: 64
- `audio_seq_len`: 645 (fixed, regardless of duration)
- `max_duration`: 30s

## 7. Deployed Optimization Stack

| Layer | Optimization | Impact |
|-------|-------------|--------|
| Compute | CoreML ANE/GPU transformer | 1.65x end-to-end speedup |
| Scheduling | 25 denoising steps (from 50) | 2.0x speedup |
| Process | subprocess.Popen GIL isolation | Zero audio artifacts |
| Playback | N-layer sin²/cos² crossfade | Seamless transitions |
| Resilience | Embedding-similarity archive fallback | No buffer underrun gaps |
| Audio | Gain normalization (÷N layers) | No clipping distortion |

Combined effective speedup over naive implementation: **~3.3x** (CoreML 1.65x × steps 2.0x), with qualitative improvements from process isolation and crossfade mixing.

## 8. Conclusions

Real-time audio generation from TangoFlux on Apple Silicon is achievable through a combination of CoreML acceleration, step reduction, and careful systems engineering. The key insight is that CoreML's internal float16 management is fundamentally more robust than manual float16 conversion on MPS — attempting the latter produces audible artifacts that CoreML handles transparently.

The RoPE refactoring required for CoreML compatibility (einsum elimination + rank reduction) is general and applicable to any Flux-based model targeting CoreML deployment. We provide the patches as standalone diffs for reuse.

The subprocess architecture, while adding IPC complexity, completely solves the GIL contention problem that plagues Python audio applications using ML inference. The ~1ms overhead per clip transfer is negligible compared to the ~46ms audio callback period.

Future work includes:
- Streaming generation (start playback before full clip generation completes)
- Multi-prompt interpolation in the latent space
- Upstream contribution of CoreML-compatible RoPE to diffusers

---

*Yoichi Ochiai, 2026*
*Digital Nature Group, University of Tsukuba / Pixie Dust Technologies, Inc.*
