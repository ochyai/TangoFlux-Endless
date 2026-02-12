# Required Patches for diffusers

TangoFlux uses `FluxTransformer2DModel` from Hugging Face diffusers. The original implementation
uses operations incompatible with CoreML (Apple Neural Engine):

1. **`einsum` with ellipsis notation** in Rotary Position Embedding (RoPE)
2. **Rank-6 tensors** from the 2x2 rotation matrix representation

These patches refactor RoPE to use a separated (cos, sin) representation that stays within
CoreML's rank-5 tensor limit while maintaining mathematical equivalence.

## How to Apply

```bash
# Find your diffusers installation
DIFFUSERS_PATH=$(python -c "import diffusers; print(diffusers.__path__[0])")

# Apply patches
patch -p1 -d "$DIFFUSERS_PATH" < patches/transformer_flux.patch
patch -p1 -d "$DIFFUSERS_PATH" < patches/attention_processor.patch
```

## What Changes

### transformer_flux.py

**`rope()` function**: Returns `(cos, sin)` tuple instead of stacked rotation matrix.

```python
# BEFORE (original diffusers)
def rope(pos, dim, theta):
    out = torch.einsum("...n,d->...nd", pos, omega)       # einsum with ellipsis
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()                                      # rank 5+

# AFTER (CoreML-compatible)
def rope(pos, dim, theta):
    out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)  # basic tensor ops
    return torch.cos(out).float(), torch.sin(out).float()       # rank 3 each
```

**`EmbedND.forward()`**: Returns `(cos_emb, sin_emb)` tuple, each rank 4.

### attention_processor.py

**`apply_rope()`**: Direct rotation using separated cos/sin instead of 2x2 matrix multiply.

```python
# BEFORE
xq_out = (freqs_cis * xq_).flatten(3)  # matrix multiply with rank-6 tensor

# AFTER
xq_out = torch.stack([
    cos_emb * xq_r - sin_emb * xq_i,
    sin_emb * xq_r + cos_emb * xq_i
], dim=-1).flatten(-2)                  # direct rotation, max rank 5
```

## Verification

The refactored implementation is mathematically equivalent to the original.
Trace verification shows max diff = 0.00e+00 between original and patched outputs.
End-to-end audio generation produces max diff = 0.0126 (float16 quantization in CoreML).
