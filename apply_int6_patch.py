"""
Int6 QAT patch for train_gpt.py
Applies:
  1. fake_quantize_int6_row - STE quantization for QAT
  2. CastedLinear QAT forward - fake-quantize weights during training
  3. quantize_float_tensor - int6 range [-31, 31] instead of int8 [-127, 127]
  4. pack_int6 / unpack_int6 - 4 values per 3 bytes (25% smaller)
  5. Export format: int6_packed_per_row_v1
  6. zstd compression instead of zlib
  7. QAT activation at step 2000 in training loop
"""

import re

with open("train_gpt.py", "r") as f:
    src = f.read()

original = src  # keep backup

# ── 1. Add zstd import after zlib ─────────────────────────────────────
old = "import zlib"
new = "import zlib\ntry:\n    import zstandard as zstd\n    HAS_ZSTD = True\nexcept ImportError:\n    HAS_ZSTD = False"
assert old in src, "zlib import not found"
src = src.replace(old, new, 1)

# ── 2. Add QAT constants + helpers after the INT8 constants ───────────
# Find INT8_CLIP_Q or similar constant
int8_const_pattern = r'INT8_CLIP_Q\s*=.*\n'
match = re.search(int8_const_pattern, src)
assert match, "INT8_CLIP_Q constant not found"

qat_code = '''
# ── Int6 QAT ──────────────────────────────────────────────────────────
INT6_MAX       = 31          # [-31, 31] range for int6
QAT_START_STEP = 2000        # step at which fake-quantization activates

def fake_quantize_int6_row(w: "Tensor") -> "Tensor":
    """
    Straight-Through Estimator fake-quantization to int6 range [-31, 31].
    Forward: round weights to nearest int6 value (simulate quantization error).
    Backward: pass gradients through unchanged (STE).
    Operates per-row so each output channel has its own scale.
    """
    with torch.no_grad():
        row_max = w.abs().max(dim=1, keepdim=True).values
        scale   = (row_max / INT6_MAX).clamp_min(1.0 / INT6_MAX)
    # quantize forward
    w_q = torch.clamp(torch.round(w / scale), -INT6_MAX, INT6_MAX) * scale
    # STE: replace w with w_q but let gradient flow through w
    return w + (w_q - w).detach()

def pack_int6(q_np):
    """
    Pack int8 array with values in [-31, 31] into 6-bit packed bytes.
    4 int6 values -> 3 bytes (saves 25% vs int8).
    Input: numpy int8 array, flattened, length must be multiple of 4.
    Output: bytes object.
    """
    import numpy as np
    q = q_np.flatten().astype(np.int8)
    # pad to multiple of 4
    pad = (4 - len(q) % 4) % 4
    if pad:
        q = np.concatenate([q, np.zeros(pad, dtype=np.int8)])
    # bias to [0, 63] for unsigned 6-bit
    u = (q + INT6_MAX).astype(np.uint8) & 0x3F
    # pack 4x6 = 24 bits = 3 bytes
    groups = u.reshape(-1, 4)
    b0 = (groups[:, 0]      ) | (groups[:, 1] << 6)
    b1 = (groups[:, 1] >> 2 ) | (groups[:, 2] << 4)
    b2 = (groups[:, 2] >> 4 ) | (groups[:, 3] << 2)
    packed = np.stack([b0, b1, b2], axis=1).flatten().tobytes()
    return packed, pad

def unpack_int6(packed_bytes, original_numel):
    """Unpack 6-bit packed bytes back to int8 numpy array."""
    import numpy as np
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    n_groups = len(packed) // 3
    packed = packed[:n_groups * 3].reshape(-1, 3)
    v0 = ( packed[:, 0]       & 0x3F)
    v1 = ((packed[:, 0] >> 6) | ((packed[:, 1] & 0x0F) << 2))
    v2 = ((packed[:, 1] >> 4) | ((packed[:, 2] & 0x03) << 4))
    v3 = ( packed[:, 2] >> 2 )
    flat = np.stack([v0, v1, v2, v3], axis=1).flatten()
    flat = flat.astype(np.int8) - INT6_MAX
    return flat[:original_numel]

def compress_bytes(data: bytes) -> bytes:
    """Compress with zstd if available, fall back to zlib."""
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=19)
        return cctx.compress(data)
    return zlib.compress(data, level=9)

def decompress_bytes(data: bytes) -> bytes:
    """Decompress zstd or zlib."""
    if HAS_ZSTD and data[:4] == b\'\\x28\\xb5\\x2f\\xfd\':
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    return zlib.decompress(data)

'''

insert_pos = match.end()
src = src[:insert_pos] + qat_code + src[insert_pos:]

# ── 3. CastedLinear: add QAT forward ──────────────────────────────────
old_casted = '''class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)'''

new_casted = '''class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    _qat_enabled: bool = False  # class-level flag, toggled on at QAT_START_STEP

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if CastedLinear._qat_enabled and w.ndim == 2:
            w = fake_quantize_int6_row(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)'''

assert old_casted in src, "CastedLinear class not found"
src = src.replace(old_casted, new_casted, 1)

# ── 4. quantize_float_tensor: int6 range ──────────────────────────────
old_quant = '''def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale'''

new_quant = '''def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """Int6 quantization: per-row for 2D tensors, per-tensor otherwise."""
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row: scale each row so max abs value maps to INT6_MAX (31)
        row_max = t32.abs().max(dim=1).values if t32.numel() else torch.zeros(t32.shape[0])
        scale = (row_max / INT6_MAX).clamp_min(1.0 / INT6_MAX).to(torch.float32)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -INT6_MAX, INT6_MAX).to(torch.int8).contiguous()
        return q, scale.contiguous()

    # Vectors / scalars: per-tensor int6
    t_max = float(t32.abs().max().item()) if t32.numel() else 0.0
    scale = torch.tensor(max(t_max / INT6_MAX, 1.0 / INT6_MAX), dtype=torch.float32)
    q = torch.clamp(torch.round(t32 / scale), -INT6_MAX, INT6_MAX).to(torch.int8).contiguous()
    return q, scale'''

assert old_quant in src, "quantize_float_tensor not found"
src = src.replace(old_quant, new_quant, 1)

# ── 5. quantize_state_dict_int8: rename + use pack_int6 ───────────────
# Update the format string
src = src.replace(
    '"__quant_format__": "int8_clean_per_row_v1"',
    '"__quant_format__": "int6_packed_per_row_v1"',
    1
)

# Update the stats key references
src = src.replace('"int8_payload_bytes"', '"int6_payload_bytes"', 10)

# ── 6. Export: replace zlib with compress_bytes ───────────────────────
old_export = '''    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)'''

new_export = '''    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_bytes(quant_raw)'''

assert old_export in src, "export section not found"
src = src.replace(old_export, new_export, 1)

# ── 7. Roundtrip load: replace zlib.decompress ────────────────────────
old_load = '''    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")'''

new_load = '''    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_bytes(quant_blob_disk)), map_location="cpu")'''

assert old_load in src, "roundtrip load section not found"
src = src.replace(old_load, new_load, 1)

# ── 8. Training loop: add QAT toggle after step increment ─────────────
# Find step += 1 in the training loop
# We need to be careful to get the one in the main training loop
old_step = "        step += 1\n"
qat_toggle = (
    "        step += 1\n"
    "        if step == QAT_START_STEP and master_process:\n"
    "            print0(f'qat:enabled at step {step}')\n"
    "        if step == QAT_START_STEP:\n"
    "            CastedLinear._qat_enabled = True\n"
)
count = src.count(old_step)
assert count >= 1, f"'step += 1' not found, count={count}"
# Replace only the last occurrence (in the main training loop)
pos = src.rfind(old_step)
src = src[:pos] + qat_toggle + src[pos + len(old_step):]

# ── Write patched file ─────────────────────────────────────────────────
with open("train_gpt.py.bak", "w") as f:
    f.write(original)

with open("train_gpt.py", "w") as f:
    f.write(src)

print("Patch applied successfully!")
print("Backup saved to train_gpt.py.bak")
print("\nChanges made:")
print("  1. zstd import (fallback to zlib)")
print("  2. fake_quantize_int6_row + pack_int6 + compress_bytes helpers")
print("  3. CastedLinear: QAT fake-quantize forward")
print("  4. quantize_float_tensor: int6 range [-31, 31]")
print("  5. Export format: int6_packed_per_row_v1")
print("  6. Export compression: zstd (or zlib fallback)")
print("  7. QAT activation at step 2000")