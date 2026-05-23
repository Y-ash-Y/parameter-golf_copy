"""
Int6 QAT patch for train_gpt_mlx.py
=====================================
Apply these changes to train_gpt_mlx.py in your parameter-golf_copy repo.

Changes:
  1. Add zstd import (falls back to zlib if not installed)
  2. Add fake_quantize_int6() for QAT during training
  3. Replace int8 export with int6-packed export (4 values per 3 bytes = 25% smaller)
  4. Apply QAT in CastedLinear forward pass

HOW TO APPLY
------------
This file documents the exact diffs. Make each change manually in your editor,
or run: python3 int6_qat_patch.py --apply /path/to/train_gpt_mlx.py

Expected result: same training, but model exports ~25% smaller → more params fit in 16MB.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Replace the zlib import (around line 17)
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_1_OLD = "import zlib"

CHANGE_1_NEW = """\
import zlib
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: Int6 QAT constants + fake_quantize function
# Add this block right BEFORE the line:
#   def quantize_float_array(arr: mx.array)
# (around line 575 in the original file)
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_2_INSERT_BEFORE = "def quantize_float_array(arr: mx.array)"

CHANGE_2_NEW = """\
# ─────────────────────────────────────────────────────────────────────────────
# INT6 QAT
# ─────────────────────────────────────────────────────────────────────────────
# Int6 signed range: [-31, 31]  (we reserve -32 to avoid asymmetry issues)
INT6_MAX = 31
INT6_MIN = -31

# QAT start step: begin fake-quantizing after warmup (helps stability)
QAT_START_STEP = int(os.environ.get("QAT_START_STEP", 2000))

# Global step counter used by QAT (set by training loop)
_current_step = 0


def fake_quantize_int6_row(w: mx.array) -> mx.array:
    \"\"\"
    Simulate int6 per-row quantization during training (STE).

    Forward pass : uses quantized weights  →  model 'sees' what it will get at export
    Backward pass: gradient flows straight through (STE)  →  training still works

    w must be 2D (rows x cols), matching CastedLinear.weight shape.
    \"\"\"
    # Per-row scale: map the max abs value in each row to INT6_MAX
    row_max = mx.max(mx.abs(w), axis=1, keepdims=True)          # (rows, 1)
    scale   = mx.maximum(row_max / INT6_MAX, 1e-8)               # (rows, 1)

    w_scaled  = w / scale                                         # float, in [-31, 31] range
    w_rounded = mx.round(mx.clip(w_scaled, INT6_MIN, INT6_MAX))  # integer values, float dtype

    # STE trick: in forward we get w_rounded * scale (quantized),
    # but gradient sees d(loss)/d(w) directly (no rounding in backward).
    # MLX way: add a term that is 0 in value but passes gradient through.
    w_q = mx.stop_gradient(w_rounded - w_scaled) + w_scaled      # STE on scaled
    return w_q * scale                                            # back to weight space


"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3: Replace quantize_float_array body with int6 version
# Find the function and replace its internals
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_3_OLD = """\
def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clip(min=1.0 / 127.0)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return q, scale.astype(np.float32)
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.float32(clip_abs / 127.0 if clip_abs > 0 else 1.0)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return q, np.array([scale], dtype=np.float32)"""

CHANGE_3_NEW = """\
def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"Quantize to int6 range [-31, 31], stored as int8 but packed at save time.\"\"\"
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        # Per-row: scale each row so max abs maps to INT6_MAX=31
        row_max = np.max(np.abs(f32), axis=1) if f32.size else np.zeros(f32.shape[0], dtype=np.float32)
        scale   = np.maximum(row_max / INT6_MAX, 1.0 / INT6_MAX).astype(np.float32)
        q = np.clip(np.round(f32 / scale[:, None]), INT6_MIN, INT6_MAX).astype(np.int8)
        return q, scale
    # Per-tensor (vectors, scalars)
    t_max = float(np.max(np.abs(f32))) if f32.size else 0.0
    scale = np.float32(max(t_max / INT6_MAX, 1.0 / INT6_MAX))
    q = np.clip(np.round(f32 / scale), INT6_MIN, INT6_MAX).astype(np.int8)
    return q, np.array([scale], dtype=np.float32)"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 4: Add int6 packing functions (insert after quantize_float_array)
# Find the line:  def quantize_state_dict_int8(
# and insert before it:
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_4_INSERT_BEFORE = "def quantize_state_dict_int8("

CHANGE_4_NEW = """\
def pack_int6(q: np.ndarray) -> np.ndarray:
    \"\"\"
    Pack int6 values (stored as int8, range [-31,31]) into 3 bytes per 4 values.
    Layout for values a,b,c,d (each 6-bit unsigned after +32 bias):
      byte0 = a[5:0] | b[5:4]<<6   (but we use simple bit layout below)
    We bias by 32 to make values unsigned [1,63], then pack 4 per 3 bytes.
    \"\"\"
    flat = q.reshape(-1).astype(np.int8)
    # Pad to multiple of 4
    pad = (-len(flat)) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    # Bias to unsigned [1, 63] — offset 32 means 0 maps to 32 (non-zero, compresses better)
    u = (flat.astype(np.int16) + 32).astype(np.uint8)  # values in [1, 63], 6 bits each
    # Pack 4 x 6-bit values into 3 bytes
    # v0[5:0] in bits 7:2 of byte0, v1[5:4] in bits 1:0 of byte0
    # v1[3:0] in bits 7:4 of byte1, v2[5:2] in bits 3:0 of byte1
    # v2[1:0] in bits 7:6 of byte2, v3[5:0] in bits 5:0 of byte2
    n4 = len(u) // 4
    v = u.reshape(n4, 4).astype(np.uint32)
    b0 = (v[:,0] << 2) | (v[:,1] >> 4)
    b1 = ((v[:,1] & 0xF) << 4) | (v[:,2] >> 2)
    b2 = ((v[:,2] & 0x3) << 6) | v[:,3]
    packed = np.stack([b0, b1, b2], axis=1).reshape(-1).astype(np.uint8)
    return packed, len(flat) - pad  # packed bytes, original count


def unpack_int6(packed: np.ndarray, count: int) -> np.ndarray:
    \"\"\"Unpack 3-bytes-per-4-values back to int8 array of length count.\"\"\"
    n4 = (count + 3) // 4
    b = packed[:n4 * 3].reshape(n4, 3).astype(np.uint32)
    v0 = (b[:,0] >> 2) & 0x3F
    v1 = ((b[:,0] & 0x3) << 4) | (b[:,1] >> 4)
    v2 = ((b[:,1] & 0xF) << 2) | (b[:,2] >> 6)
    v3 = b[:,2] & 0x3F
    u = np.stack([v0, v1, v2, v3], axis=1).reshape(-1)[:count].astype(np.uint8)
    return (u.astype(np.int16) - 32).astype(np.int8)


def compress_bytes(data: bytes) -> bytes:
    \"\"\"Compress with zstd level 22 if available, else zlib level 9.\"\"\"
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        return b'zstd:' + cctx.compress(data)
    return b'zlib:' + zlib.compress(data, level=9)


def decompress_bytes(data: bytes) -> bytes:
    if data[:5] == b'zstd:':
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data[5:])
    return zlib.decompress(data[5:])  # strip 'zlib:' prefix


"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 5: Update quantize_state_dict_int8 to use int6 packing
# Replace the format string and add packing
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_5_OLD = '        "__quant_format__": "int8_clean_per_row_v1",'

CHANGE_5_NEW = '        "__quant_format__": "int6_packed_per_row_v1",'

# Also update the quantized dict to store packed bytes instead of int8 arrays.
# Find where quantized[name] = q is set and also store packed version.
# The simplest change: after building the quantized dict, pack all 2D arrays.

CHANGE_5B_OLD = """\
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s"""

CHANGE_5B_NEW = """\
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            packed_data, orig_count = pack_int6(q)
            quantized[name] = packed_data          # store packed bytes
            qmeta[name]["orig_count"] = orig_count
            qmeta[name]["orig_shape"] = list(q.shape)
        else:
            quantized[name] = q                    # small tensors stay as int8
        scales[name] = s"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 6: Update dequantize_state_dict_int8 to unpack int6
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_6_OLD = """\
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)"""

CHANGE_6_NEW = """\
    for name, q_raw in quant_obj["quantized"].items():
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        meta = qmeta.get(name, {})
        if meta.get("scheme") == "per_row" and "orig_count" in meta:
            q_np = unpack_int6(np.asarray(q_raw, dtype=np.uint8), meta["orig_count"])
            q_np = q_np.reshape(meta["orig_shape"])
        else:
            q_np = np.asarray(q_raw, dtype=np.int8)"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 7: QAT in CastedLinear forward pass
# Find the CastedLinear class and patch its __call__ to fake-quantize when training
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_7_OLD = """\
class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T"""

CHANGE_7_NEW = """\
class CastedLinear(nn.Linear):
    # Keep weights in fp32, cast at matmul time. Apply int6 QAT when training.
    _qat_enabled: bool = False   # toggled on by training loop after QAT_START_STEP

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        if CastedLinear._qat_enabled and w.ndim == 2:
            w = fake_quantize_int6_row(w)
        return x @ w.T"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 8: Enable QAT flag in training loop
# Find the main training loop step increment and add QAT toggle.
# Look for where step is incremented, add this after:
#
#   if step == QAT_START_STEP:
#       CastedLinear._qat_enabled = True
#       log(f"qat:enabled at step {step}")
#
# The exact insertion point depends on your loop structure.
# Search for:  step += 1
# and add the toggle right after.
# ─────────────────────────────────────────────────────────────────────────────

# This one is context-dependent so just print the instruction:
CHANGE_8_INSTRUCTION = """
CHANGE 8 (manual): In the training loop, after 'step += 1', add:

    if step == QAT_START_STEP:
        CastedLinear._qat_enabled = True
        log(f"qat:enabled at step {step}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-APPLY SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

import sys
import re

def apply_patch(filepath: str):
    with open(filepath, 'r') as f:
        src = f.read()

    original = src
    changes_applied = []

    def replace(old, new, name):
        nonlocal src
        if old in src:
            src = src.replace(old, new, 1)
            changes_applied.append(f"  ✓ {name}")
        else:
            print(f"  ✗ FAILED: {name}")
            print(f"    Could not find:\n    {repr(old[:80])}")

    def insert_before(marker, new_text, name):
        nonlocal src
        idx = src.find(marker)
        if idx != -1:
            src = src[:idx] + new_text + src[idx:]
            changes_applied.append(f"  ✓ {name}")
        else:
            print(f"  ✗ FAILED: {name}")
            print(f"    Could not find marker: {repr(marker[:60])}")

    print(f"\nApplying int6 QAT patch to: {filepath}\n")

    replace(CHANGE_1_OLD, CHANGE_1_NEW, "Change 1: zstd import")
    insert_before(CHANGE_2_INSERT_BEFORE, CHANGE_2_NEW, "Change 2: fake_quantize_int6 function")
    replace(CHANGE_3_OLD, CHANGE_3_NEW, "Change 3: quantize_float_array → int6 range")
    insert_before(CHANGE_4_INSERT_BEFORE, CHANGE_4_NEW, "Change 4: pack_int6 / unpack_int6 / compress helpers")
    replace(CHANGE_5_OLD, CHANGE_5_NEW, "Change 5a: format string → int6_packed")
    replace(CHANGE_5B_OLD, CHANGE_5B_NEW, "Change 5b: use pack_int6 in quantize loop")
    replace(CHANGE_6_OLD, CHANGE_6_NEW, "Change 6: dequantize unpacks int6")
    replace(CHANGE_7_OLD, CHANGE_7_NEW, "Change 7: CastedLinear QAT forward")

    print("\n".join(changes_applied))
    print(CHANGE_8_INSTRUCTION)

    if src == original:
        print("\nNo changes made — all patterns failed. Check file version.")
        return

    # Write backup
    backup = filepath + ".bak"
    with open(backup, 'w') as f:
        f.write(original)
    print(f"\nBackup saved to: {backup}")

    with open(filepath, 'w') as f:
        f.write(src)
    print(f"Patched file written to: {filepath}")
    print("\nNext: pip install zstandard  (optional but recommended)")
    print("Then: re-run your smoke test to confirm it works.")

if __name__ == "__main__":
    if "--apply" in sys.argv:
        idx = sys.argv.index("--apply")
        path = sys.argv[idx + 1]
        apply_patch(path)
    else:
        print("Usage: python3 int6_qat_patch.py --apply /path/to/train_gpt_mlx.py")
        print("       Or read this file to apply changes manually.")
