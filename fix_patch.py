# Save this as fix_patch.py in your project folder and run it

with open("train_gpt_mlx.py", "r") as f:
    src = f.read()

# Fix 1: CastedLinear QAT forward
old1 = """class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T"""

new1 = """class CastedLinear(nn.Module):
    _qat_enabled: bool = False  # toggled on after QAT_START_STEP

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        if CastedLinear._qat_enabled and w.ndim == 2:
            w = fake_quantize_int6_row(w)
        return x @ w.astype(x.dtype).T"""

# Fix 2: quantize_float_array → int6 range
old2 = """def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale"""

new2 = """def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Per-row int6: scale each row so max abs maps to 31
        row_max = np.max(np.abs(f32), axis=1) if f32.size else np.zeros(f32.shape[0], dtype=np.float32)
        scale = np.maximum(row_max / 31.0, 1.0 / 31.0).astype(np.float32)
        q = np.clip(np.round(f32 / scale[:, None]), -31, 31).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale)

    # Vectors / scalars: per-tensor int6
    t_max = float(np.max(np.abs(f32))) if f32.size else 0.0
    scale = np.float32(max(t_max / 31.0, 1.0 / 31.0))
    q = np.clip(np.round(f32 / scale), -31, 31).astype(np.int8)
    return np.ascontiguousarray(q), np.array([scale], dtype=np.float32)"""

assert old1 in src, "CastedLinear pattern not found!"
assert old2 in src, "quantize_float_array pattern not found!"

src = src.replace(old1, new1, 1)
src = src.replace(old2, new2, 1)

with open("train_gpt_mlx.py", "w") as f:
    f.write(src)

print("Both patches applied successfully!")