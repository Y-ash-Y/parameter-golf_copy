with open("train_gpt.py", "r") as f:
    src = f.read()

old = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )"""

new = """        if os.environ.get("NO_COMPILE"):
            # Manual causal attention — no SDPA backend dependency
            # Used in eager mode on environments where SDPA dispatcher is broken
            scale = q.shape[-1] ** -0.5
            # Expand k,v heads to match q heads if GQA
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.ones(q.shape[-2], q.shape[-2],
                              dtype=torch.bool, device=q.device).tril()
            attn = attn.masked_fill(~mask, float('-inf'))
            attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )"""

assert old in src, "SDPA call not found — check exact whitespace"
src = src.replace(old, new, 1)

with open("train_gpt.py", "w") as f:
    f.write(src)
print("Manual attention fix applied")
