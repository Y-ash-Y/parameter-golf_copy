with open("train_gpt.py", "r") as f:
    src = f.read()

# Fix 1: zeropower compile
old1 = "    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"
new1 = """    if not os.environ.get("NO_COMPILE"):
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"""
assert old1 in src, f"Fix1 not found"
src = src.replace(old1, new1, 1)

# Fix 2: model compile
old2 = "    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)"
new2 = """    if os.environ.get("NO_COMPILE"):
        compiled_model = base_model
    else:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)"""
assert old2 in src, f"Fix2 not found"
src = src.replace(old2, new2, 1)

with open("train_gpt.py", "w") as f:
    f.write(src)
print("Both compile fixes applied")

# Verify
with open("train_gpt.py", "r") as f:
    check = f.read()
assert 'NO_COMPILE' in check
print("Verified NO_COMPILE guard present")
