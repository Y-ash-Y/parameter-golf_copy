with open("train_gpt.py", "r") as f:
    src = f.read()

# When NO_COMPILE/eager mode, enable math as fallback
old = """    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)"""

new = """    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    # Enable math as fallback when flash backend unavailable (e.g. Colab T4 eager mode)
    enable_math_sdp(bool(os.environ.get("NO_COMPILE")))"""

assert old in src, "SDP backend block not found"
src = src.replace(old, new, 1)

with open("train_gpt.py", "w") as f:
    f.write(src)
print("SDP fix applied")
