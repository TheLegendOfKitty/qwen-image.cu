#!/usr/bin/env python3
"""Verify our tokenizer matches the HuggingFace reference."""
from transformers import AutoTokenizer

MODEL_DIR = "/home/parsa/qwen-image.cu/Qwen-Image-2512/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

prompt = "a red rose"

# Manually construct the prompt like our C++ code
# <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
im_start = 151644
im_end = 151645

# Encode the text parts separately, then combine with special token IDs
user_tokens = tokenizer.encode("user\n" + prompt, add_special_tokens=False)
newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

cond_tokens = [im_start] + user_tokens + [im_end] + newline_tokens + [im_start] + assistant_tokens
print(f"Cond tokens ({len(cond_tokens)}): {cond_tokens}")
print(f"Decoded: '{tokenizer.decode(cond_tokens)}'")

# Uncond: empty prompt
user_empty = tokenizer.encode("user\n", add_special_tokens=False)
uncond_tokens = [im_start] + user_empty + [im_end] + newline_tokens + [im_start] + assistant_tokens
print(f"\nUncond tokens ({len(uncond_tokens)}): {uncond_tokens}")
print(f"Decoded: '{tokenizer.decode(uncond_tokens)}'")

# Verify simple tokenization
print(f"\nTokenize 'a red rose': {tokenizer.encode('a red rose', add_special_tokens=False)}")
print(f"Tokenize 'user\\n': {tokenizer.encode('user' + chr(10), add_special_tokens=False)}")
print(f"Tokenize 'assistant\\n': {tokenizer.encode('assistant' + chr(10), add_special_tokens=False)}")
print(f"Tokenize '\\n': {tokenizer.encode(chr(10), add_special_tokens=False)}")
