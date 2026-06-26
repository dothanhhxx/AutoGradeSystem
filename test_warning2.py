import sys
print("Step 1: start", flush=True)

print("Step 2: importing transformers", flush=True)
import transformers

print("Step 3: importing AutoModelForCausalLM", flush=True)
from transformers import AutoModelForCausalLM

print("Step 4: importing torch", flush=True)
import torch

print("Step 5: loading Qwen", flush=True)
try:
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16, trust_remote_code=True)
except Exception as e:
    print("Exception:", e, flush=True)
