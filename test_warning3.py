import sys
print("Step 5: loading Qwen without trust_remote_code", flush=True)
from transformers import AutoModelForCausalLM
import torch
try:
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16)
except Exception as e:
    print("Exception:", e, flush=True)
