import sys
print("Loading Qwen with ONLY dtype as string", flush=True)
from transformers import AutoModelForCausalLM
import torch
try:
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', dtype="float16", low_cpu_mem_usage=True)
    print("Model loaded successfully. dtype:", model.dtype, flush=True)
except Exception as e:
    print("Exception:", e, flush=True)
