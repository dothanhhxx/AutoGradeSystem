import sys
from transformers import AutoModelForCausalLM
from transformers import logging as hf_logging
import torch

old_verbosity = hf_logging.get_verbosity()
hf_logging.set_verbosity_error()

print("Loading Qwen with suppressed logging", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16, trust_remote_code=True)
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print("Exception:", e, flush=True)
    
hf_logging.set_verbosity(old_verbosity)
