import warnings
import traceback
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")

print("Loading model...")
from transformers import AutoModelForCausalLM
import torch

try:
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16, trust_remote_code=True)
except Exception as e:
    print("Exception:", e)
