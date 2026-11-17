"""Minimal batched chat-generation wrapper (HF transformers, optional 4-bit)."""

from typing import Dict, List, Optional

import torch

from . import common  # noqa: F401  (HF cache env)


class ChatLLM:
    _instances: Dict[str, "ChatLLM"] = {}

    @classmethod
    def get(cls, model_id: str, four_bit: Optional[bool] = None) -> "ChatLLM":
        # Only one LLM fits in 8 GB VRAM alongside nothing else: unload others first.
        for k in list(cls._instances):
            if k != model_id:
                cls._instances.pop(k).unload()
        if model_id not in cls._instances:
            cls._instances[model_id] = cls(model_id, four_bit)
        return cls._instances[model_id]

    def __init__(self, model_id: str, four_bit: Optional[bool] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.model_id = model_id
        if four_bit is None:
            four_bit = "phi-3" in model_id.lower() or "7b" in model_id.lower()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to("cuda")
        self.model.eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    def _render(self, prompt: str, system: Optional[str]) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 128, temperature: float = 0.0,
                 seed: int = 0, system: Optional[str] = None) -> List[str]:
        texts = [self._render(p, system) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        torch.manual_seed(seed)
        gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**enc, **gen_kw)
        new = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(x, skip_special_tokens=True).strip() for x in new]
