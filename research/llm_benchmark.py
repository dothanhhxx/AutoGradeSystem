"""
llm_benchmark.py — LLMComparativeBenchmark for HybridASAGGrader research.

WHY THIS EXISTS (vs old code):
  Old: Qwen2.5-3B is hardcoded. No way to swap LLMs without editing grader.py.
  New: Load each LLM one at a time, run the same prompt set, measure:
       JSON compliance, tag accuracy, latency, RAM peak, parse error rate.

USAGE:
  configs = [
      {"name": "Qwen2.5-3B",  "model_id": "Qwen/Qwen2.5-3B-Instruct",       "use_4bit": False},
      {"name": "Qwen2.5-1.5B","model_id": "Qwen/Qwen2.5-1.5B-Instruct",     "use_4bit": False},
      {"name": "Phi-3-Mini",  "model_id": "microsoft/Phi-3-mini-4k-instruct","use_4bit": False},
  ]
  bench = LLMComparativeBenchmark(output_dir="./results/llm_bench")
  df = bench.benchmark_all(configs, samples, reference_tags)
  df.to_csv("llm_comparison.csv")
"""

import gc
import json
import os
import re
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm


# Prompt template — identical to grader.py so comparison is fair
_SYSTEM_PROMPT = """You are an expert educational assessment AI. Analyze the student's answer and provide feedback in valid JSON format only.

Output ONLY a JSON object with these exact keys:
{
  "tags": ["tag1", "tag2"],
  "explanation": "Brief explanation of the grade",
  "suggestion": "Specific suggestion for improvement"
}

Valid tags: Correct, Partially Correct, Incorrect, Missing Concepts, Factual Error, Logical Error, Vague Expression, Grammar Error, Off-Topic, Incomplete"""


def _build_prompt(question: str, reference: str, student: str, tags: List[str]) -> str:
    return f"""Question: {question}
Reference Answer: {reference}
Student Answer: {student}
Preliminary Tags (from rule-based system): {', '.join(tags)}

Provide feedback JSON:"""


class LLMComparativeBenchmark:
    """
    Benchmark multiple LLM backends on the same prompt set.

    Loads one model at a time and explicitly frees memory between runs
    to prevent OOM accumulation on CPU/low-VRAM systems.
    """

    def __init__(self, output_dir: str = "./results/llm_bench"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark_all(
        self,
        model_configs: List[Dict[str, Any]],
        samples: List,
        reference_tags: List[List[str]],
        max_new_tokens: int = 400,
    ) -> pd.DataFrame:
        """
        Run the benchmark across all model configs.

        Args:
            model_configs: List of dicts with keys:
                name (str), model_id (str), use_4bit (bool, optional)
            samples: List[ASAGSample]
            reference_tags: List of tag lists from rule-based `_assign_tags()`
                            (the "ground truth" tags without LLM synthesis)
            max_new_tokens: Max tokens to generate per sample.

        Returns:
            pd.DataFrame with one row per model_config.
        """
        rows = []

        for cfg in model_configs:
            name = cfg["name"]
            model_id = cfg["model_id"]
            use_4bit = cfg.get("use_4bit", False)

            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}  ({model_id})")
            print(f"  4-bit quantization: {use_4bit}")
            print(f"{'='*60}")

            row = self._benchmark_one(
                name=name,
                model_id=model_id,
                use_4bit=use_4bit,
                samples=samples,
                reference_tags=reference_tags,
                max_new_tokens=max_new_tokens,
            )
            rows.append(row)

            # Save incremental results so we don't lose data if a later model crashes
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(
                os.path.join(self.output_dir, "llm_benchmark_partial.csv"), index=False
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "llm_benchmark_final.csv"), index=False)
        print(f"\n[LLMBenchmark] Results saved to {self.output_dir}")
        return df

    # ------------------------------------------------------------------
    # Per-model benchmark
    # ------------------------------------------------------------------

    def _benchmark_one(
        self,
        name: str,
        model_id: str,
        use_4bit: bool,
        samples: List,
        reference_tags: List[List[str]],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """Load one model, run all samples, measure metrics, then free memory."""

        from transformers import AutoTokenizer, AutoModelForCausalLM

        # --- Load ---
        load_start = time.time()
        tracemalloc.start()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )

            if use_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to(device)

            model.eval()
            load_time_s = time.time() - load_start

        except Exception as exc:
            tracemalloc.stop()
            print(f"  ERROR loading model: {exc}")
            return {
                "model_name": name,
                "model_id": model_id,
                "use_4bit": use_4bit,
                "load_error": str(exc),
                "json_compliance_rate": 0.0,
                "tag_accuracy": 0.0,
                "avg_latency_s": None,
                "peak_ram_mb": None,
                "parse_error_rate": 1.0,
                "load_time_s": None,
                "hallucination_rate": None,
            }

        # --- Inference ---
        latencies = []
        parse_successes = []
        tag_matches = []
        hallucinations = []

        for sample, ref_tags in tqdm(
            zip(samples, reference_tags),
            total=len(samples),
            desc=f"  {name}",
        ):
            prompt = _build_prompt(
                sample.question,
                sample.reference_answer,
                sample.student_answer,
                ref_tags,
            )

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            t_start = time.time()
            try:
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt")
                else:
                    inputs = tokenizer(prompt, return_tensors="pt")

                # Move to model device
                device_for_input = next(model.parameters()).device
                inputs = {k: v.to(device_for_input) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            except Exception as exc:
                raw = ""
                print(f"    Generate error: {exc}")

            latencies.append(time.time() - t_start)

            # Parse JSON
            parsed, parse_ok = self._parse_json_response(raw)
            parse_successes.append(parse_ok)

            if parse_ok and parsed:
                # Tag accuracy: overlap with reference tags (Jaccard)
                pred_tags = set(parsed.get("tags", []))
                ref_set = set(ref_tags)
                if ref_set:
                    jaccard = len(pred_tags & ref_set) / len(pred_tags | ref_set)
                else:
                    jaccard = 1.0 if not pred_tags else 0.0
                tag_matches.append(jaccard)

                # Hallucination: check for claims not grounded in input
                halluc = self._estimate_hallucination(
                    parsed.get("explanation", ""),
                    sample.reference_answer,
                    sample.student_answer,
                )
                hallucinations.append(halluc)
            else:
                tag_matches.append(0.0)
                hallucinations.append(None)

        # --- Peak RAM ---
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_bytes / (1024 * 1024)

        # --- Cleanup ---
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "model_name": name,
            "model_id": model_id,
            "use_4bit": use_4bit,
            "load_time_s": round(load_time_s, 2),
            "json_compliance_rate": round(sum(parse_successes) / len(parse_successes), 4),
            "parse_error_rate": round(1 - sum(parse_successes) / len(parse_successes), 4),
            "tag_accuracy": round(sum(tag_matches) / len(tag_matches), 4),
            "avg_latency_s": round(sum(latencies) / len(latencies), 3),
            "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 3),
            "peak_ram_mb": round(peak_ram_mb, 1),
            "hallucination_rate": (
                round(sum(h for h in hallucinations if h is not None)
                      / max(1, sum(1 for h in hallucinations if h is not None)), 4)
                if hallucinations else None
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> Tuple[Optional[Dict], bool]:
        """Extract and parse the first JSON object from model output."""
        # Try to find ```json ... ``` block first
        code_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if code_match:
            raw = code_match.group(1).strip()

        # Find first { ... }
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return None, False

        try:
            obj = json.loads(json_match.group())
            # Validate expected keys
            if "tags" in obj and "explanation" in obj:
                return obj, True
            return obj, False
        except json.JSONDecodeError:
            return None, False

    @staticmethod
    def _estimate_hallucination(explanation: str, reference: str, student: str) -> float:
        """
        Simple hallucination proxy: fraction of 4-gram chunks in explanation
        that appear in neither reference nor student text.

        Not a perfect measure — used as a relative comparison across models.
        """
        if not explanation:
            return 0.0

        words = explanation.lower().split()
        if len(words) < 4:
            return 0.0

        combined_text = (reference + " " + student).lower()
        ngrams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        unsupported = sum(1 for ng in ngrams if ng not in combined_text)
        return unsupported / len(ngrams) if ngrams else 0.0
