"""
embedding_benchmark.py — EmbeddingComparativeBenchmark for HybridASAGGrader research.

WHY THIS EXISTS (vs old code):
  Old: SimCSE (princeton-nlp/sup-simcse-roberta-large) is hardcoded as the only
       embedding model. No evaluation of alternatives exists.
  New: Pre-compute embeddings for all models → compare offline (no re-loading).
       Measures: Pearson r, Spearman ρ, ROC-AUC (binary discrimination),
       inference speed (ms/sample), model size (MB).

USAGE:
  model_names = [
      "princeton-nlp/sup-simcse-roberta-large",  # Baseline
      "sentence-transformers/all-mpnet-base-v2",
      "BAAI/bge-m3",
      "intfloat/e5-large-v2",
  ]
  bench = EmbeddingComparativeBenchmark(output_dir="./results/emb_bench")
  df = bench.benchmark_all(model_names, samples, human_scores)
  df.to_csv("embedding_comparison.csv")
"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clamped to [0, 1]."""
    return float(max(0.0, min(1.0, 1 - cosine(a, b))))


class EmbeddingComparativeBenchmark:
    """
    Compare embedding models on semantic similarity for ASAG.

    All similarity scores are computed in a single batch pass per model,
    then compared offline against human labels.
    """

    def __init__(self, output_dir: str = "./results/emb_bench"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark_all(
        self,
        model_names: List[str],
        samples: List,
        human_scores: Optional[List[float]] = None,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Run the benchmark across all embedding model names.

        Args:
            model_names: HuggingFace model IDs (SentenceTransformer-compatible).
            samples: List[ASAGSample] (must have .reference_answer, .student_answer, .gold_label).
            human_scores: Optional continuous human scores (e.g., 0-5 scale). If None,
                          derives from gold_label (correct=1.0, partial=0.5, incorrect=0.0).
            batch_size: Encoding batch size.

        Returns:
            pd.DataFrame with one row per model.
        """
        # Prepare ground truth
        if human_scores is None:
            label_map = {"correct": 1.0, "partially_correct_incomplete": 0.5, "contradictory": 0.0}
            human_scores = [label_map.get(s.gold_label, 0.0) for s in samples]

        binary_labels = [1 if s >= 0.75 else 0 for s in human_scores]  # correct vs not

        rows = []

        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Benchmarking embedding: {model_name}")
            print(f"{'='*60}")

            row = self._benchmark_one(
                model_name=model_name,
                samples=samples,
                human_scores=human_scores,
                binary_labels=binary_labels,
                batch_size=batch_size,
            )
            rows.append(row)

            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(
                os.path.join(self.output_dir, "embedding_benchmark_partial.csv"), index=False
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "embedding_benchmark_final.csv"), index=False)
        print(f"\n[EmbeddingBenchmark] Results saved to {self.output_dir}")
        return df

    # ------------------------------------------------------------------
    # Per-model benchmark
    # ------------------------------------------------------------------

    def _benchmark_one(
        self,
        model_name: str,
        samples: List,
        human_scores: List[float],
        binary_labels: List[int],
        batch_size: int,
    ) -> Dict:
        """Load one SentenceTransformer, compute all similarities, free memory."""
        from sentence_transformers import SentenceTransformer

        # Load model and time it
        t_load = time.time()
        try:
            model = SentenceTransformer(model_name)
            load_time_s = time.time() - t_load
        except Exception as exc:
            return {
                "model_name": model_name,
                "load_error": str(exc),
                "pearson_r": None,
                "spearman_rho": None,
                "roc_auc": None,
                "avg_ms_per_sample": None,
                "model_size_mb": None,
            }

        # Estimate model size from parameters
        total_params = sum(p.numel() for p in model[0].auto_model.parameters())
        model_size_mb = round(total_params * 2 / (1024 * 1024), 1)  # approx fp16 size

        # Encode all texts in batch
        refs = [s.reference_answer for s in samples]
        stus = [s.student_answer for s in samples]

        t_encode = time.time()
        ref_embeddings = model.encode(refs, batch_size=batch_size, show_progress_bar=True,
                                      convert_to_numpy=True)
        stu_embeddings = model.encode(stus, batch_size=batch_size, show_progress_bar=True,
                                      convert_to_numpy=True)
        encode_time_s = time.time() - t_encode
        ms_per_sample = round((encode_time_s / len(samples)) * 1000, 2)

        # Compute per-sample cosine similarity
        similarities = [
            _cosine_sim(ref_embeddings[i], stu_embeddings[i])
            for i in range(len(samples))
        ]

        # Metrics
        pearson, _ = pearsonr(human_scores, similarities)
        spearman, _ = spearmanr(human_scores, similarities)

        try:
            auc = roc_auc_score(binary_labels, similarities)
        except ValueError:
            auc = None  # Only one class in labels

        # Paraphrase robustness: std of similarities for samples with same gold label
        robustness = self._compute_paraphrase_robustness(samples, similarities)

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "model_name": model_name,
            "load_time_s": round(load_time_s, 2),
            "pearson_r": round(pearson, 4),
            "spearman_rho": round(spearman, 4),
            "roc_auc": round(auc, 4) if auc else None,
            "avg_ms_per_sample": ms_per_sample,
            "model_size_mb": model_size_mb,
            "paraphrase_robustness": robustness,
            "mean_similarity": round(float(np.mean(similarities)), 4),
            "std_similarity": round(float(np.std(similarities)), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_paraphrase_robustness(samples: List, similarities: List[float]) -> Optional[float]:
        """
        Robustness proxy: mean within-class std of similarity scores.
        Lower std = more consistent within same label = more robust.

        Returns None if fewer than 2 classes with ≥2 samples.
        """
        from collections import defaultdict
        groups: Dict[str, List[float]] = defaultdict(list)
        for sample, sim in zip(samples, similarities):
            groups[sample.gold_label].append(sim)

        stds = [np.std(v) for v in groups.values() if len(v) >= 2]
        if not stds:
            return None
        return round(float(np.mean(stds)), 4)
