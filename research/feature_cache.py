"""
feature_cache.py — FeatureCache for HybridASAGGrader research pipeline.

WHY THIS EXISTS (vs old code):
  Old: AblationStudy.create_ablated_grader() re-runs all 5 models on every ablation variant.
       8 variants × N samples × 5 models = 40N model calls.
  New: Extract features ONCE → cache to disk → ablation reads cache (0 model calls).
       Speedup: ~10x on CPU where each model call takes 1-5s per sample.

USAGE:
  cache = FeatureCache("./cache/features.pkl")
  features = cache.extract_and_cache(grader, samples)  # Run once, saves to disk
  
  # Later runs (ablation experiments):
  features = cache.load_cache()  # Instant load from disk
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from tqdm import tqdm


class FeatureCache:
    """
    Disk-persistent cache for the 5-dimensional feature scores.

    Stores per-sample: semantic_score, coverage_score, missing_keywords,
    formality_score, grammar_score, logic_score, logic_details.

    Cache is keyed by a hash of all sample IDs to detect stale caches.
    """

    def __init__(self, cache_path: str = "./cache/features.pkl"):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self._data: Optional[Dict] = None  # in-memory cache after first load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_cache(
        self,
        grader,
        samples: List,
        force_recompute: bool = False,
    ) -> Dict[str, dict]:
        """
        Extract all 5 feature scores for every sample and persist to disk.

        Args:
            grader: HybridASAGGrader instance (must have all 5 feature methods).
            samples: List[ASAGSample] from evaluation_framework.
            force_recompute: Ignore existing cache and re-run.

        Returns:
            Dict keyed by sample.id → dict of raw scores
            {
                "sample_id": {
                    "semantic_score": float,
                    "coverage_score": float,
                    "missing_keywords": List[str],
                    "formality_score": float,
                    "grammar_score": float,
                    "logic_score": float,
                    "logic_details": Dict[str, float],
                }
            }
        """
        if not force_recompute and self.is_cache_valid(samples):
            print(f"[FeatureCache] Cache hit: {self.cache_path}")
            return self.load_cache()["features"]

        print(f"[FeatureCache] Extracting features for {len(samples)} samples…")
        t0 = time.time()

        features: Dict[str, dict] = {}

        for sample in tqdm(samples, desc="Feature extraction"):
            sid = sample.id

            # Semantic similarity
            semantic = grader.get_semantic_score(
                sample.reference_answer, sample.student_answer
            )

            # Keyword coverage
            coverage, missing = grader.get_keyword_coverage(
                sample.question, sample.reference_answer, sample.student_answer
            )

            # Grammar
            grammar = grader.get_grammar_score(sample.student_answer)

            # Formality
            formality = grader.get_formality_score(sample.student_answer)

            # Logic / NLI
            logic, logic_details = grader.get_logic_score(
                sample.reference_answer, sample.student_answer, grammar
            )

            features[sid] = {
                "semantic_score": semantic,
                "coverage_score": coverage,
                "missing_keywords": missing,
                "formality_score": formality,
                "grammar_score": grammar,
                "logic_score": logic,
                "logic_details": logic_details,
            }

        elapsed = time.time() - t0
        print(f"[FeatureCache] Extraction done in {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")

        # Persist to disk
        cache_obj = {
            "sample_hash": self._hash_samples(samples),
            "n_samples": len(samples),
            "elapsed_s": elapsed,
            "features": features,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[FeatureCache] Saved to {self.cache_path}")
        self._data = cache_obj
        return features

    def load_cache(self) -> Dict:
        """Load cache from disk. Returns full cache object."""
        if self._data is not None:
            return self._data
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Cache not found: {self.cache_path}. "
                "Run extract_and_cache() first."
            )
        with open(self.cache_path, "rb") as f:
            self._data = pickle.load(f)
        print(f"[FeatureCache] Loaded {self._data['n_samples']} samples from {self.cache_path}")
        return self._data

    def is_cache_valid(self, samples: List) -> bool:
        """Return True if cache exists and matches the current sample set."""
        if not os.path.exists(self.cache_path):
            return False
        try:
            cached = self.load_cache()
            return cached["sample_hash"] == self._hash_samples(samples)
        except Exception:
            return False

    def get_feature(self, sample_id: str, feature_name: str, default=0.5):
        """
        Get a single feature for a sample from the loaded cache.

        Args:
            sample_id: The sample's unique ID.
            feature_name: One of: semantic_score, coverage_score, formality_score,
                          grammar_score, logic_score, logic_details, missing_keywords
            default: Value to return if sample not found (neutral = 0.5 for scores).
        """
        if self._data is None:
            self.load_cache()
        return self._data["features"].get(sample_id, {}).get(feature_name, default)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_samples(samples: List) -> str:
        """Stable hash of sample IDs to detect cache staleness."""
        ids = sorted([s.id for s in samples])
        return hashlib.md5(json.dumps(ids).encode()).hexdigest()
