"""
Research package for HybridASAGGrader.

Modules:
- feature_cache: Cache 5-dim feature scores to disk for fast ablation
- llm_benchmark: Comparative benchmark for LLM synthesis models
- embedding_benchmark: Comparative benchmark for embedding models
- ablation_v2: AblationStudyV2 with Bonferroni-corrected significance tests
- experiment_runner: MLflow-tracked experiment orchestration
"""

from .feature_cache import FeatureCache
from .llm_benchmark import LLMComparativeBenchmark
from .embedding_benchmark import EmbeddingComparativeBenchmark
from .ablation_v2 import AblationStudyV2
from .experiment_runner import ExperimentRunner

__all__ = [
    "FeatureCache",
    "LLMComparativeBenchmark",
    "EmbeddingComparativeBenchmark",
    "AblationStudyV2",
    "ExperimentRunner",
]
