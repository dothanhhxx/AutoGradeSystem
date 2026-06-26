"""
experiment_runner.py — ExperimentRunner with MLflow tracking.

WHY THIS EXISTS (vs old code):
  Old: No experiment tracking. Results printed to stdout, lost after session.
  New: MLflow tracks every run with params, metrics, and artifacts.
       - Local mlruns/ directory (no cloud needed)
       - Reproducible with logged random seeds
       - Generates paper-ready tables from stored run IDs

USAGE:
  runner = ExperimentRunner(experiment_name="HybridASAG-Research")
  
  # Run ablation
  run_id = runner.run_ablation(
      cached_features=features,
      samples=samples,
      config={"weights": {"semantic": 0.45, ...}}
  )
  
  # Run embedding comparison
  run_id = runner.run_embedding_benchmark(
      model_names=["BAAI/bge-m3", "princeton-nlp/sup-simcse-roberta-large"],
      samples=samples,
  )
  
  # Export tables from completed runs
  tables = runner.generate_paper_tables([run_id_ablation, run_id_emb])
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd


class ExperimentRunner:
    """
    Orchestrates research experiments with MLflow tracking.

    Falls back gracefully if mlflow is not installed
    (logs to JSON file instead).
    """

    def __init__(
        self,
        experiment_name: str = "HybridASAG-Research",
        mlflow_uri: str = "./mlruns",
        output_dir: str = "./results",
    ):
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._mlflow_available = False
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow_available = True
            print(f"[ExperimentRunner] MLflow tracking at {mlflow_uri}")
        except ImportError:
            print("[ExperimentRunner] mlflow not installed. Logging to JSON fallback.")
            print("  Install: pip install mlflow")

    # ------------------------------------------------------------------
    # Public: run ablation
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        cached_features: Dict[str, dict],
        samples: List,
        config: Optional[Dict] = None,
        n_bootstrap: int = 500,
        random_seed: int = 42,
    ) -> str:
        """
        Run full ablation study and track with MLflow.

        Returns:
            run_id (str) for later retrieval.
        """
        from .ablation_v2 import AblationStudyV2

        config = config or {}
        run_id = f"ablation_{int(time.time())}"
        output_dir = os.path.join(self.output_dir, "ablation", run_id)

        ablation = AblationStudyV2(output_dir=output_dir)
        results = ablation.run_from_cache(
            cached_features=cached_features,
            samples=samples,
            weights=config.get("weights"),
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        # Log to MLflow
        if self._mlflow_available:
            import mlflow
            with mlflow.start_run(run_name=f"ablation_{random_seed}") as run:
                mlflow.log_params({"n_samples": len(samples), "n_bootstrap": n_bootstrap,
                                   "random_seed": random_seed, **config})
                for variant_name, r in results.items():
                    safe = variant_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "m")
                    mlflow.log_metric(f"{safe}_accuracy", r["accuracy"])
                    mlflow.log_metric(f"{safe}_macro_f1", r["macro_f1"])
                    mlflow.log_metric(f"{safe}_qwk", r["qwk"])

                mlflow.log_artifact(os.path.join(output_dir, "ablation_results.csv"))
                mlflow.log_artifact(os.path.join(output_dir, "ablation_latex.tex"))
                run_id = run.info.run_id
        else:
            self._save_json_fallback(run_id, "ablation", results, config)

        print(f"[ExperimentRunner] Ablation run completed: {run_id}")
        return run_id

    # ------------------------------------------------------------------
    # Public: run embedding benchmark
    # ------------------------------------------------------------------

    def run_embedding_benchmark(
        self,
        model_names: List[str],
        samples: List,
        human_scores: Optional[List[float]] = None,
        batch_size: int = 32,
    ) -> str:
        """Run embedding comparative benchmark and track with MLflow."""
        from .embedding_benchmark import EmbeddingComparativeBenchmark

        run_id = f"emb_{int(time.time())}"
        output_dir = os.path.join(self.output_dir, "embedding", run_id)

        bench = EmbeddingComparativeBenchmark(output_dir=output_dir)
        df = bench.benchmark_all(
            model_names=model_names,
            samples=samples,
            human_scores=human_scores,
            batch_size=batch_size,
        )

        if self._mlflow_available:
            import mlflow
            with mlflow.start_run(run_name="embedding_benchmark") as run:
                mlflow.log_param("n_models", len(model_names))
                mlflow.log_param("n_samples", len(samples))
                for _, row in df.iterrows():
                    safe = row["model_name"].replace("/", "_").replace("-", "_")
                    if row.get("pearson_r") is not None:
                        mlflow.log_metric(f"{safe}_pearson_r", row["pearson_r"])
                    if row.get("spearman_rho") is not None:
                        mlflow.log_metric(f"{safe}_spearman_rho", row["spearman_rho"])
                    if row.get("roc_auc") is not None:
                        mlflow.log_metric(f"{safe}_roc_auc", row["roc_auc"])

                mlflow.log_artifact(os.path.join(output_dir, "embedding_benchmark_final.csv"))
                run_id = run.info.run_id
        else:
            self._save_json_fallback(run_id, "embedding", df.to_dict("records"), {})

        print(f"[ExperimentRunner] Embedding benchmark run completed: {run_id}")
        return run_id

    # ------------------------------------------------------------------
    # Public: run LLM benchmark
    # ------------------------------------------------------------------

    def run_llm_benchmark(
        self,
        model_configs: List[Dict[str, Any]],
        samples: List,
        reference_tags: List[List[str]],
        max_new_tokens: int = 400,
    ) -> str:
        """Run LLM comparative benchmark and track with MLflow."""
        from .llm_benchmark import LLMComparativeBenchmark

        run_id = f"llm_{int(time.time())}"
        output_dir = os.path.join(self.output_dir, "llm", run_id)

        bench = LLMComparativeBenchmark(output_dir=output_dir)
        df = bench.benchmark_all(
            model_configs=model_configs,
            samples=samples,
            reference_tags=reference_tags,
            max_new_tokens=max_new_tokens,
        )

        if self._mlflow_available:
            import mlflow
            with mlflow.start_run(run_name="llm_benchmark") as run:
                mlflow.log_param("n_models", len(model_configs))
                mlflow.log_param("n_samples", len(samples))
                for _, row in df.iterrows():
                    safe = row["model_name"].replace(" ", "_").replace("-", "_")
                    for metric in ["json_compliance_rate", "tag_accuracy", "avg_latency_s"]:
                        if row.get(metric) is not None:
                            mlflow.log_metric(f"{safe}_{metric}", row[metric])

                mlflow.log_artifact(os.path.join(output_dir, "llm_benchmark_final.csv"))
                run_id = run.info.run_id
        else:
            self._save_json_fallback(run_id, "llm", df.to_dict("records"), {})

        print(f"[ExperimentRunner] LLM benchmark run completed: {run_id}")
        return run_id

    # ------------------------------------------------------------------
    # Public: generate paper tables from run IDs
    # ------------------------------------------------------------------

    def generate_paper_tables(self, run_ids: List[str]) -> Dict[str, str]:
        """
        Load results from MLflow (or JSON fallback) and generate LaTeX tables.

        Returns:
            Dict: {table_name: latex_string}
        """
        tables = {}

        if self._mlflow_available:
            import mlflow
            client = mlflow.tracking.MlflowClient(self.mlflow_uri)
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    artifacts = client.list_artifacts(run_id)
                    for art in artifacts:
                        if art.path.endswith(".csv"):
                            local = client.download_artifacts(run_id, art.path)
                            df = pd.read_csv(local)
                            tables[art.path] = df.to_latex(index=False)
                except Exception as e:
                    print(f"[ExperimentRunner] Could not load run {run_id}: {e}")
        else:
            for run_id in run_ids:
                fallback_path = os.path.join(self.output_dir, f"{run_id}_fallback.json")
                if os.path.exists(fallback_path):
                    with open(fallback_path) as f:
                        data = json.load(f)
                    df = pd.DataFrame(data.get("results", []))
                    tables[run_id] = df.to_latex(index=False)

        return tables

    # ------------------------------------------------------------------
    # Private: JSON fallback logging
    # ------------------------------------------------------------------

    def _save_json_fallback(
        self,
        run_id: str,
        run_type: str,
        results: Any,
        config: Dict,
    ):
        """Save results to JSON when MLflow is not available."""
        path = os.path.join(self.output_dir, f"{run_id}_fallback.json")
        payload = {
            "run_id": run_id,
            "run_type": run_type,
            "config": config,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": results if isinstance(results, list) else str(results),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[ExperimentRunner] Fallback log saved: {path}")
