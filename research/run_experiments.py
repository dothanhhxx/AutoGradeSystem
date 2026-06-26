import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation_framework import SemEvalDataLoader
from app.config import GraderConfig
from research.feature_cache import FeatureCache
from research.ablation_v2 import AblationStudyV2
from app.grader import HybridASAGGrader

def main():
    print("--- 1. Generating Synthetic Dataset ---")
    loader = SemEvalDataLoader()
    # 20 samples to be fast for demo. In real paper, 500 or more.
    samples = loader.create_synthetic_dataset(20) 
    
    print("\n--- 2. Initializing Models ---")
    # Using 4-bit config to avoid OOM if we enable LLM
    config = GraderConfig(use_4bit_quantization=True, skip_llm=True) 
    grader = HybridASAGGrader(config)
    
    cache_path = "./experiments/cache/features.pkl"
    os.makedirs("./experiments/cache", exist_ok=True)
    cache = FeatureCache(cache_path)
    
    print("\n--- 3. Extracting & Caching Features ---")
    cache.extract_and_cache(grader, samples)
    
    features_data = cache.load_cache()
    features = features_data["features"]
    
    print("\n--- 4. Running Ablation Study V2 ---")
    ablation = AblationStudyV2(output_dir="./experiments/results/ablation")
    # Using random_seed and low bootstrap for quick testing
    results = ablation.run_from_cache(features, samples, n_bootstrap=100)
    df = ablation.generate_significance_table(results)
    
    print("\n=== RESULTS ===")
    print(df.to_string(index=False))
    
    print("\n=== LATEX READY FOR PAPER ===")
    print(ablation.to_latex(df))

if __name__ == "__main__":
    main()
