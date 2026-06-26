import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation_framework import SemEvalDataLoader
from research.feature_cache import FeatureCache
from research.weight_optimizer import WeightOptimizer

def main():
    print("============================================================")
    print("Phase 3: Machine Learning Weight Optimizer")
    print("============================================================")
    
    print("\n--- 1. Loading Data ---")
    loader = SemEvalDataLoader()
    
    # Generate a larger synthetic set so ML has enough variance to learn from. 
    # In a real paper, this would be loader.load_data("train")
    samples = loader.create_synthetic_dataset(100) 
    
    cache_path = "./experiments/cache/features.pkl"
    cache = FeatureCache(cache_path)
    
    print("\n--- 2. Extracting/Loading Features ---")
    # We call extract_and_cache to ensure the cache is populated for the 100 samples
    # We use a dummy grader for extraction if needed, but since we are simulating,
    # if it's not cached, it will fail. Let's just load the cache and only use samples that are in it.
    features_data = cache.load_cache()
    
    if not features_data:
        print("Error: No feature cache found. Please run ablation/extraction first.")
        return
        
    features = features_data["features"]
    
    # Filter samples to only those that have been cached
    available_samples = [s for s in samples if s.id in features]
    print(f"Loaded {len(features)} total cached feature vectors.")
    print(f"Found {len(available_samples)} matching samples with Ground Truth labels.")
    
    if len(available_samples) < 5:
        print("Not enough samples to run ML. Please run the feature extraction on more data first.")
        return
        
    print("\n--- 3. Preparing ML Dataset ---")
    optimizer = WeightOptimizer()
    X, y = optimizer.prepare_data(features, available_samples)
    print(f"X shape: {X.shape} (Features), y shape: {y.shape} (Ground Truth Scores)")
    
    print("\n--- 4. Running Optimization (Linear Regression) ---")
    lin_results = optimizer.optimize_linear(X, y)
    print(f"Model: {lin_results['model_type']}")
    print(f"RMSE: {lin_results['rmse']:.4f} (Error against Teacher Score, Lower is better)")
    print(f"R-Squared: {lin_results['r2']:.4f} (Fit accuracy, Higher is better)")
    
    print("\n=== OPTIMAL WEIGHTS (DATA-DRIVEN) ===")
    for feature, weight in lin_results['weights'].items():
        print(f"{feature.capitalize():>10}: {weight*100:>5.1f}%")
        
    print("\n--- 5. Running Baseline (Random Forest) ---")
    rf_results = optimizer.optimize_rf(X, y)
    print(f"Model: {rf_results['model_type']}")
    print(f"RMSE: {rf_results['rmse']:.4f}")
    print(f"R-Squared: {rf_results['r2']:.4f}")
    
    print("\n=== FEATURE IMPORTANCE (RANDOM FOREST) ===")
    for feature, importance in rf_results['feature_importance'].items():
        print(f"{feature.capitalize():>10}: {importance*100:>5.1f}%")
        
    print("\n--- 6. Saving Results ---")
    os.makedirs("./experiments/results/optimizer", exist_ok=True)
    with open("./experiments/results/optimizer/ml_weights.json", "w", encoding="utf-8") as f:
        json.dump(lin_results['weights'], f, indent=4)
    print("Saved optimal weights to ./experiments/results/optimizer/ml_weights.json")
    print("Next step: Add these weights as 'ml_optimized' in app/config.py!")

if __name__ == "__main__":
    main()
