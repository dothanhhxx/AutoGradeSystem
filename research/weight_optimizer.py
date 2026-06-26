import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class WeightOptimizer:
    """
    Data-driven weight optimization using Machine Learning.
    Learns optimal criteria weights from the feature cache and ground truth labels.
    """
    def __init__(self):
        self.feature_names = ["semantic", "coverage", "formality", "grammar", "logic"]
        # Map qualitative labels to quantitative target scores (0.0 to 1.0)
        self.label_mapping = {
            "correct": 1.0,
            "partially_correct_incomplete": 0.5,
            "contradictory": 0.0,
            "irrelevant": 0.0,
            "non_domain": 0.0
        }

    def prepare_data(self, features: Dict[str, dict], samples: List) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples and feature cache into X (features matrix) and y (target vector)."""
        X_list = []
        Y_list = []
        for sample in samples:
            if sample.id in features:
                f = features[sample.id]
                x_row = [f.get(fn, 0.0) for fn in self.feature_names]
                
                # Use gold_score if available, otherwise map the categorical gold_label
                if hasattr(sample, 'gold_score') and sample.gold_score is not None:
                    y_val = sample.gold_score
                else:
                    y_val = self.label_mapping.get(sample.gold_label, 0.0)
                    
                X_list.append(x_row)
                Y_list.append(y_val)
        return np.array(X_list), np.array(Y_list)

    def optimize_linear(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Use Linear Regression to find explicit, interpretable weights.
        Constrains intercept to 0 so the final score is purely a sum of weighted features.
        """
        model = LinearRegression(fit_intercept=False) 
        model.fit(X, y)
        
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        # Handle cases where variance is 0 (synthetic datasets)
        if np.var(y) == 0:
            r2 = 1.0 if np.allclose(preds, y) else 0.0
        else:
            r2 = r2_score(y, preds)
        
        # Extract and normalize coefficients to be valid percentages that sum to 1
        coefs = model.coef_
        coefs = np.maximum(coefs, 0) # Cap negative weights at 0 (a feature shouldn't penalize below 0 inherently)
        
        if coefs.sum() > 0:
            coefs = coefs / coefs.sum()
        else:
            # Fallback if regression completely fails
            coefs = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        weights = {name: float(w) for name, w in zip(self.feature_names, coefs)}
        
        return {
            "model_type": "LinearRegression",
            "weights": weights,
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def optimize_rf(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Use Random Forest as a non-linear baseline to measure theoretical max accuracy.
        Outputs Feature Importance instead of direct weights.
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        
        if np.var(y) == 0:
            r2 = 1.0 if np.allclose(preds, y) else 0.0
        else:
            r2 = r2_score(y, preds)
        
        importance = model.feature_importances_
        feature_importance = {name: float(w) for name, w in zip(self.feature_names, importance)}
        
        return {
            "model_type": "RandomForest",
            "feature_importance": feature_importance,
            "rmse": float(rmse),
            "r2": float(r2)
        }
