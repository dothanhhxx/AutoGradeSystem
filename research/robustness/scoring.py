"""Single source of truth for turning the five indicators into a grade and a 3-way label.

`grade()` reproduces app.models.calculate_final_grade (weighted sum + contradiction caps +
near-perfect bonus). Weights are fitted with NNLS on train; label thresholds are chosen on dev.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls

from .common import LABELS3
from .metrics import qwk

FEATS = ["semantic", "coverage", "formality", "grammar", "logic"]
DEFAULT_WEIGHTS = {k: 0.2 for k in FEATS}
# Weights reported in research/paper_draft.md (Eq. 16).
PAPER_WEIGHTS = {"semantic": 0.15, "coverage": 0.30, "grammar": 0.10, "formality": 0.05, "logic": 0.40}


def max_contradiction(f: Dict) -> float:
    d = f.get("logic_details") or {}
    return max(d.get("contradiction", 0.0), d.get("backward_contradiction", 0.0))


def grade(f: Dict, w: Dict[str, float], rules: bool = True) -> float:
    """0-100 grade. rules=False gives the pure weighted sum (used for single-criterion systems)."""
    tot = sum(w.values()) or 1.0
    g = 100.0 * sum(f[k] * w.get(k, 0.0) for k in FEATS) / tot
    if not rules:
        return float(min(100.0, max(0.0, g)))
    mc = max_contradiction(f)
    if mc > 0.90:
        g = min(g, 40.0)
    elif mc > 0.70:
        g = min(g, 55.0)
    elif mc > 0.50:
        g = max(0.0, g - (mc - 0.50) * 30)
    if (f["semantic"] >= 0.95 and f["grammar"] >= 0.90 and f["logic"] >= 0.50
            and len(f.get("missing_keywords") or []) <= 1):
        g = max(g, 85.0)
    return float(round(min(100.0, max(0.0, g)), 2))


def to_label(g: float, thr: Tuple[float, float]) -> str:
    if g >= thr[1]:
        return "correct"
    if g >= thr[0]:
        return "partial"
    return "incorrect"


def fit_weights_nnls(feats: List[Dict], gold: Sequence[float]) -> Dict[str, float]:
    """Non-negative least squares with a free intercept (fit on centred data), weights sum to 1.
    Without the intercept, NNLS favours whichever feature's mean happens to match the label mean."""
    X = np.array([[f[k] for k in FEATS] for f in feats])
    y = np.asarray(gold, dtype=float)
    w, _ = nnls(X - X.mean(0), y - y.mean())
    if w.sum() == 0:
        return dict(DEFAULT_WEIGHTS)
    w = w / w.sum()
    return {k: float(round(v, 4)) for k, v in zip(FEATS, w)}


def fit_thresholds(grades: Sequence[float], gold_ord: Sequence[int], step: float = 1.0) -> Tuple[float, float]:
    """Grid search (t_partial < t_correct) maximising QWK on the given set."""
    g = np.asarray(grades)
    y = np.asarray(gold_ord)
    grid = np.arange(0, 100 + step, step)
    best, best_t = -2.0, (45.0, 75.0)
    for i, a in enumerate(grid):
        lo = g >= a
        for b in grid[i + 1:]:
            pred = lo.astype(int) + (g >= b).astype(int)
            k = qwk(y, pred)
            if k > best + 1e-9:
                best, best_t = k, (float(a), float(b))
    return best_t


@dataclass
class FeatureSystem:
    """A grader defined purely on cached features (Hybrid or a single criterion)."""
    name: str
    weights: Dict[str, float]
    rules: bool = True
    thresholds: Tuple[float, float] = (45.0, 75.0)
    meta: Dict = field(default_factory=dict)

    def score(self, f: Dict) -> float:
        return grade(f, self.weights, self.rules)

    def predict(self, feats: List[Dict]) -> Tuple[List[float], List[str]]:
        s = [self.score(f) for f in feats]
        return [x / 100.0 for x in s], [to_label(x, self.thresholds) for x in s]

    def per_criterion(self, f: Dict) -> Dict[str, float]:
        return {k: float(f[k]) for k in FEATS}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["thresholds"] = list(self.thresholds)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureSystem":
        d = dict(d)
        d["thresholds"] = tuple(d["thresholds"])
        return cls(**d)


def build_feature_systems(train_f, train_gold, dev_f, dev_ord) -> List[FeatureSystem]:
    systems = [
        FeatureSystem("hybrid", fit_weights_nnls(train_f, train_gold), True, meta={"weights": "nnls(train)"}),
        FeatureSystem("hybrid_paperw", dict(PAPER_WEIGHTS), True, meta={"weights": "paper Eq.16"}),
        FeatureSystem("hybrid_uniform", dict(DEFAULT_WEIGHTS), True, meta={"weights": "app default 0.2"}),
    ]
    for k in FEATS:
        systems.append(FeatureSystem(f"only_{k}", {k: 1.0}, False, meta={"single": k}))
    for s in systems:
        s.thresholds = fit_thresholds([s.score(f) for f in dev_f], dev_ord)
    return systems


LABEL_ORDER = LABELS3
