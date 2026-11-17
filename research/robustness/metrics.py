"""Metrics: agreement, attack success, paired deltas, question-level bootstrap."""

from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .common import LABEL_TO_ORD


_W3 = np.array([[(i - j) ** 2 for j in range(3)] for i in range(3)], dtype=float)


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa on labels {0,1,2}; equals sklearn's cohen_kappa_score(weights='quadratic')."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0
    O = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (_W3 * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - (_W3 * O).sum() / den)


def agreement(gold_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    g = [LABEL_TO_ORD[x] for x in gold_labels]
    p = [LABEL_TO_ORD[x] for x in pred_labels]
    return {
        "n": len(g),
        "qwk": qwk(g, p),
        "accuracy": float(accuracy_score(g, p)),
        "macro_f1": float(f1_score(g, p, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


def bootstrap_ci(groups: Sequence[str], stat: Callable[[np.ndarray], float], n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05):
    """Cluster bootstrap: resample *questions* (groups) with replacement, not answers."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        v = stat(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def attack_success(expected: Sequence[str], pred: Sequence[str]) -> Dict[str, float]:
    """For score-inflating attacks the expected label is 'incorrect' (or 'partial' for A7):
    success = predicted strictly above expected; 'to_correct' = predicted 'correct'."""
    e = np.array([LABEL_TO_ORD[x] for x in expected])
    p = np.array([LABEL_TO_ORD[x] for x in pred])
    return {
        "asr": float(np.mean(p > e)) if len(e) else float("nan"),
        "asr_to_correct": float(np.mean(p == 2)) if len(e) else float("nan"),
    }


def invariance_failure(orig_pred: Sequence[str], adv_pred: Sequence[str]) -> Dict[str, float]:
    """For meaning-preserving attacks (A4): fraction whose label drops."""
    o = np.array([LABEL_TO_ORD[x] for x in orig_pred])
    a = np.array([LABEL_TO_ORD[x] for x in adv_pred])
    return {"label_drop": float(np.mean(a < o)) if len(o) else float("nan"),
            "label_change": float(np.mean(a != o)) if len(o) else float("nan")}


def paired_delta(orig: Sequence[float], adv: Sequence[float]) -> Dict[str, float]:
    o, a = np.asarray(orig, float), np.asarray(adv, float)
    d = a - o
    out = {"mean_delta": float(d.mean()) if len(d) else float("nan"),
           "median_delta": float(np.median(d)) if len(d) else float("nan")}
    try:
        out["wilcoxon_p"] = float(wilcoxon(a, o).pvalue) if len(d) >= 10 and np.any(d != 0) else float("nan")
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    return out
