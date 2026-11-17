"""Attack detection from criterion disagreement, including the Cognitive Discrepancy Index (CDI).

D0  CDI rule      : flag if coverage >= 0.85 and logic <= 0.40 (fixed thresholds, as proposed)
D1  CDI score     : delta_rote = coverage - logic (continuous, threshold chosen on dev)
D2  disagreement LR: logistic regression on pairwise criterion gaps + NLI probabilities (interpretable)
D3  boosting      : gradient boosting on all detector features (upper bound)

Training: hybrid features of train-split answers (negatives) + train-split attacks (positives).
Threshold: chosen on dev so that 5% of genuine dev answers are flagged.
Test: report splits. A4 (meaning-preserving paraphrase) is a negative: flagging it is a false alarm.
The detector only routes answers to a teacher; it never changes the grade.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import FAMILY, load_attacks
from .common import FIT_SPLITS, LABEL_TO_ORD, REPORT_SPLITS, RESULTS_DIR, write_json
from .data import load
from .scoring import FEATS

POSITIVE_ATTACKS = ["A1a", "A1b", "A1c", "A2", "A3", "A5", "A6", "A7"]
TARGET_FPR = 0.05
CDI_COV, CDI_LOG = 0.85, 0.40


def detector_features(f: Dict, student: str) -> Dict[str, float]:
    d = f.get("logic_details") or {}
    n_words = max(1, len(student.split()))
    nk = f.get("n_keywords") or 0
    found = nk - len(f.get("missing_keywords") or [])
    x = {k: f[k] for k in FEATS}
    x.update({
        "cdi": f["coverage"] - f["logic"],
        "gap_sem_log": f["semantic"] - f["logic"],
        "gap_cov_sem": f["coverage"] - f["semantic"],
        "gap_gram_log": f["grammar"] - f["logic"],
        "p_contra": max(d.get("contradiction", 0.0), d.get("backward_contradiction", 0.0)),
        "p_entail": (d.get("entailment", 0.0) + d.get("backward_entailment", 0.0)) / 2,
        "p_neutral": (d.get("neutral", 0.0) + d.get("backward_neutral", 0.0)) / 2,
        "log_words": float(np.log(n_words)),
        "kw_density": found / n_words,
        "short_branch": float(bool(f.get("short_branch"))),
    })
    return x


D2_COLS = ["cdi", "gap_sem_log", "gap_cov_sem", "gap_gram_log", "p_contra", "p_entail", "p_neutral"]


def _frame(rows: List[Dict], feats: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([detector_features(f, r["student"]) for r, f in zip(rows, feats)])


def _dataset(name: str, splits: List[str]):
    """Rows (clean + attacks) with y=1 for score-inflating attacks."""
    from .pipeline import feats_for
    clean = load(name, splits)
    adv = load_attacks(name, splits)
    from .validation import auto_ok_ids
    ok = auto_ok_ids()
    if ok is not None:
        adv = [r for r in adv if r["id"] in ok]
    rows = [{"student": it.student, "question": it.question, "reference": it.reference, "attack": "CLEAN",
             "gold": it.label3, "question_id": it.question_id} for it in clean]
    rows += [{"student": r["student"], "question": r["question"], "reference": r["reference"],
              "attack": r["attack"], "gold": r["expected_label"], "question_id": r["question_id"]} for r in adv]
    feats = feats_for(rows)
    X = _frame(rows, feats)
    y = np.array([int(r["attack"] in POSITIVE_ATTACKS) for r in rows])
    meta = pd.DataFrame({"attack": [r["attack"] for r in rows], "gold": [r["gold"] for r in rows],
                         "question_id": [r["question_id"] for r in rows]})
    return X, y, meta, feats


class Detectors:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.d2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        self.d3 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed,
                                                 class_weight="balanced")
        self.thr: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.d2.fit(X[D2_COLS], y)
        self.d3.fit(X, y)
        return self

    def scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "D0_cdi_rule": ((X["coverage"] >= CDI_COV) & (X["logic"] <= CDI_LOG)).astype(float).values,
            "D1_cdi_score": X["cdi"].values,
            "D2_disagreement_lr": self.d2.predict_proba(X[D2_COLS])[:, 1],
            "D3_boosting": self.d3.predict_proba(X)[:, 1],
        }

    def calibrate(self, X_dev: pd.DataFrame, y_dev: np.ndarray, fpr: float = TARGET_FPR):
        s = self.scores(X_dev)
        for k, v in s.items():
            if k == "D0_cdi_rule":
                self.thr[k] = 0.5
            else:
                neg = v[y_dev == 0]
                self.thr[k] = float(np.quantile(neg, 1 - fpr))
        return self

    def lr_coefficients(self) -> Dict[str, float]:
        lr = self.d2.named_steps["logisticregression"]
        return {c: float(w) for c, w in zip(D2_COLS, lr.coef_[0])}


def _tpr_at_fpr(y, s, fpr):
    neg = s[y == 0]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return np.nan
    t = np.quantile(neg, 1 - fpr)
    return float(np.mean(s[y == 1] > t))


def evaluate(det: Detectors, X, y, meta, grade_labels: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Overall metrics and per-attack flag rates at the dev-calibrated threshold."""
    overall, per_attack = [], []
    s_all = det.scores(X)
    gold = meta["gold"].values
    attack = meta["attack"].values
    clean = attack == "CLEAN"
    from sklearn.metrics import roc_curve
    det.roc = {}
    for k, s in s_all.items():
        if len(np.unique(y)) > 1:
            f_, t_, _ = roc_curve(y, s)
            keep = np.unique(np.linspace(0, len(f_) - 1, min(len(f_), 200)).astype(int))
            det.roc[k] = {"fpr": f_[keep].round(4).tolist(), "tpr": t_[keep].round(4).tolist()}
        flag = s > det.thr[k] if k != "D0_cdi_rule" else s > 0.5
        # Attack succeeds against the defended system if it is graded above expected AND not flagged.
        above = np.array([LABEL_TO_ORD[g] > LABEL_TO_ORD[e] for g, e in zip(grade_labels, gold)])
        pos = y == 1
        row = {
            "detector": k,
            "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
            "auprc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
            "tpr@1%fpr(oracle)": _tpr_at_fpr(y, s, 0.01),
            "tpr@5%fpr(oracle)": _tpr_at_fpr(y, s, 0.05),
            "threshold(dev)": det.thr[k],
            "tpr(dev thr)": float(flag[pos].mean()),
            "fpr_clean(dev thr)": float(flag[clean].mean()),
            "fpr_clean_correct": float(flag[clean & (gold == "correct")].mean()),
            "fpr_clean_incorrect": float(flag[clean & (gold == "incorrect")].mean()),
            "fpr_A4_paraphrase": float(flag[attack == "A4"].mean()) if (attack == "A4").any() else np.nan,
            "asr_undefended": float(above[pos].mean()),
            "asr_defended": float((above & ~flag)[pos].mean()),
        }
        overall.append(row)
        for a in sorted(set(attack)):
            m = attack == a
            per_attack.append({"detector": k, "attack": a, "family": FAMILY.get(a, "clean"),
                               "n": int(m.sum()), "flag_rate": float(flag[m].mean()),
                               "asr_undefended": float(above[m].mean()) if a != "CLEAN" else np.nan,
                               "asr_defended": float((above & ~flag)[m].mean()) if a != "CLEAN" else np.nan})
    return overall, per_attack


def run(datasets: List[str]):
    from .pipeline import load_feature_systems
    all_overall, all_attack, loao, coefs, rocs = [], [], [], {}, {}
    for name in datasets:
        Xtr, ytr, mtr, _ = _dataset(name, ["train"])
        Xdv, ydv, mdv, _ = _dataset(name, ["dev"])
        Xte, yte, mte, fte = _dataset(name, REPORT_SPLITS[name])
        hybrid = load_feature_systems(name)[0]
        _, grade_labels = hybrid.predict(fte)

        det = Detectors().fit(Xtr, ytr).calibrate(Xdv, ydv)
        coefs[name] = det.lr_coefficients()
        o, pa = evaluate(det, Xte, yte, mte, grade_labels)
        rocs[name] = det.roc
        for r in o:
            r["dataset"] = name
        for r in pa:
            r["dataset"] = name
        all_overall += o
        all_attack += pa

        # Leave-one-family-out: train without a family, measure flag rate on it at the dev threshold.
        fam_tr = mtr["attack"].map(lambda a: FAMILY.get(a, "clean")).values
        fam_dv = mdv["attack"].map(lambda a: FAMILY.get(a, "clean")).values
        fam_te = mte["attack"].map(lambda a: FAMILY.get(a, "clean")).values
        for fam in sorted({FAMILY[a] for a in POSITIVE_ATTACKS}):
            keep_tr, keep_dv = fam_tr != fam, fam_dv != fam
            if (ytr[keep_tr] == 1).sum() == 0:
                continue
            d = Detectors().fit(Xtr[keep_tr], ytr[keep_tr]).calibrate(Xdv[keep_dv], ydv[keep_dv])
            s = d.scores(Xte)
            m = fam_te == fam
            if m.sum() == 0:
                continue
            for k in ("D2_disagreement_lr", "D3_boosting"):
                loao.append({"dataset": name, "held_out_family": fam, "detector": k, "n": int(m.sum()),
                             "flag_rate_held_out": float((s[k][m] > d.thr[k]).mean()),
                             "flag_rate_in_dist": float((s[k][(~m) & (yte == 1)] > d.thr[k]).mean()),
                             "fpr_clean": float((s[k][mte["attack"].values == "CLEAN"] > d.thr[k]).mean())})
    pd.DataFrame(all_overall).to_csv(RESULTS_DIR / "detector_overall.csv", index=False)
    pd.DataFrame(all_attack).to_csv(RESULTS_DIR / "detector_per_attack.csv", index=False)
    pd.DataFrame(loao).to_csv(RESULTS_DIR / "detector_loao.csv", index=False)
    write_json(RESULTS_DIR / "detector_lr_coefficients.json", coefs)
    write_json(RESULTS_DIR / "detector_roc.json", rocs)
    cols = ["dataset", "detector", "auroc", "tpr(dev thr)", "fpr_clean(dev thr)", "fpr_clean_correct",
            "fpr_A4_paraphrase", "asr_undefended", "asr_defended"]
    print(pd.DataFrame(all_overall)[cols].round(3).to_string(index=False))
