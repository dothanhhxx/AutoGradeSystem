import numpy as np
from sklearn.metrics import cohen_kappa_score

from research.robustness import data
from research.robustness.common import text_key
from research.robustness.metrics import attack_success, bootstrap_ci, invariance_failure, qwk
from research.robustness.scoring import fit_thresholds, grade, to_label


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
        assert abs(qwk(a, b) - cohen_kappa_score(a, b, weights="quadratic")) < 1e-9


def test_attack_success_by_hand():
    r = attack_success(["incorrect"] * 4, ["incorrect", "partial", "correct", "correct"])
    assert r["asr"] == 0.75 and r["asr_to_correct"] == 0.5
    inv = invariance_failure(["correct", "correct", "partial"], ["partial", "correct", "correct"])
    assert abs(inv["label_drop"] - 1 / 3) < 1e-9 and abs(inv["label_change"] - 2 / 3) < 1e-9


def test_grade_rules():
    f = {"semantic": 0.9, "coverage": 0.9, "formality": 0.9, "grammar": 0.9, "logic": 0.9,
         "logic_details": {"contradiction": 0.95}, "missing_keywords": []}
    assert grade(f, {"semantic": 1}) == 40.0          # severe contradiction cap
    f["logic_details"] = {}
    assert grade(f, {"semantic": 1}) == 90.0
    assert to_label(90, (45, 75)) == "correct" and to_label(50, (45, 75)) == "partial"


def test_thresholds_recover_separable():
    g = [10] * 30 + [50] * 30 + [90] * 30
    y = [0] * 30 + [1] * 30 + [2] * 30
    a, b = fit_thresholds(g, y)
    assert 10 < a <= 50 < b <= 90


def test_bootstrap_ci_brackets_mean():
    groups = np.repeat(np.arange(20), 10)
    vals = np.random.default_rng(1).normal(0.5, 0.1, 200)
    lo, hi = bootstrap_ci(groups, lambda idx: vals[idx].mean(), n_boot=300)
    assert lo < vals.mean() < hi


def test_text_key_changes_with_content():
    assert text_key("q", "r", "a") != text_key("q", "r", "a ")


def test_splits_have_no_question_leak():
    for name in data.DATASETS if hasattr(data, "DATASETS") else ["scientsbank", "beetle", "mohler"]:
        data.check_no_question_leak(data.load(name))
