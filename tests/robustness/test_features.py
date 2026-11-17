"""Batched extractor must match HybridASAGGrader's per-sample methods (slow: loads models)."""

import random

import pytest

from research.robustness import data
from research.robustness.features import FeatureExtractor

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(batch_size=8)


def test_batched_matches_grader(extractor):
    g = extractor.g
    items = data.load("scientsbank", ["test_ua"]) + data.load("mohler", ["test"])
    random.Random(0).shuffle(items)
    items = [it for it in items if len(it.student.split()) >= 5][:24]
    rows = [{"question": it.question, "reference": it.reference, "student": it.student} for it in items]
    feats = extractor.extract(rows)
    for r, f in zip(rows, feats):
        sem = g.get_semantic_score(r["reference"], r["student"])
        cov, missing = g.get_keyword_coverage(r["question"], r["reference"], r["student"])
        gram = g.get_grammar_score(r["student"])
        form = g.get_formality_score(r["student"])
        logic, _ = g.get_logic_score(r["reference"], r["student"], gram)
        assert f["semantic"] == pytest.approx(sem, abs=2e-3)
        assert f["coverage"] == pytest.approx(cov, abs=1e-6)
        assert f["grammar"] == pytest.approx(gram, abs=2e-3)
        assert f["formality"] == pytest.approx(form, abs=2e-3)
        assert f["logic"] == pytest.approx(logic, abs=5e-3)


def test_short_branch(extractor):
    f = extractor.extract([{"question": "q", "reference": "a stack is lifo", "student": "stack lifo"}])[0]
    assert f["short_branch"] and f["grammar"] == 0.5 and f["semantic"] == pytest.approx(0.5)
