from research.robustness.systems import parse_llm


def test_parse_holistic_json():
    p = parse_llm('```json\n{"label": "partially_correct", "score": 6}\n```', "holistic")
    assert p == {"label": "partial", "score": 0.6}


def test_parse_analytic_json():
    p = parse_llm('{"semantic": 8, "coverage": 6, "grammar": 10, "formality": 7, "logic": 9, "label": "correct"}',
                  "analytic")
    assert p["label"] == "correct" and abs(p["per_criterion"]["coverage"] - 0.6) < 1e-9
    assert abs(p["score"] - 0.8) < 1e-9


def test_parse_fallback_and_failure():
    assert parse_llm('label: incorrect, score: 2', "holistic")["label"] == "incorrect"
    assert parse_llm("I cannot grade this.", "holistic") is None
