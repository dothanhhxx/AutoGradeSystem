"""Label validation for generated attacks.

auto   : automatic sanity checks (no humans); writes validation/auto_checks.csv and auto_ok_ids.json.
         Evaluation drops rows that fail these checks.
export : blind annotation sheets (300 stratified items) for two human annotators. Annotators see
         question, reference and answer only, and pick correct / partial / incorrect.
agree  : after both sheets are filled: per-attack label validity and Cohen's kappa.
         Writes validation/human_valid_ids.json; report.py then prints human-validated ASR.

The automatic checks are NOT a substitute for the human study; they only remove generation
failures (empty, unchanged, leaked instructions, implausible length).
"""

import csv
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .attacks import load_attacks
from .common import DATASETS, REPORT_SPLITS, RESULTS_DIR, read_json, write_json

VDIR = RESULTS_DIR / "validation"
VDIR.mkdir(parents=True, exist_ok=True)
_META = re.compile(r"\b(incorrect|wrong|rewritten|paraphrase[ds]?|student answer|as an ai|"
                   r"factually|this answer)\b", re.I)


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / len(x | y) if x | y else 1.0


def check_row(r: Dict) -> List[str]:
    """Return a list of failed checks (empty = ok)."""
    fails = []
    t = r["student"].strip()
    src = (r.get("source_student") or "").strip()
    if not t:
        fails.append("empty")
    if src and t.lower() == src.lower():
        fails.append("unchanged")
    if r["attack"] in ("A2", "A4"):
        ratio = len(t.split()) / max(1, len(src.split()))
        if not 0.5 <= ratio <= 2.0:
            fails.append(f"len_ratio={ratio:.2f}")
        if _META.search(t) and not _META.search(src):
            fails.append("meta_leak")
        if r["attack"] == "A4" and _jaccard(t, src) > 0.85:
            fails.append("near_copy")
    if r["attack"] == "A3" and _jaccard(t, src) < 0.6:
        fails.append("too_different")
    return fails


def auto():
    rows_out, ok = [], []
    for name in DATASETS:
        for r in load_attacks(name):
            f = check_row(r)
            rows_out.append({"id": r["id"], "dataset": name, "split": r["split"], "attack": r["attack"],
                             "ok": not f, "fails": ";".join(f)})
            if not f:
                ok.append(r["id"])
    with open(VDIR / "auto_checks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    write_json(VDIR / "auto_ok_ids.json", ok)
    by = defaultdict(lambda: [0, 0])
    for r in rows_out:
        by[r["attack"]][0] += r["ok"]
        by[r["attack"]][1] += 1
    for a in sorted(by):
        print(f"{a}: {by[a][0]}/{by[a][1]} pass ({by[a][0] / by[a][1]:.1%})")


def auto_ok_ids() -> Optional[Set[str]]:
    p = VDIR / "auto_ok_ids.json"
    return set(read_json(p)) if p.exists() else None


def export(n_total: int = 300, seed: int = 7):
    """Stratified by attack (equal share), mixed with 10% genuine answers as attention checks."""
    from .data import load
    rng = random.Random(seed)
    ok = auto_ok_ids()
    by_attack = defaultdict(list)
    for name in DATASETS:
        for r in load_attacks(name, REPORT_SPLITS[name]):
            if ok is None or r["id"] in ok:
                by_attack[r["attack"]].append(r)
    n_checks = n_total // 10
    per = (n_total - n_checks) // len(by_attack)
    picked = []
    for a, rows in sorted(by_attack.items()):
        rng.shuffle(rows)
        picked += rows[:per]
    genuine = []
    for name in DATASETS:
        genuine += load(name, REPORT_SPLITS[name])
    rng.shuffle(genuine)
    for it in genuine[:n_checks]:
        picked.append({"id": it.id, "dataset": it.dataset, "question": it.question, "reference": it.reference,
                       "student": it.student, "attack": "GENUINE", "expected_label": it.label3})
    rng.shuffle(picked)
    key = []
    for i, r in enumerate(picked):
        key.append({"item": i, "id": r["id"], "dataset": r["dataset"], "attack": r["attack"],
                    "expected_label": r["expected_label"]})
    for ann in (1, 2):
        with open(VDIR / f"annotator_{ann}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "question", "reference_answer", "student_answer",
                        "label (correct / partial / incorrect)", "comment"])
            for i, r in enumerate(picked):
                w.writerow([i, r["question"], r["reference"], r["student"], "", ""])
    with open(VDIR / "annotation_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    (VDIR / "ANNOTATION_GUIDE.md").write_text(GUIDE, encoding="utf-8")
    print(f"wrote {len(picked)} items -> {VDIR}/annotator_1.csv, annotator_2.csv (+ annotation_key.csv)")


GUIDE = """# Annotation guide

For each row, read the question and the reference answer, then judge the student answer as a
teacher would for this question:

- **correct**: fully answers the question; meaning matches the reference (wording may differ).
- **partial**: on topic and partly right, but incomplete or vague.
- **incorrect**: wrong, contradicts the reference, off topic, or just lists terms without
  answering.

Judge each row independently. Do not discuss rows with the other annotator until both sheets
are finished. Write the label in lower case in the `label` column; use `comment` for doubts.
"""

_NORM = {"correct": "correct", "partial": "partial", "partially correct": "partial",
         "incorrect": "incorrect", "c": "correct", "p": "partial", "i": "incorrect"}


def _read_sheet(path) -> Dict[int, str]:
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lab = _NORM.get(row["label (correct / partial / incorrect)"].strip().lower())
            if lab:
                out[int(row["item"])] = lab
    return out


def agree():
    a1, a2 = _read_sheet(VDIR / "annotator_1.csv"), _read_sheet(VDIR / "annotator_2.csv")
    with open(VDIR / "annotation_key.csv", encoding="utf-8") as fh:
        key = {int(r["item"]): r for r in csv.DictReader(fh)}
    both = sorted(set(a1) & set(a2))
    if not both:
        print("No items labelled by both annotators yet.")
        return
    ordm = {"incorrect": 0, "partial": 1, "correct": 2}
    k = cohen_kappa_score([ordm[a1[i]] for i in both], [ordm[a2[i]] for i in both], weights="quadratic")
    print(f"items labelled by both: {len(both)} / {len(key)}; quadratic-weighted kappa = {k:.3f}")
    stats = defaultdict(lambda: [0, 0, 0])  # valid_both, valid_any_one, n
    valid_ids = []
    for i in both:
        exp = key[i]["expected_label"]
        v1, v2 = a1[i] == exp, a2[i] == exp
        s = stats[key[i]["attack"]]
        s[0] += v1 and v2
        s[1] += v1 or v2
        s[2] += 1
        if v1 and v2:
            valid_ids.append(key[i]["id"])
    rows = []
    for a in sorted(stats):
        vb, va, n = stats[a]
        rows.append({"attack": a, "n": n, "valid_both": vb / n, "valid_either": va / n})
        print(f"{a:8s} n={n:3d}  valid (both annotators) {vb / n:.1%}  (either) {va / n:.1%}")
    write_json(VDIR / "human_agreement.json", {"kappa_qw": float(k), "n_both": len(both), "per_attack": rows})
    write_json(VDIR / "human_valid_ids.json", valid_ids)
