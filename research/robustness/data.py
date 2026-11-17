"""Loaders for the real ASAG benchmarks and question-level splits.

Sources (all CC-BY-4.0, downloaded to data/raw/ from the HuggingFace Hub):
  - SciEntsBank, Beetle: SemEval-2013 Task 7 (Dzikovska et al., 2013), mirrors nkazi/SciEntsBank, nkazi/Beetle
  - Mohler: Mohler, Bunescu & Mihalcea (2011), mirror Viswa45/ASAG (open-ended subset)

Official SemEval test splits are kept (test_ua = unseen answers, test_uq = unseen questions,
test_ud = unseen domains). A dev split is carved from train at the question level.
Mohler has no official split, so it is split 60/20/20 by question.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .common import DATA_PROCESSED, DATA_RAW, LABEL_TO_ORD, read_jsonl, text_key, write_json, write_jsonl

SEMEVAL_5WAY = {0: "correct", 1: "contradictory", 2: "partially_correct_incomplete", 3: "irrelevant", 4: "non_domain"}
FIVE_TO_THREE = {
    "correct": "correct",
    "partially_correct_incomplete": "partial",
    "contradictory": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}
THREE_TO_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
SPLIT_SEED = 42


@dataclass
class Item:
    id: str
    dataset: str
    split: str
    question_id: str
    question: str
    reference: str
    student: str
    label3: str                 # incorrect | partial | correct
    gold_score: float           # in [0, 1]
    label5: Optional[str] = None
    # Filled for adversarial variants
    attack: Optional[str] = None
    orig_id: Optional[str] = None

    @property
    def key(self) -> str:
        return text_key(self.question, self.reference, self.student)

    @property
    def label_ord(self) -> int:
        return LABEL_TO_ORD[self.label3]

    def to_dict(self) -> Dict:
        return asdict(self)


def mohler_label3(avg: float) -> str:
    if avg >= 4.0:
        return "correct"
    if avg >= 2.5:
        return "partial"
    return "incorrect"


def _load_semeval(name: str) -> List[Item]:
    folder = {"scientsbank": "SciEntsBank", "beetle": "Beetle"}[name]
    files = {
        "train": "train-00001.parquet",
        "test_ua": "test-ua-00001.parquet",
        "test_uq": "test-uq-00001.parquet",
        "test_ud": "test-ud-00001.parquet",
    }
    items: List[Item] = []
    for split, fname in files.items():
        path = DATA_RAW / folder / "data" / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for r in df.itertuples(index=False):
            l5 = SEMEVAL_5WAY[int(r.label)]
            l3 = FIVE_TO_THREE[l5]
            q = str(r.question).strip()
            items.append(Item(
                id=f"{name}:{r.id}",
                dataset=name,
                split=split,
                # Official question code = first two id parts (Beetle "FF.Q2", SciEntsBank "EM.45b").
                # Question text alone is ambiguous ("Explain your reasoning." spans many questions).
                question_id=f"{name}:" + ".".join(str(r.id).split(".")[:2]),
                question=q,
                reference=str(r.reference_answer).strip(),
                student=str(r.student_answer).strip(),
                label3=l3,
                gold_score=THREE_TO_SCORE[l3],
                label5=l5,
            ))
    return items


def _load_mohler() -> List[Item]:
    df = pd.read_parquet(DATA_RAW / "ASAG" / "data" / "raw-oe-00001.parquet")
    items = []
    for r in df.itertuples(index=False):
        avg = float(r.score_avg)
        items.append(Item(
            id=f"mohler:{r.id}",
            dataset="mohler",
            split="all",
            question_id="mohler:" + str(r.id)[:7],   # e.g. E01.Q01
            question=str(r.question).strip(),
            reference=str(r.instructor_answer).strip(),
            student=str(r.student_answer).strip(),
            label3=mohler_label3(avg),
            gold_score=avg / 5.0,
        ))
    return items


def _group_split(items: List[Item], frac: float, seed: int):
    groups = [it.question_id for it in items]
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(np.zeros(len(items)), groups=groups))
    return [items[i] for i in a], [items[i] for i in b]


def build_all() -> Dict[str, List[Item]]:
    """Build processed datasets with fixed splits; writes data/processed/*.jsonl and splits.json."""
    out: Dict[str, List[Item]] = {}
    for name in ("scientsbank", "beetle"):
        items = _load_semeval(name)
        train = [it for it in items if it.split == "train"]
        rest = [it for it in items if it.split != "train"]
        tr, dv = _group_split(train, 0.2, SPLIT_SEED)
        for it in dv:
            it.split = "dev"
        out[name] = tr + dv + rest
    moh = _load_mohler()
    tr, rest = _group_split(moh, 0.4, SPLIT_SEED)
    dv, te = _group_split(rest, 0.5, SPLIT_SEED)
    for split, part in (("train", tr), ("dev", dv), ("test", te)):
        for it in part:
            it.split = split
    out["mohler"] = tr + dv + te

    manifest = {}
    for name, items in out.items():
        write_jsonl(DATA_PROCESSED / f"{name}.jsonl", (it.to_dict() for it in items))
        by_split: Dict[str, Dict] = {}
        for it in items:
            s = by_split.setdefault(it.split, {"n": 0, "questions": set(), "labels": {}})
            s["n"] += 1
            s["questions"].add(it.question_id)
            s["labels"][it.label3] = s["labels"].get(it.label3, 0) + 1
        manifest[name] = {
            k: {"n": v["n"], "n_questions": len(v["questions"]), "labels": v["labels"],
                "question_ids": sorted(v["questions"])}
            for k, v in by_split.items()
        }
        check_no_question_leak(items)
    write_json(DATA_PROCESSED / "splits.json", manifest)
    return out


def check_no_question_leak(items: List[Item]) -> None:
    """Dev/test question sets must not overlap with train (test_ua shares questions by design)."""
    q = {}
    for it in items:
        q.setdefault(it.split, set()).add(it.question_id)
    train = q.get("train", set())
    for split in ("dev", "test", "test_uq", "test_ud"):
        if split in q:
            overlap = train & q[split]
            assert not overlap, f"{items[0].dataset}: {len(overlap)} questions leak train->{split}"


def load(name: str, splits: Optional[List[str]] = None) -> List[Item]:
    path = DATA_PROCESSED / f"{name}.jsonl"
    if not path.exists():
        build_all()
    rows = read_jsonl(path)
    items = [Item(**r) for r in rows]
    if splits:
        items = [it for it in items if it.split in splits]
    return items
