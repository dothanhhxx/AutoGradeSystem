"""Shared paths, environment setup and small JSON/JSONL helpers."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

# Model cache: the C: drive on the dev machine is nearly full, so new downloads go to
# D:\hf_cache when it exists. Override with HF_HUB_CACHE as usual.
if "HF_HUB_CACHE" not in os.environ and Path(r"D:\hf_cache").exists():
    os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ADV_DIR = ROOT / "data" / "adversarial"
CACHE_DIR = ROOT / "experiments" / "cache" / "robustness"
RESULTS_DIR = ROOT / "experiments" / "results" / "robustness"

for _d in (DATA_PROCESSED, ADV_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELS3 = ["incorrect", "partial", "correct"]
LABEL_TO_ORD = {l: i for i, l in enumerate(LABELS3)}
DATASETS = ["scientsbank", "beetle", "mohler"]
# Splits used for reporting attack results (never used for fitting or detector training).
REPORT_SPLITS = {
    "scientsbank": ["test_ua", "test_uq"],
    "beetle": ["test_ua", "test_uq"],
    "mohler": ["test"],
}
FIT_SPLITS = ["train", "dev"]


def text_key(question: str, reference: str, student: str) -> str:
    """Content hash: adversarial variants share an id prefix but differ in text."""
    h = hashlib.sha1()
    h.update("\x1f".join([question or "", reference or "", student or ""]).encode("utf-8"))
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"
