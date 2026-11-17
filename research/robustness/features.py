"""Batched, cached extraction of the five Deterministic Linguistic Indicators.

Uses the exact models loaded by app.grader.HybridASAGGrader and reproduces its per-sample
logic (including the <5-word short-answer branch and the grammar boost inside the logic
score), but runs each model in batches on GPU. tests/robustness/test_features.py checks
equality against the grader's own per-sample methods.

Cache: experiments/cache/robustness/features.jsonl, keyed by a hash of
(question, reference, student) so adversarial variants never collide with originals.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .common import CACHE_DIR, append_jsonl, read_jsonl, text_key

FEATURE_CACHE = CACHE_DIR / "features.jsonl"


def logic_from_probs(fwd: Sequence[float], bwd: Sequence[float], grammar: float, th: Dict[str, float]) -> float:
    """Copy of HybridASAGGrader.get_logic_score post-processing (label order: C, E, N)."""
    fwd_c, fwd_e, fwd_n = float(fwd[0]), float(fwd[1]), float(fwd[2])
    bwd_c, bwd_e, bwd_n = float(bwd[0]), float(bwd[1]), float(bwd[2])
    max_c = max(fwd_c, bwd_c)
    avg_e = (fwd_e + bwd_e) / 2
    avg_n = (fwd_n + bwd_n) / 2
    if max_c > th["contradiction_high"]:
        s = (1 - max_c) * 0.15
    elif max_c > th["contradiction_moderate"]:
        s = avg_e * (1 - max_c * 0.3) + 0.25
    elif avg_n > 0.8:
        s = 0.55 + avg_e * 0.25 + (1 - max_c) * 0.15
    else:
        s = fwd_e * 0.55 + bwd_e * 0.35 + min(fwd_n, bwd_n) * 0.15 + 0.15
    if grammar is not None and grammar < th["grammar_good"] and max_c < th["contradiction_high"]:
        s = min(1.0, s + (th["grammar_good"] - grammar) * 0.3)
    return float(max(0.0, min(1.0, s)))


def word_match(reference: str, student: str) -> float:
    ref = set(reference.lower().split())
    stu = set(student.lower().split())
    return len(ref & stu) / len(ref) if ref else 0.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FeatureExtractor:
    """Wraps a HybridASAGGrader (LLM skipped) and extracts features in batches."""

    def __init__(self, batch_size: int = 32, grader=None):
        if grader is None:
            from app.config import GraderConfig
            from app.grader import HybridASAGGrader
            grader = HybridASAGGrader(GraderConfig(skip_llm=True, verbose=False), verbose=False)
        self.g = grader
        self.bs = batch_size
        self.th = grader.THRESHOLDS
        self.kw_thr = grader.KEYWORD_SIMILARITY_THRESHOLD
        self.min_words = grader.MIN_WORDS_FOR_FULL_ANALYSIS
        self._kw_cache: Dict[str, List[str]] = {}
        self._kw_emb_cache: Dict[str, np.ndarray] = {}

    # ---------------- model wrappers ----------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            return self.g.semantic_model.encode(
                texts, batch_size=self.bs, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float64)

    def _classify(self, tok, model, texts: List[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.bs):
                enc = tok(texts[i:i + self.bs], return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to(self.g.device)
                probs = torch.softmax(model(**enc).logits.float(), dim=-1)
                out.append(probs[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def _nli(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        return np.asarray(self.g.logic_model.predict(
            pairs, apply_softmax=True, batch_size=self.bs, show_progress_bar=False))

    def _keywords(self, question: str, reference: str) -> List[str]:
        """Reproduces the KeyBERT step of get_keyword_coverage (deterministic per reference)."""
        if reference in self._kw_cache:
            return self._kw_cache[reference]
        kb = self.g.keybert_model
        n = max(3, min(8, len(reference.split()) // 15 + 2))
        uni = kb.extract_keywords(reference, keyphrase_ngram_range=(1, 1), stop_words="english",
                                  top_n=n, use_mmr=True, diversity=0.5)
        bi = kb.extract_keywords(reference, keyphrase_ngram_range=(2, 2), stop_words="english",
                                 top_n=max(2, n // 2), use_mmr=True, diversity=0.7)
        allk = uni + [(k, s) for k, s in bi if s > 0.35]
        kws = [k for k, _ in allk if len(k) > 2]
        seen = set()
        kws = [x for x in kws if not (x.lower() in seen or seen.add(x.lower()))]
        self._kw_cache[reference] = kws
        if kws:
            self._kw_emb_cache[reference] = self._encode(kws)
        return kws

    def _coverage(self, reference: str, student: str, stu_emb: np.ndarray) -> Tuple[float, List[str], int]:
        kws = self._keywords("", reference)
        if not kws:
            return 1.0, [], 0
        kemb = self._kw_emb_cache[reference]
        low = student.lower()
        words = set(low.split())
        missing = []
        found = 0
        for i, k in enumerate(kws):
            kl = k.lower()
            if kl in low or (" " not in k and kl in words):
                found += 1
            elif _cos(kemb[i], stu_emb) >= self.kw_thr:
                found += 1
            else:
                missing.append(k)
        return found / len(kws), missing, len(kws)

    # ---------------- public ----------------
    def extract(self, rows: List[Dict], desc: str = "extract") -> List[Dict]:
        """rows: dicts with question/reference/student. Returns feature dicts in order."""
        results: List[Optional[Dict]] = [None] * len(rows)
        full_idx = []
        for i, r in enumerate(rows):
            stu = r["student"] or ""
            if not stu.strip():
                results[i] = self._short(r, empty=True)
            elif len(stu.split()) < self.min_words:
                results[i] = self._short(r)
            else:
                full_idx.append(i)
        step = 256
        for s in tqdm(range(0, len(full_idx), step), desc=desc, leave=False):
            chunk = [rows[i] for i in full_idx[s:s + step]]
            for i, feat in zip(full_idx[s:s + step], self._extract_full(chunk)):
                results[i] = feat
        return results

    def _short(self, r: Dict, empty: bool = False) -> Dict:
        wm = 0.0 if empty else word_match(r["reference"], r["student"])
        return {
            "semantic": wm, "coverage": wm, "formality": 0.5, "grammar": 0.5, "logic": wm,
            "logic_details": {}, "missing_keywords": [], "n_keywords": 0,
            "short_branch": True, "empty": empty,
        }

    def _extract_full(self, rows: List[Dict]) -> List[Dict]:
        refs = [r["reference"] for r in rows]
        stus = [r["student"] for r in rows]
        for r in rows:
            self._keywords(r["question"], r["reference"])
        uniq = list(dict.fromkeys(refs + stus))
        emb = dict(zip(uniq, self._encode(uniq)))
        gram = self._classify(self.g.grammar_tokenizer, self.g.grammar_model, stus)
        form = self._classify(self.g.formality_tokenizer, self.g.formality_model, stus)
        nli = self._nli([(a, b) for a, b in zip(refs, stus)] + [(b, a) for a, b in zip(refs, stus)])
        n = len(rows)
        out = []
        for j, r in enumerate(rows):
            sem = max(0.0, min(1.0, _cos(emb[r["reference"]], emb[r["student"]])))
            cov, missing, nk = self._coverage(r["reference"], r["student"], emb[r["student"]])
            fwd, bwd = nli[j], nli[n + j]
            details = {
                "contradiction": float(fwd[0]), "entailment": float(fwd[1]), "neutral": float(fwd[2]),
                "backward_contradiction": float(bwd[0]), "backward_entailment": float(bwd[1]),
                "backward_neutral": float(bwd[2]),
            }
            out.append({
                "semantic": sem, "coverage": float(cov), "formality": float(form[j]),
                "grammar": float(gram[j]),
                "logic": logic_from_probs(fwd, bwd, float(gram[j]), self.th),
                "logic_details": details, "missing_keywords": missing, "n_keywords": nk,
                "short_branch": False, "empty": False,
            })
        return out


class FeatureStore:
    """Content-keyed, append-only JSONL cache shared by clean and adversarial items."""

    def __init__(self, path=FEATURE_CACHE):
        self.path = path
        self._d: Dict[str, Dict] = {r["key"]: r["f"] for r in read_jsonl(path)}
        self._extractor: Optional[FeatureExtractor] = None

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str) -> Dict:
        return self._d[key]

    def ensure(self, rows: List[Dict], desc: str = "extract") -> List[Dict]:
        """Return features for rows, extracting (and persisting) only missing ones."""
        keys = [text_key(r["question"], r["reference"], r["student"]) for r in rows]
        todo, seen = [], set()
        for k, r in zip(keys, rows):
            if k not in self._d and k not in seen:
                todo.append((k, r))
                seen.add(k)
        if todo:
            if self._extractor is None:
                self._extractor = FeatureExtractor()
            chunk = 2048
            for s in range(0, len(todo), chunk):
                part = todo[s:s + chunk]
                feats = self._extractor.extract([r for _, r in part], desc=f"{desc} {s}/{len(todo)}")
                new = [{"key": k, "f": f} for (k, _), f in zip(part, feats)]
                append_jsonl(self.path, new)
                for row in new:
                    self._d[row["key"]] = row["f"]
        return [self._d[k] for k in keys]
