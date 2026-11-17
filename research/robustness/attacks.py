"""Adversarial answer generators.

Threat model: the student knows the topic's key terms (approximated by keywords of the
reference answer) but not the reference answer itself, and can write freely.

| code | name                       | source answer              | expected  | generator |
|------|----------------------------|----------------------------|-----------|-----------|
| A1a  | keyword list               | none (keywords only)       | incorrect | rule      |
| A1b  | off-topic + keywords       | answer to another question | incorrect | rule      |
| A1c  | keyword filler sentence    | none (template)            | incorrect | rule      |
| A2   | fluent but wrong           | correct answer             | incorrect | LLM       |
| A3   | negation / antonym flip    | correct answer             | incorrect | rule      |
| A4   | meaning-preserving rewrite | correct answer             | correct   | LLM       |
| A5   | restate the question       | none (question text)       | incorrect | rule      |
| A6   | A1b with broken grammar    | A1b text                   | incorrect | rule      |
| A7   | <5-word keyword answer     | none (keywords only)       | incorrect | rule      |

A6 and A7 are white-box probes of this grader (grammar boost inside the logic score,
short-answer word-match branch) and are reported separately from A1-A5.
"""

import random
import re
from typing import Dict, List, Optional

from tqdm import tqdm

from .common import ADV_DIR, REPORT_SPLITS, read_jsonl, write_jsonl
from .data import Item, load

RULE_ATTACKS = ["A1a", "A1b", "A1c", "A3", "A5", "A6", "A7"]
LLM_ATTACKS = ["A2", "A4"]
EXPECTED = {"A1a": "incorrect", "A1b": "incorrect", "A1c": "incorrect", "A2": "incorrect",
            "A3": "incorrect", "A4": "correct", "A5": "incorrect", "A6": "incorrect", "A7": "incorrect"}
FAMILY = {"A1a": "stuffing", "A1b": "stuffing", "A1c": "stuffing", "A2": "fluent_wrong",
          "A3": "negation", "A4": "paraphrase", "A5": "restate", "A6": "whitebox", "A7": "whitebox"}
GEN_SPLITS = {  # attack sources: report splits + train/dev for the detector
    name: {**{s: None for s in REPORT_SPLITS[name]}, "train": None, "dev": None} for name in REPORT_SPLITS
}
DEV_FRACTION = 0.5  # dev gets half the per-split budget

_STOP = set("""a an the of to in on for and or is are was were be been being it its this that these those
with as by at from into than then so such can will would should could may might do does did not no
there their they them he she his her we our you your i me my which who whom what when where why how
because also if but about more most less very just only same other each all any both""".split())

_FILLERS = [
    "The answer is about {k} and how they work.",
    "This is related to {k}, which are important here.",
    "It depends on {k} in this situation.",
    "We need to think about {k} to explain this.",
]

# Negation / antonym rules, applied to the first match only.
_AUX = r"\b(is|are|was|were|will|would|can|could|should|does|do|did|has|have|had|must)\b"
_ANTONYMS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("more", "less"), ("higher", "lower"), ("larger", "smaller"), ("bigger", "smaller"),
    ("faster", "slower"), ("open", "closed"), ("series", "parallel"), ("positive", "negative"),
    ("same", "different"), ("true", "false"), ("always", "never"), ("before", "after"),
    ("first", "last"), ("connected", "separated"), ("conductor", "insulator"),
    ("conductors", "insulators"), ("attract", "repel"), ("attracts", "repels"),
    ("hot", "cold"), ("heavier", "lighter"),
    ("lifo", "fifo"), ("smallest", "largest"), ("minimum", "maximum"),
    ("inside", "outside"),
]
_ANT = {}
for _a, _b in _ANTONYMS:
    _ANT.setdefault(_a, _b)
    _ANT.setdefault(_b, _a)


def content_words(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    out, seen = [], set()
    for t in toks:
        if t not in _STOP and len(t) > 2 and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def negate(text: str) -> Optional[Dict]:
    """Return {'text', 'op'} or None if no rule applies."""
    m = re.search(r"\b(not|never|no longer)\s+", text, flags=re.I)
    if m:
        return {"text": (text[:m.start()] + text[m.end():]).strip(), "op": f"remove '{m.group(1)}'"}
    m = re.search(r"\b(\w+)n't\b", text, flags=re.I)
    if m:
        base = {"wo": "will", "ca": "can", "sha": "shall"}.get(m.group(1).lower(), m.group(1))
        return {"text": text[:m.start()] + base + text[m.end():], "op": "remove n't"}
    m = re.search(r"\bcannot\b", text, flags=re.I)
    if m:
        return {"text": text[:m.start()] + _match_case(m.group(0), "can") + text[m.end():], "op": "cannot->can"}
    for w in re.finditer(r"[A-Za-z]+", text):
        low = w.group(0).lower()
        if low in _ANT:
            rep = _match_case(w.group(0), _ANT[low])
            return {"text": text[:w.start()] + rep + text[w.end():], "op": f"antonym {low}->{_ANT[low]}"}
    for m in re.finditer(_AUX, text, flags=re.I):
        nxt = text[m.end():].lstrip().split(" ", 1)[0].lower()
        if nxt in {"only", "also", "not"}:
            continue
        return {"text": text[:m.end()] + " not" + text[m.end():], "op": f"insert not after '{m.group(1)}'"}
    return None


def break_grammar(text: str, rng: random.Random) -> str:
    """Drop articles/auxiliaries and shuffle word order inside each sentence; keeps all content words."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sents:
        words = [w for w in s.split() if w.lower() not in {"a", "an", "the", "is", "are", "of", "to"}]
        rng.shuffle(words)
        out.append(" ".join(words).lower().strip(".") + ("." if words else ""))
    return " ".join(out)


class KeywordSource:
    """Keywords per reference: the grader's own KeyBERT keywords when available (white-box upper
    bound), otherwise reference content words. Both are recorded in meta."""

    def __init__(self, use_keybert: bool = True):
        self.kb = None
        if use_keybert:
            from keybert import KeyBERT
            self.kb = KeyBERT(model="all-MiniLM-L6-v2")
        self._c: Dict[str, List[str]] = {}

    def get(self, reference: str, k: int = 5) -> List[str]:
        if reference not in self._c:
            kws = []
            if self.kb is not None:
                kws = [w for w, _ in self.kb.extract_keywords(
                    reference, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=8,
                    use_mmr=True, diversity=0.5) if len(w) > 2]
            if len(kws) < 3:
                kws = list(dict.fromkeys(kws + content_words(reference)))
            self._c[reference] = kws
        return self._c[reference][:k]


def _record(src: Item, attack: str, text: str, meta: Dict, source_student: Optional[str]) -> Dict:
    return {
        "id": f"{src.id}#{attack}", "orig_id": src.id, "dataset": src.dataset, "split": src.split,
        "question_id": src.question_id, "question": src.question, "reference": src.reference,
        "student": text.strip(), "attack": attack, "family": FAMILY[attack],
        "expected_label": EXPECTED[attack], "source_label": src.label3,
        "source_student": source_student, "meta": meta,
    }


def _sample(items: List[Item], n: int, rng: random.Random) -> List[Item]:
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def _one_per_question(items: List[Item], n: int, rng: random.Random) -> List[Item]:
    """Question-level attacks (A1a, A1c, A5, A7) do not depend on the student answer:
    use one source per (question, reference) to avoid duplicates."""
    by = {}
    for it in items:
        by.setdefault((it.question_id, it.reference), []).append(it)
    reps = [rng.choice(v) for v in by.values()]
    rng.shuffle(reps)
    return reps[:n]


def rule_attacks_for_split(items: List[Item], pool: List[Item], n: int, kw: KeywordSource,
                           seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    correct = [it for it in items if it.label3 == "correct"]

    for src in _one_per_question(items, n, rng):
        kws = kw.get(src.reference)
        out.append(_record(src, "A1a", ", ".join(kws) + ".", {"keywords": kws}, None))
        k3 = kws[:3]
        txt = rng.choice(_FILLERS).format(k=", ".join(k3[:-1]) + " and " + k3[-1] if len(k3) > 1 else k3[0])
        out.append(_record(src, "A1c", txt, {"keywords": k3}, None))
        out.append(_record(src, "A5", src.question, {}, None))
        out.append(_record(src, "A7", " ".join(kws[:4]), {"keywords": kws[:4]}, None))

    other_q = {}
    for it in pool:
        other_q.setdefault(it.question_id, []).append(it)
    qids = list(other_q)
    for src in _sample(items, n, rng):
        kws = kw.get(src.reference)
        cands = [q for q in qids if q != src.question_id]
        donor = rng.choice(other_q[rng.choice(cands)])
        base = donor.student.rstrip(".") + "."
        txt = base + " " + " ".join(f"{w}" for w in kws[:4]).capitalize() + "."
        meta = {"keywords": kws[:4], "donor_id": donor.id}
        out.append(_record(src, "A1b", txt, meta, donor.student))
        out.append(_record(src, "A6", break_grammar(txt, rng), {**meta, "base_attack": "A1b"}, txt))

    for src in _sample(correct, n, rng):
        neg = negate(src.student)
        if neg and neg["text"].strip() and neg["text"].strip().lower() != src.student.strip().lower():
            out.append(_record(src, "A3", neg["text"], {"op": neg["op"]}, src.student))
    return out


def _sources_for(name: str):
    allitems = load(name)
    by_split = {}
    for it in allitems:
        by_split.setdefault(it.split, []).append(it)
    return allitems, by_split


def generate_all(datasets: List[str], use_llm: bool = False, n_per_attack: int = 300, seed: int = 1234):
    kw = KeywordSource()
    for name in datasets:
        allitems, by_split = _sources_for(name)
        rows: List[Dict] = []
        for i, split in enumerate(GEN_SPLITS[name]):
            n = int(n_per_attack * (DEV_FRACTION if split == "dev" else 1))
            items = by_split.get(split, [])
            # Donors for A1b come from the same split so that no train text enters test attacks.
            rows += rule_attacks_for_split(items, items, n, kw, seed + 17 * i)
        path = ADV_DIR / f"{name}_rule.jsonl"
        write_jsonl(path, rows)
        print(f"{name}: {len(rows)} rule-based attacks -> {path}")
        if use_llm:
            llm_attacks(name, by_split, n_per_attack, seed)


# ---------------------------------------------------------------------------
# LLM attacks (A2, A4)
# ---------------------------------------------------------------------------
GEN_MODEL = "microsoft/Phi-3.5-mini-instruct"

PROMPTS = {
    "A2": (
        "You help build a test set for automatic grading.\n"
        "Question: {q}\nA correct student answer: {a}\n\n"
        "Rewrite the student answer so that it still sounds fluent and confident and keeps the same "
        "technical terms, but is factually WRONG: change one key fact, relation, cause or direction "
        "so that a teacher would mark it incorrect. Never say or hint that the answer is wrong. "
        "Use about {n} words and a student-like style. "
        "Output only the rewritten answer, nothing else."
    ),
    "A4": (
        "Question: {q}\nStudent answer: {a}\n\n"
        "Paraphrase the student answer with different wording and sentence structure while keeping "
        "exactly the same meaning and the same level of correctness. Do not add, remove, explain or "
        "fix any information. Use about {n} words and a student-like style. "
        "Output only the paraphrase, nothing else."
    ),
}


def clean_generation(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^(here is |here's )?(the |a |an )?(correct |incorrect |wrong |rewritten |paraphrased )?"
               r"(student )?(answer|paraphrase|rewrite|version)\s*:\s*", "", t, flags=re.I)
    t = t.split("\n\n")[0].strip()
    if len(t) > 1 and t[0] in "\"'" and t[-1] in "\"'":
        t = t[1:-1].strip()
    return t


def llm_attacks(name: str, by_split: Dict[str, List[Item]], n_per_attack: int, seed: int):
    from .llm import ChatLLM
    path = ADV_DIR / f"{name}_llm.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    llm = ChatLLM.get(GEN_MODEL)
    rng = random.Random(seed + 99)
    todo = []
    for split in GEN_SPLITS[name]:
        # LLM budget (generation is ~1 s/answer on the 8 GB GPU): 200 per report split, 150 train, 75 dev.
        n = {"train": 150, "dev": 75}.get(split, min(n_per_attack, 200))
        correct = [it for it in by_split.get(split, []) if it.label3 == "correct"]
        for src in _sample(correct, n, rng):
            for atk in LLM_ATTACKS:
                if f"{src.id}#{atk}" not in done:
                    todo.append((src, atk))
    print(f"{name}: {len(todo)} LLM generations to run with {GEN_MODEL}")
    from .common import append_jsonl
    # Length-sorted batches so short answers do not wait for the longest one.
    todo.sort(key=lambda t: len(t[0].student.split()))
    bs = 16
    for s in tqdm(range(0, len(todo), bs), desc=f"{name} LLM attacks"):
        part = todo[s:s + bs]
        prompts = [PROMPTS[a].format(q=src.question, a=src.student, n=max(5, len(src.student.split())))
                   for src, a in part]
        longest = max(len(src.student.split()) for src, _ in part)
        outs = llm.generate(prompts, max_new_tokens=min(200, int(2.2 * longest) + 30),
                            temperature=0.7, seed=seed + s)
        recs = []
        for (src, atk), raw in zip(part, outs):
            txt = clean_generation(raw)
            if txt:
                ratio = len(txt.split()) / max(1, len(src.student.split()))
                recs.append(_record(src, atk, txt, {"model": GEN_MODEL, "raw": raw,
                                                    "len_ratio": round(ratio, 3)}, src.student))
        append_jsonl(path, recs)


def load_attacks(name: str, splits: Optional[List[str]] = None) -> List[Dict]:
    rows = read_jsonl(ADV_DIR / f"{name}_rule.jsonl") + read_jsonl(ADV_DIR / f"{name}_llm.jsonl")
    if splits:
        rows = [r for r in rows if r["split"] in splits]
    return rows
