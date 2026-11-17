"""Detector-aware (adaptive) attacks.

AQ: black-box greedy search. The attacker can query the deployed system and sees the grade and
    whether the answer was flagged. Starting from an answer to a different question, it applies
    keyword edits to maximise the grade while staying unflagged by D2. Success is recorded at
    query budgets 10 / 50 / 150 (anytime). Transfer to the unseen D3 detector is also reported.
AL: (--llm) the generator LLM writes coherent sentences that weave the key terms in without
    answering the question, aiming to keep the NLI signal neutral rather than contradictory.
"""

import random
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .attacks import GEN_MODEL, KeywordSource, clean_generation
from .common import ADV_DIR, LABEL_TO_ORD, REPORT_SPLITS, RESULTS_DIR, append_jsonl, read_jsonl
from .data import load
from .detector import Detectors, _dataset, _frame

BUDGETS = [10, 50, 150]
N_SOURCES = 100
_BRIDGES = ["{k}", "which involves {k}", "because of the {k}", "and the {k} matters", "using {k}"]


def _edits(text: str, kws: List[str], rng: random.Random, n: int = 8) -> List[str]:
    out = set()
    sents = [s for s in text.split(". ") if s]
    tries = 0
    while len(out) < n and tries < 50:
        tries += 1
        op = rng.random()
        k = rng.choice(kws)
        if op < 0.6:
            b = rng.choice(_BRIDGES).format(k=k)
            out.add(text.rstrip(". ") + (", " if rng.random() < 0.5 else " ") + b + ".")
        elif op < 0.8 and len(sents) > 1:
            out.add(". ".join(s for j, s in enumerate(sents) if j != rng.randrange(len(sents))).strip() + ".")
        else:
            words = text.split()
            pos = rng.randrange(len(words) + 1)
            out.add(" ".join(words[:pos] + [k] + words[pos:]))
    return list(out)


def greedy_search(name: str, hybrid, det: Detectors, sources: List[Dict], kw: KeywordSource, seed: int = 0):
    from .pipeline import feats_for
    rng = random.Random(seed)
    thr2, thr3 = det.thr["D2_disagreement_lr"], det.thr["D3_boosting"]
    results = []
    for src in tqdm(sources, desc=f"{name} adaptive search"):
        kws = kw.get(src["reference"], k=6)
        cur = src["start"]
        queries, best = 0, None
        record = {}

        def evaluate(texts):
            rows = [{"question": src["question"], "reference": src["reference"], "student": t} for t in texts]
            f = feats_for(rows)
            X = _frame(rows, f)
            s = det.scores(X)
            g = [hybrid.score(x) for x in f]
            labs = hybrid.predict(f)[1]
            return [{"text": t, "grade": gg, "label": lb, "d2": float(a), "d3": float(b)}
                    for t, gg, lb, a, b in zip(texts, g, labs, s["D2_disagreement_lr"], s["D3_boosting"])]

        state = evaluate([cur])[0]
        queries += 1

        def snapshot():
            for b in BUDGETS:
                if queries <= b:
                    record[b] = dict(best) if best else None

        if LABEL_TO_ORD[state["label"]] > 0 and state["d2"] <= thr2:
            best = dict(state)
        snapshot()
        while queries < max(BUDGETS):
            cands = _edits(state["text"], kws, rng)[: max(BUDGETS) - queries]
            if not cands:
                break
            ev = evaluate(cands)
            queries += len(ev)
            feas = [e for e in ev if e["d2"] <= thr2]
            pool = feas if feas else ev
            # Objective: grade if unflagged, otherwise heavily penalised.
            cand = max(pool, key=lambda e: e["grade"] - (0 if e["d2"] <= thr2 else 100))
            if (cand["grade"] - (0 if cand["d2"] <= thr2 else 100)) >= \
                    (state["grade"] - (0 if state["d2"] <= thr2 else 100)):
                state = cand
            ok = LABEL_TO_ORD[state["label"]] > 0 and state["d2"] <= thr2
            if ok and (best is None or state["grade"] > best["grade"]):
                best = dict(state)
            snapshot()
        for b in BUDGETS:
            r = record.get(b)
            results.append({"dataset": name, "orig_id": src["orig_id"], "budget": b,
                            "success_vs_D2": r is not None,
                            "evades_D3": bool(r is not None and r["d3"] <= thr3),
                            "graded_correct": bool(r is not None and r["label"] == "correct"),
                            "grade": r["grade"] if r else np.nan, "text": r["text"] if r else None})
    return results


def llm_adaptive(name: str, srcs: List[Dict], kw: KeywordSource) -> List[Dict]:
    from .llm import ChatLLM
    path = ADV_DIR / f"{name}_adaptive_llm.jsonl"
    have = {r["orig_id"] for r in read_jsonl(path)}
    todo = [s for s in srcs if s["orig_id"] not in have]
    if todo:
        llm = ChatLLM.get(GEN_MODEL)
        prompt = ("Question: {q}\nKey terms: {k}\n\nWrite a fluent, confident one- or two-sentence "
                  "student answer that naturally uses all of the key terms, but does NOT actually answer "
                  "the question correctly (talk around it). Never say it is wrong. Output only the answer.")
        for s in tqdm(range(0, len(todo), 16), desc=f"{name} adaptive LLM"):
            part = todo[s:s + 16]
            outs = llm.generate([prompt.format(q=p["question"], k=", ".join(kw.get(p["reference"], 5)))
                                 for p in part], max_new_tokens=100, temperature=0.7, seed=s)
            append_jsonl(path, [{**p, "student": clean_generation(o), "raw": o} for p, o in zip(part, outs)])
    return read_jsonl(path)


def run(datasets: List[str], use_llm: bool = False):
    from .pipeline import feats_for, load_feature_systems
    kw = KeywordSource()
    all_rows, llm_rows = [], []
    for name in datasets:
        Xtr, ytr, _, _ = _dataset(name, ["train"])
        Xdv, ydv, _, _ = _dataset(name, ["dev"])
        det = Detectors().fit(Xtr, ytr).calibrate(Xdv, ydv)
        hybrid = load_feature_systems(name)[0]
        items = load(name, REPORT_SPLITS[name])
        rng = random.Random(5)
        rng.shuffle(items)
        by_q = {}
        for it in items:
            by_q.setdefault(it.question_id, []).append(it)
        srcs = []
        for it in items[:N_SOURCES]:
            others = [q for q in by_q if q != it.question_id]
            donor = rng.choice(by_q[rng.choice(others)])
            srcs.append({"orig_id": it.id, "question": it.question, "reference": it.reference,
                         "start": donor.student})
        all_rows += greedy_search(name, hybrid, det, srcs, kw)
        if use_llm:
            rows = llm_adaptive(name, srcs, kw)
            rr = [{"question": r["question"], "reference": r["reference"], "student": r["student"]} for r in rows]
            f = feats_for(rr)
            s = det.scores(_frame(rr, f))
            labs = hybrid.predict(f)[1]
            for r, lb, a, b in zip(rows, labs, s["D2_disagreement_lr"], s["D3_boosting"]):
                llm_rows.append({"dataset": name, "orig_id": r["orig_id"], "label": lb,
                                 "graded_above_incorrect": LABEL_TO_ORD[lb] > 0,
                                 "flag_D2": a > det.thr["D2_disagreement_lr"],
                                 "flag_D3": b > det.thr["D3_boosting"], "text": r["student"]})
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "adaptive_search.csv", index=False)
    summ = df.groupby(["dataset", "budget"])[["success_vs_D2", "evades_D3", "graded_correct"]].mean()
    print(summ.round(3).to_string())
    if llm_rows:
        dl = pd.DataFrame(llm_rows)
        dl["success_vs_D2"] = dl["graded_above_incorrect"] & ~dl["flag_D2"]
        dl.to_csv(RESULTS_DIR / "adaptive_llm.csv", index=False)
        print(dl.groupby("dataset")[["graded_above_incorrect", "flag_D2", "flag_D3", "success_vs_D2"]]
              .mean().round(3).to_string())
