"""External baselines evaluated under the same attacks.

- deberta_s{seed}: microsoft/deberta-v3-base fine-tuned per dataset on (question+reference, student),
  3-way classification, epoch chosen by dev QWK.
- llm_holistic / llm_analytic: Qwen2.5-3B-Instruct (the standalone LLM baseline of the paper),
  greedy decoding. 'analytic' scores the same five criteria as the Hybrid grader, so the
  per-criterion attack analysis can be compared across architectures.

Predictions are cached in experiments/cache/robustness/preds/<dataset>__<system>.jsonl keyed by
text hash, and are picked up automatically by pipeline.predict_all.
"""

import json
import math
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .attacks import load_attacks
from .common import FIT_SPLITS, LABELS3, REPORT_SPLITS, append_jsonl, read_jsonl, text_key
from .data import load
from .metrics import qwk

PRED_DIR = None  # set lazily to avoid importing pipeline at module import


def _pred_dir():
    from .pipeline import PRED_DIR as P
    return P


def _rows_to_score(name: str, include_fit_attacks: bool = False, include_ud: bool = True) -> List[Dict]:
    """Clean report splits + attack rows of report splits (deduplicated by text)."""
    splits = REPORT_SPLITS[name] + (["test_ud"] if name == "scientsbank" and include_ud else [])
    rows = [{"question": it.question, "reference": it.reference, "student": it.student}
            for it in load(name, splits)]
    rows += [{"question": r["question"], "reference": r["reference"], "student": r["student"]}
             for r in load_attacks(name, REPORT_SPLITS[name] + (FIT_SPLITS if include_fit_attacks else []))]
    seen, out = set(), []
    for r in rows:
        k = text_key(r["question"], r["reference"], r["student"])
        if k not in seen:
            seen.add(k)
            out.append({**r, "key": k})
    return out


# ---------------------------------------------------------------------------
# Fine-tuned encoder
# ---------------------------------------------------------------------------
FT_MODEL = "microsoft/deberta-v3-base"


def _encode_pairs(tok, rows, max_len=256):
    a = [f"{r['question']} [SEP] {r['reference']}" for r in rows]
    b = [r["student"] for r in rows]
    return tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


@torch.no_grad()
def _ft_predict(model, tok, rows, bs=64):
    model.eval()
    probs = []
    for i in range(0, len(rows), bs):
        enc = _encode_pairs(tok, rows[i:i + bs]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 3))


# Attempt ladder: a run whose best dev QWK stays below MIN_DEV_QWK has collapsed to one class
# (seen on Mohler, 76% 'correct'); it is retrained with the same seed, class-weighted loss,
# more epochs and a higher learning rate. All attempts are logged in the .meta.json file.
MIN_DEV_QWK = 0.2
ATTEMPTS = [dict(epochs=3, lr=2e-5, class_weighted=False),
            dict(epochs=6, lr=3e-5, class_weighted=True)]


def finetune_one(name: str, seed: int, bs: int = 16):
    from .common import write_json
    out_path = _pred_dir() / f"{name}__deberta_s{seed}.jsonl"
    if out_path.exists():
        print(f"[skip] {out_path.name} exists")
        return
    log = []
    for cfg in ATTEMPTS:
        model, tok, best = _train(name, seed, bs=bs, **cfg)
        log.append({**cfg, "best_dev_qwk": best})
        if best >= MIN_DEV_QWK:
            break
        print(f"  {name} seed={seed}: dev QWK {best:.3f} < {MIN_DEV_QWK}, retrying with {ATTEMPTS[-1]}")
        del model
        torch.cuda.empty_cache()
    rows = _rows_to_score(name)
    p = _ft_predict(model, tok, rows)
    recs = [{"key": r["key"], "label": LABELS3[int(pp.argmax())], "score": float((pp[1] + 2 * pp[2]) / 2),
             "probs": [float(x) for x in pp]} for r, pp in zip(rows, p)]
    append_jsonl(out_path, recs)
    write_json(out_path.with_suffix(".meta.json"), {"dataset": name, "seed": seed, "attempts": log})
    print(f"  wrote {len(recs)} predictions -> {out_path.name} (best dev QWK {best:.3f})")
    del model
    torch.cuda.empty_cache()


def _train(name: str, seed: int, epochs: int, lr: float, class_weighted: bool, bs: int = 16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    # fp32 master weights: transformers v5 loads the checkpoint dtype (fp16) by default, and AdamW's
    # eps=1e-8 underflows in fp16 -> NaN embeddings after the first step.
    model = AutoModelForSequenceClassification.from_pretrained(
        FT_MODEL, num_labels=3, dtype=torch.float32).to("cuda")
    train = load(name, ["train"])
    dev = load(name, ["dev"])
    tr_rows = [{"question": i.question, "reference": i.reference, "student": i.student} for i in train]
    y = torch.tensor([i.label_ord for i in train])
    dv_rows = [{"question": i.question, "reference": i.reference, "student": i.student} for i in dev]
    dv_y = np.array([i.label_ord for i in dev])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(seed)
    weight = None
    if class_weighted:
        cnt = np.bincount(y.numpy(), minlength=3).astype(float)
        weight = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device="cuda")
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        for s in tqdm(range(0, len(order), bs), desc=f"{name} s{seed} ep{ep}", leave=False):
            idx = order[s:s + bs]
            enc = _encode_pairs(tok, [tr_rows[i] for i in idx]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y[idx].to("cuda"),
                                                         weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        k = qwk(dv_y, _ft_predict(model, tok, dv_rows).argmax(1))
        print(f"  {name} seed={seed} epoch={ep} dev_qwk={k:.3f}")
        if k > best:
            best = k
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, best


def run_finetune_all(datasets: List[str], seeds=(13, 42, 87)):
    for name in datasets:
        for seed in seeds:
            finetune_one(name, seed)


# ---------------------------------------------------------------------------
# LLM graders
# ---------------------------------------------------------------------------
LLM_GRADER = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = "You are a strict but fair teacher grading short answers. Reply with JSON only."
HOLISTIC = (
    "Question: {q}\nReference answer: {r}\nStudent answer: {s}\n\n"
    "Grade the student answer against the reference answer.\n"
    'Reply with JSON: {{"label": "correct" | "partially_correct" | "incorrect", "score": <0-10>}}'
)
ANALYTIC = (
    "Question: {q}\nReference answer: {r}\nStudent answer: {s}\n\n"
    "Rate the student answer on each criterion from 0 to 10:\n"
    "- semantic: same meaning as the reference\n"
    "- coverage: covers the key concepts of the reference\n"
    "- grammar: grammatical correctness\n"
    "- formality: academic register\n"
    "- logic: logically consistent with the reference, no contradiction\n"
    "Then give the overall label.\n"
    'Reply with JSON: {{"semantic": n, "coverage": n, "grammar": n, "formality": n, "logic": n, '
    '"label": "correct" | "partially_correct" | "incorrect"}}'
)
_LABEL_MAP = {"correct": "correct", "partially_correct": "partial", "partially correct": "partial",
              "partial": "partial", "incorrect": "incorrect", "wrong": "incorrect"}


def parse_llm(text: str, mode: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        obj = {}
        lm = re.search(r'"?label"?\s*:\s*"?([a-z_ ]+)"?', text, flags=re.I)
        if lm:
            obj["label"] = lm.group(1)
        for k in ("score", "semantic", "coverage", "grammar", "formality", "logic"):
            km = re.search(rf'"?{k}"?\s*:\s*([0-9.]+)', text, flags=re.I)
            if km:
                obj[k] = km.group(1)
    lab = _LABEL_MAP.get(str(obj.get("label", "")).strip().lower())
    if lab is None:
        return None
    out = {"label": lab}
    try:
        if mode == "holistic":
            out["score"] = min(1.0, max(0.0, float(obj.get("score", {"incorrect": 0, "partial": 5, "correct": 10}[lab])) / 10))
        else:
            crit = {k: min(1.0, max(0.0, float(obj[k]) / 10))
                    for k in ("semantic", "coverage", "grammar", "formality", "logic") if k in obj}
            out["per_criterion"] = crit
            out["score"] = float(np.mean(list(crit.values()))) if crit else {"incorrect": 0, "partial": .5, "correct": 1}[lab]
    except (TypeError, ValueError):
        out["score"] = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}[lab]
    return out


def run_llm(name: str, mode: str, limit: Optional[int] = None, bs: int = 24):
    from .llm import ChatLLM
    path = _pred_dir() / f"{name}__llm_{mode}.jsonl"
    done = {r["key"] for r in read_jsonl(path)}
    # test_ud (4.5k clean answers, no attacks) is skipped for the LLM to bound GPU time.
    rows = [r for r in _rows_to_score(name, include_ud=False) if r["key"] not in done]
    if limit:
        rows = rows[:limit]
    if not rows:
        return
    llm = ChatLLM.get(LLM_GRADER)
    tmpl = HOLISTIC if mode == "holistic" else ANALYTIC
    n_fail = 0
    for s in tqdm(range(0, len(rows), bs), desc=f"{name} llm_{mode}"):
        part = rows[s:s + bs]
        outs = llm.generate([tmpl.format(q=r["question"], r=r["reference"], s=r["student"]) for r in part],
                            max_new_tokens=40 if mode == "holistic" else 90, temperature=0.0, system=SYSTEM)
        recs = []
        for r, o in zip(part, outs):
            p = parse_llm(o, mode)
            if p is None:
                n_fail += 1
                # Unparseable output is graded 'incorrect' (a deployed system must return a grade).
                p = {"label": "incorrect", "score": 0.0, "parse_fail": True}
            recs.append({"key": r["key"], **p, "raw": o})
        append_jsonl(path, recs)
    print(f"{name} llm_{mode}: {len(rows)} graded, {n_fail} parse failures")


def run_llm_all(datasets: List[str], mode: str = "both", limit: Optional[int] = None):
    modes = ["holistic", "analytic"] if mode == "both" else [mode]
    for m in modes:
        for name in datasets:
            run_llm(name, m, limit)
