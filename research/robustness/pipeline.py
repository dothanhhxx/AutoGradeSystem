"""Fit the system under test, evaluate on clean data, evaluate attacks for every system."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .attacks import FAMILY, load_attacks
from .common import (CACHE_DIR, LABEL_TO_ORD, REPORT_SPLITS, RESULTS_DIR, git_commit, read_json,
                     read_jsonl, text_key, write_json)
from .data import Item, load
from .features import FeatureStore
from .metrics import agreement, attack_success, bootstrap_ci, invariance_failure, paired_delta, qwk
from .scoring import FEATS, FeatureSystem, build_feature_systems

PRED_DIR = CACHE_DIR / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
_STORE: Optional[FeatureStore] = None


def store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = FeatureStore()
    return _STORE


def feats_for(rows: List) -> List[Dict]:
    as_dict = [r if isinstance(r, dict) else {"question": r.question, "reference": r.reference,
                                              "student": r.student} for r in rows]
    return store().ensure(as_dict)


def _key(r) -> str:
    if isinstance(r, dict):
        return text_key(r["question"], r["reference"], r["student"])
    return r.key


# ---------------------------------------------------------------------------
# System registry: feature systems (computed) + external systems (cached predictions)
# ---------------------------------------------------------------------------
def sut_path(name: str):
    return RESULTS_DIR / name / "sut_config.json"


def load_feature_systems(name: str) -> List[FeatureSystem]:
    return [FeatureSystem.from_dict(d) for d in read_json(sut_path(name))["systems"]]


def external_predictions(name: str) -> Dict[str, Dict[str, Dict]]:
    """{system_name: {text_key: {"score", "label", "per_criterion"?}}} from preds/<dataset>__*.jsonl."""
    out: Dict[str, Dict[str, Dict]] = {}
    for p in sorted(PRED_DIR.glob(f"{name}__*.jsonl")):
        sysname = p.stem.split("__", 1)[1]
        out[sysname] = {r["key"]: r for r in read_jsonl(p)}
    return out


def predict_all(name: str, rows: List, systems: Optional[List[FeatureSystem]] = None,
                ext: Optional[Dict] = None) -> Dict[str, Dict[str, List]]:
    """{system: {"score": [...], "label": [...], "covered": [bool]}} aligned with rows."""
    systems = systems if systems is not None else load_feature_systems(name)
    ext = ext if ext is not None else external_predictions(name)
    f = feats_for(rows)
    res = {}
    for s in systems:
        sc, lb = s.predict(f)
        res[s.name] = {"score": sc, "label": lb, "covered": [True] * len(rows)}
    keys = [_key(r) for r in rows]
    for sysname, table in ext.items():
        sc, lb, cov = [], [], []
        for k in keys:
            hit = table.get(k)
            cov.append(hit is not None)
            sc.append(hit["score"] if hit else np.nan)
            lb.append(hit["label"] if hit else None)
        res[sysname] = {"score": sc, "label": lb, "covered": cov}
    return res


# ---------------------------------------------------------------------------
# Fit + clean evaluation
# ---------------------------------------------------------------------------
def fit_systems(name: str) -> List[FeatureSystem]:
    train = load(name, ["train"])
    dev = load(name, ["dev"])
    trf, dvf = feats_for(train), feats_for(dev)
    systems = build_feature_systems(trf, [it.gold_score for it in train], dvf, [it.label_ord for it in dev])
    write_json(sut_path(name), {
        "dataset": name, "commit": git_commit(),
        "fit": {"weights": "NNLS on train gold_score", "thresholds": "max QWK on dev"},
        "n_train": len(train), "n_dev": len(dev),
        "systems": [s.to_dict() for s in systems],
    })
    return systems


def clean_table(name: str, splits: Optional[List[str]] = None) -> pd.DataFrame:
    splits = splits or (["dev"] + REPORT_SPLITS[name] + (["test_ud"] if name == "scientsbank" else []))
    rows = []
    for split in splits:
        items = load(name, [split])
        if not items:
            continue
        preds = predict_all(name, items)
        gold = [it.label3 for it in items]
        groups = [it.question_id for it in items]
        for sysname, p in preds.items():
            mask = np.array(p["covered"])
            if mask.sum() == 0:
                continue
            g = [x for x, m in zip(gold, mask) if m]
            pl = [x for x, m in zip(p["label"], mask) if m]
            grp = [x for x, m in zip(groups, mask) if m]
            a = agreement(g, pl)
            go = np.array([LABEL_TO_ORD[x] for x in g])
            po = np.array([LABEL_TO_ORD[x] for x in pl])
            lo, hi = bootstrap_ci(grp, lambda idx: qwk(go[idx], po[idx]), n_boot=500)
            rows.append({"dataset": name, "split": split, "system": sysname, **a,
                         "qwk_lo": lo, "qwk_hi": hi, "coverage": float(mask.mean())})
    return pd.DataFrame(rows)


def fit_and_report_clean(datasets: List[str]):
    allt = []
    for name in datasets:
        systems = fit_systems(name)
        print(f"[{name}] hybrid weights={systems[0].weights} thr={systems[0].thresholds}")
        t = clean_table(name)
        allt.append(t)
        print(t[["split", "system", "n", "qwk", "qwk_lo", "qwk_hi", "accuracy", "macro_f1"]]
              .round(3).to_string(index=False))
    df = pd.concat(allt, ignore_index=True)
    df.to_csv(RESULTS_DIR / "clean_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Attack evaluation
# ---------------------------------------------------------------------------
PAIRED = {"A2", "A3", "A4"}  # adversarial text derived from a real (correct) answer


def _llm_criterion_shift(name, atk, adv, idx, srcs, incorrect, allclean, ext) -> List[Dict]:
    table = ext.get("llm_analytic")
    if not table:
        return []

    def crit(row_or_item, k):
        hit = table.get(_key(row_or_item))
        if not hit:
            return np.nan
        return (hit.get("per_criterion") or {}).get(k, np.nan)

    out = []
    for k in FEATS:
        a = np.array([crit(adv[i], k) for i in idx], float)
        if atk in PAIRED:
            r = np.array([crit(srcs[i], k) for i in idx], float)
            m = ~np.isnan(a) & ~np.isnan(r)
            a_m, r_m = a[m], r[m]
            p = paired_delta(r_m, a_m)["wilcoxon_p"]
            ref_name = "source answer"
        else:
            r = np.array([crit(it, k) for it in incorrect], float)
            a_m, r_m = a[~np.isnan(a)], r[~np.isnan(r)]
            p = float(mannwhitneyu(a_m, r_m).pvalue) if len(a_m) and len(r_m) else np.nan
            ref_name = "genuine incorrect"
        allv = np.array([crit(it, k) for it in allclean], float)
        sdv = float(np.nanstd(allv)) or 1.0
        if len(a_m) == 0 or len(r_m) == 0:
            continue
        out.append({"dataset": name, "attack": atk, "criterion": k, "n": len(a_m), "system": "llm_analytic",
                    "adv_mean": float(a_m.mean()), "ref_mean": float(r_m.mean()), "reference": ref_name,
                    "delta": float(a_m.mean() - r_m.mean()),
                    "delta_sd": float((a_m.mean() - r_m.mean()) / sdv), "p": p})
    return out


def _auto_ok_filter(adv: List[Dict]):
    """Drop attack rows that failed the automatic sanity checks (validation.auto), if run."""
    from .validation import auto_ok_ids
    ok = auto_ok_ids()
    if ok is None:
        return adv, 0
    kept = [r for r in adv if r["id"] in ok]
    return kept, len(adv) - len(kept)


def _clean_baseline_rows(name, clean_items, systems, ext):
    """Reference rates on genuine answers: how often a real incorrect answer is graded above
    'incorrect' (the floor any ASR must be compared with), and how often a real correct answer is
    graded 'correct'."""
    rows = []
    preds = predict_all(name, clean_items, systems, ext)
    gold = np.array([it.label3 for it in clean_items])
    groups = np.array([it.question_id for it in clean_items])
    for sysname, p in preds.items():
        cov = np.array(p["covered"])
        lab = np.array([x if x is not None else "incorrect" for x in p["label"]])
        for g, atk in (("incorrect", "CLEAN_incorrect"), ("correct", "CLEAN_correct")):
            m = (gold == g) & cov
            if m.sum() == 0:
                continue
            if g == "incorrect":
                succ = np.array([LABEL_TO_ORD[x] > 0 for x in lab[m]], float)
            else:
                succ = np.array([x == "correct" for x in lab[m]], float)
            lo, hi = bootstrap_ci(groups[m], lambda ix: succ[ix].mean(), n_boot=500)
            key = "asr" if g == "incorrect" else "kept_correct"
            rows.append({"dataset": name, "attack": atk, "family": "clean", "system": sysname,
                         "n": int(m.sum()), "coverage": float(m.sum() / (gold == g).sum()),
                         key: float(succ.mean()), "asr_lo": lo, "asr_hi": hi,
                         "asr_to_correct": float(np.mean(lab[m] == "correct"))})
    return rows


def evaluate_attacks(datasets: List[str]):
    summary, crit_rows = [], []
    for name in datasets:
        adv = load_attacks(name, REPORT_SPLITS[name])
        adv, n_drop = _auto_ok_filter(adv)
        if n_drop:
            print(f"[{name}] dropped {n_drop} attack rows failing automatic checks")
        if not adv:
            continue
        clean = {it.id: it for it in load(name, REPORT_SPLITS[name])}
        systems = load_feature_systems(name)
        ext = external_predictions(name)
        summary += _clean_baseline_rows(name, list(clean.values()), systems, ext)
        pa = predict_all(name, adv, systems, ext)
        srcs = [clean[r["orig_id"]] for r in adv]
        ps = predict_all(name, srcs, systems, ext)
        fa, fs = feats_for(adv), feats_for(srcs)
        # Genuine incorrect answers per split: reference group for unpaired attacks.
        incorrect = [it for it in clean.values() if it.label3 == "incorrect"]
        fi = feats_for(incorrect)
        allclean = list(clean.values())
        fc = feats_for(allclean)
        sd = {k: float(np.std([f[k] for f in fc]) or 1.0) for k in FEATS}

        attacks = sorted({r["attack"] for r in adv})
        for atk in attacks:
            idx = [i for i, r in enumerate(adv) if r["attack"] == atk]
            exp = [adv[i]["expected_label"] for i in idx]
            groups = [adv[i]["question_id"] for i in idx]
            for sysname in pa:
                cov = [i for i in idx if pa[sysname]["covered"][i] and ps[sysname]["covered"][i]]
                if not cov:
                    continue
                lab = [pa[sysname]["label"][i] for i in cov]
                ex = [adv[i]["expected_label"] for i in cov]
                row = {"dataset": name, "attack": atk, "family": FAMILY[atk], "system": sysname,
                       "n": len(cov), "coverage": len(cov) / len(idx)}
                if atk == "A4":
                    row.update(invariance_failure([ps[sysname]["label"][i] for i in cov], lab))
                    e_ok = np.array([lab[j] == "correct" for j in range(len(cov))], float)
                    row["kept_correct"] = float(e_ok.mean())
                else:
                    row.update(attack_success(ex, lab))
                    succ = np.array([LABEL_TO_ORD[l] > LABEL_TO_ORD[e] for l, e in zip(lab, ex)], float)
                    grp = [adv[i]["question_id"] for i in cov]
                    row["asr_lo"], row["asr_hi"] = bootstrap_ci(grp, lambda ix: succ[ix].mean(), n_boot=500)
                if atk in PAIRED:
                    row.update(paired_delta([ps[sysname]["score"][i] for i in cov],
                                            [pa[sysname]["score"][i] for i in cov]))
                    # Source answers were correct: how many did the system grade correct to begin with?
                    row["src_pred_correct"] = float(np.mean([ps[sysname]["label"][i] == "correct" for i in cov]))
                summary.append(row)

            # Per-criterion shift (feature systems share one feature vector).
            for k in FEATS:
                adv_v = np.array([fa[i][k] for i in idx])
                if atk in PAIRED:
                    ref_v = np.array([fs[i][k] for i in idx])
                    pd_ = paired_delta(ref_v, adv_v)
                    p = pd_["wilcoxon_p"]
                    ref_name = "source answer"
                else:
                    ref_v = np.array([f[k] for f in fi])
                    p = float(mannwhitneyu(adv_v, ref_v).pvalue) if len(ref_v) and len(adv_v) else np.nan
                    ref_name = "genuine incorrect"
                crit_rows.append({
                    "dataset": name, "attack": atk, "criterion": k, "n": len(idx), "system": "hybrid",
                    "adv_mean": float(adv_v.mean()), "ref_mean": float(ref_v.mean()), "reference": ref_name,
                    "delta": float(adv_v.mean() - ref_v.mean()),
                    "delta_sd": float((adv_v.mean() - ref_v.mean()) / sd[k]), "p": p,
                })
            # Same analysis for the analytic LLM grader's self-reported criteria.
            crit_rows += _llm_criterion_shift(name, atk, adv, idx, srcs, incorrect, allclean, ext)
    s = pd.DataFrame(summary)
    c = pd.DataFrame(crit_rows)
    if len(c):
        m = len(c)
        c["p_bonf"] = np.minimum(1.0, c["p"] * m)
    s.to_csv(RESULTS_DIR / "attack_results.csv", index=False)
    c.to_csv(RESULTS_DIR / "criterion_shift.csv", index=False)
    print(s.round(3).to_string(index=False))
    return s, c
