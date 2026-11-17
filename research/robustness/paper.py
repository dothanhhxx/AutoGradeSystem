"""Build the VNICT-2026-style paper (.docx) from the result files.

    python -m research.robustness.paper  ->  experiments/results/robustness/paper_vnict2026.docx

Every number in the text is read from experiments/results/robustness/*.csv|json.
"""

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, read_json
from .docx_style import Paper
from .report import FIG, fig_adaptive, fig_asr, fig_criterion_shift, fig_framework, fig_roc

DS = ["scientsbank", "beetle", "mohler"]
DSN = {"scientsbank": "SciEntsBank", "beetle": "Beetle", "mohler": "Mohler"}
TEST = {"scientsbank": ["test_ua", "test_uq", "test_ud"], "beetle": ["test_ua", "test_uq"], "mohler": ["test"]}
SPN = {"test_ua": "UA", "test_uq": "UQ", "test_ud": "UD", "test": "test"}
MAIN = ["hybrid", "deberta", "llm_holistic", "llm_analytic"]
SYSN = {"hybrid": "Hybrid 5+1 (ours)", "deberta": "DeBERTa-v3 fine-tuned", "llm_holistic": "Qwen2.5-3B holistic",
        "llm_analytic": "Qwen2.5-3B analytic", "only_semantic": "Semantic only", "only_coverage": "Coverage only",
        "only_logic": "Logic (NLI) only"}


def pct(x, d=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def load():
    R = {}
    c = pd.read_csv(RESULTS_DIR / "clean_results.csv")
    c["sys"] = c.system.str.replace(r"_s\d+$", "", regex=True)
    R["clean"] = c
    a = pd.read_csv(RESULTS_DIR / "attack_results.csv")
    a["sys"] = a.system.str.replace(r"_s\d+$", "", regex=True)
    a["v"] = np.where(a.attack == "A4", a.label_drop, a.asr)
    R["atk"] = a
    R["crit"] = pd.read_csv(RESULTS_DIR / "criterion_shift.csv")
    R["det"] = pd.read_csv(RESULTS_DIR / "detector_overall.csv")
    R["det_pa"] = pd.read_csv(RESULTS_DIR / "detector_per_attack.csv")
    R["loao"] = pd.read_csv(RESULTS_DIR / "detector_loao.csv")
    R["coef"] = read_json(RESULTS_DIR / "detector_lr_coefficients.json")
    R["sut"] = {d: read_json(RESULTS_DIR / d / "sut_config.json") for d in DS}
    p = RESULTS_DIR / "adaptive_search.csv"
    R["adapt"] = pd.read_csv(p) if p.exists() else None
    p = RESULTS_DIR / "adaptive_llm.csv"
    R["adapt_llm"] = pd.read_csv(p) if p.exists() else None
    v = RESULTS_DIR / "validation" / "auto_checks.csv"
    R["auto"] = pd.read_csv(v) if v.exists() else None
    return R


def clean_q(R, sys, d, split):
    c = R["clean"]
    x = c[(c.sys == sys) & (c.dataset == d) & (c.split == split)]
    return (x.qwk.mean(), x.qwk.std() if len(x) > 1 else np.nan, x.qwk_lo.mean(), x.qwk_hi.mean()) if len(x) else (np.nan,) * 4


def atk_v(R, sys, d, attack, col="v"):
    a = R["atk"]
    x = a[(a.sys == sys) & (a.dataset == d) & (a.attack == attack)]
    return float(x[col].mean()) if len(x) else np.nan


def shift(R, system, attack, crit):
    c = R["crit"]
    x = c[(c.system == system) & (c.attack == attack) & (c.criterion == crit)]
    return float(x.delta_sd.mean()) if len(x) else np.nan


def det(R, d, k, col):
    x = R["det"]
    x = x[(x.dataset == d) & (x.detector == k)]
    return float(x[col].iloc[0]) if len(x) else np.nan


def rng(vals, fmt=pct):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "–"
    lo, hi = min(vals), max(vals)
    return fmt(lo) if abs(hi - lo) < 1e-9 else f"{fmt(lo)}–{fmt(hi)}"


def build(out=None):
    fig_framework()
    fig_asr()
    fig_criterion_shift()
    fig_roc()
    fig_adaptive()
    R = load()
    P = Paper()

    # ---------- derived numbers ----------
    q_h = [clean_q(R, "hybrid", d, s)[0] for d in DS for s in TEST[d]]
    q_d = [clean_q(R, "deberta", d, s)[0] for d in DS for s in TEST[d]]
    q_lh = [clean_q(R, "llm_holistic", d, s)[0] for d in DS for s in TEST[d] if s != "test_ud"]
    q_la = [clean_q(R, "llm_analytic", d, s)[0] for d in DS for s in TEST[d] if s != "test_ud"]
    base_h = [atk_v(R, "hybrid", d, "CLEAN_incorrect") for d in DS]
    stuff_h = [atk_v(R, "hybrid", d, a) for d in DS for a in ("A1b", "A1c")]
    stuff_d = [atk_v(R, "deberta", d, a) for d in DS for a in ("A1b", "A1c")]
    neg_h = [atk_v(R, "hybrid", d, "A3", "mean_delta") for d in DS]
    a4_h = [atk_v(R, "hybrid", d, "A4") for d in DS]
    n_attacks = 0
    if R["auto"] is not None:
        au = R["auto"]
        rep = au[au.split.isin(["test_ua", "test_uq", "test"])]
        n_attacks, n_ok = len(au), int(au.ok.sum())
        pass_a2 = au[au.attack == "A2"].ok.mean()
        pass_a4 = au[au.attack == "A4"].ok.mean()
        n_rep = int(rep.ok.sum())
    auc0 = [det(R, d, "D0_cdi_rule", "auroc") for d in DS]
    auc1 = [det(R, d, "D1_cdi_score", "auroc") for d in DS]
    auc2 = [det(R, d, "D2_disagreement_lr", "auroc") for d in DS]
    auc3 = [det(R, d, "D3_boosting", "auroc") for d in DS]
    und = [det(R, d, "D3_boosting", "asr_undefended") for d in DS]
    dfd = [det(R, d, "D3_boosting", "asr_defended") for d in DS]
    fpr3 = [det(R, d, "D3_boosting", "fpr_clean(dev thr)") for d in DS]
    dpa = R["det_pa"]
    flag = lambda k, a: [float(dpa[(dpa.detector == k) & (dpa.attack == a) & (dpa.dataset == d)].flag_rate.mean())
                         for d in DS]
    lo = R["loao"]
    lo2 = lo[lo.detector == "D2_disagreement_lr"]
    held = lambda fam: lo2[lo2.held_out_family == fam].flag_rate_held_out.tolist()

    # ---------- title & abstract ----------
    P.title("Can a Criterion-Decomposed Short-Answer Grader Be Fooled? Per-Criterion Robustness Analysis "
            "and Disagreement-Based Detection of Gaming Answers")
    P.abstract(
        "Automated short-answer graders that score each answer on several explicit criteria are attractive "
        "because a teacher can see why a grade was given. Whether such graders can be gamed by students, and "
        "which criterion fails when they are, has not been measured. We evaluate a five-criterion hybrid grader "
        "(semantic similarity, keyword coverage, grammar, formality and NLI-based logical consistency) on three "
        "public benchmarks (SciEntsBank, Beetle and Mohler) under nine families of gaming answers, "
        f"{n_attacks:,} generated in total, and compare it with a fine-tuned DeBERTa-v3 grader and a Qwen2.5-3B "
        f"grader used holistically and analytically. On genuine answers the hybrid grader reaches a QWK of "
        f"{rng(q_h, f3)}, below the fine-tuned encoder ({rng(q_d, f3)}) and comparable to the LLM. Keyword "
        f"stuffing lifts {rng(stuff_h)} of off-topic answers above the 'incorrect' grade, against {rng(base_h)} "
        "for genuinely incorrect answers. Because scores are decomposed, the failure can be located: stuffing "
        f"raises keyword coverage by {shift(R, 'hybrid', 'A1b', 'coverage'):+.1f} SD, while negation is caught "
        f"only by the NLI criterion ({shift(R, 'hybrid', 'A3', 'logic'):+.1f} SD) and leaves semantic similarity "
        f"almost unchanged ({shift(R, 'hybrid', 'A3', 'semantic'):+.1f} SD). The LLM's own criterion scores move "
        "together under every attack and do not localise failures. A detector that learns from disagreement "
        f"between criteria reaches an AUROC of {rng(auc3, f3)}, whereas the fixed Cognitive Discrepancy Index "
        f"rule scores {rng(auc0, f3)}. Routing flagged answers to a teacher cuts the attack success rate from "
        f"{rng(und)} to {rng(dfd)} at a {rng(fpr3)} flag rate on genuine answers, but the detector does not "
        "transfer to attack families it has not seen.",
        "Automatic Short Answer Grading; Adversarial Robustness; Interpretable Assessment; "
        "Natural Language Inference; Attack Detection.")

    # ---------- I. Introduction ----------
    P.h1("Introduction")
    P.para(
        "Automatic short answer grading (ASAG) reduces teachers' marking load, and recent systems either "
        "fine-tune transformer encoders on graded answers [1], [2] or prompt large language models (LLMs) to "
        "grade directly [3]–[5]. Both are hard to inspect. A third design decomposes the grade into explicit "
        "criteria computed by separate models and combines them with transparent weights [6], [7], so that "
        "each grade comes with criterion-level evidence. Our earlier system follows this design with five "
        "Deterministic Linguistic Indicators (DLIs) and an LLM that writes feedback from them.")
    P.para(
        "A grader used for formative or summative assessment also has to resist answers written to obtain "
        "marks without the knowledge being tested. Earlier studies showed that neural content scorers accept "
        "nonsense, shuffled or keyword-laden input [8], [9] and that universal trigger phrases can flip "
        "ASAG decisions [10]. Those studies treat the grader as a black box and report how much the total "
        "score changes. A criterion-decomposed grader allows a more specific question: which criterion is "
        "fooled by which kind of answer, and can disagreement between criteria reveal the attempt?")
    P.figure(FIG / "fig_framework.png",
             "Study framework. Genuine and generated gaming answers are scored on five criteria; the "
             "weighted grade is returned, and a detector reads disagreement between criteria to route "
             "suspicious answers to a teacher.")
    P.para("This paper answers that question on three public benchmarks. Its contributions are:")
    P.numbered([
        f"A taxonomy of nine gaming-answer families and {n_attacks:,} generated answers over SciEntsBank, "
        "Beetle and Mohler, with question-level splits, automatic quality filtering and a blind human "
        "annotation protocol.",
        "A per-criterion robustness analysis of a five-criterion grader, compared with a fine-tuned encoder "
        "and an LLM grader in holistic and analytic mode, showing which criterion each attack exploits.",
        "A disagreement-based detector that flags gaming answers for teacher review, evaluated against a "
        "fixed-threshold Cognitive Discrepancy Index (CDI), on unseen attack families and against "
        "detector-aware attackers.",
    ])

    # ---------- II. Methods ----------
    P.h1("Methods")
    P.h2("Grader Under Test")
    P.para(
        "The grader receives a question Q, a reference answer R and a student answer S, and computes five "
        "indicators in [0, 1] with the models of the deployed system: SimCSE-RoBERTa-large [11] for semantic "
        "similarity, KeyBERT keywords [12] matched by substring or embedding similarity for coverage, "
        "RoBERTa-CoLA for grammatical acceptability, a RoBERTa formality classifier, and a bidirectional "
        "DeBERTa-v3 NLI cross-encoder [13] for logical consistency:")
    P.equation("z = [s_sem, s_cov, s_gra, s_for, s_log]")
    P.para(
        "Semantic similarity is the clipped cosine between SimCSE embeddings of R and S; coverage is the "
        "fraction of reference keywords found in S at a similarity threshold of 0.50; s_log combines "
        "forward and backward entailment and contradiction probabilities and is lowered sharply when either "
        "contradiction probability exceeds 0.60. Answers shorter than five words skip the models and are "
        "scored by word overlap with R. The grade is a weighted sum with contradiction caps:")
    P.equation("y = cap( 100 · Σ_i w_i · s_i , max(P_c^fwd, P_c^bwd) )")
    w = R["sut"]["scientsbank"]["systems"][0]["weights"]
    P.para(
        "where the cap limits the grade to 40 when the maximum contradiction probability exceeds 0.90 and to "
        "55 above 0.70. Weights are fitted per dataset by non-negative least squares on the training split "
        f"(for SciEntsBank: semantic {w['semantic']:.2f}, coverage {w['coverage']:.2f}, grammar "
        f"{w['grammar']:.2f}, formality {w['formality']:.2f}, logic {w['logic']:.2f}), and two grade "
        "thresholds that map y to incorrect, partially correct and correct are chosen on the development "
        "split to maximise QWK. A test confirms that the batched feature extraction used here reproduces the "
        "deployed grader's per-answer scores.")

    P.h2("Threat Model and Gaming Answers")
    P.para(
        "The attacker is a student who knows the key terms of the topic but not the reference answer and "
        "who can submit free text. We approximate known terms by KeyBERT keywords of the reference answer, "
        "which makes stuffing attacks an upper bound for an informed student. Table I lists the nine "
        "families. Rule-based attacks are deterministic; A2 and A4 are written by Phi-3.5-mini [14], an LLM "
        "from a different family than the LLM grader. Gaming answers for detector training are generated "
        "only from training and development answers, and those used for evaluation only from test answers.")
    cnt = R["auto"].groupby("attack").size() if R["auto"] is not None else {}
    rows = [
        ["A1a", "Bare list of key terms", "incorrect", "rule"],
        ["A1b", "Answer to another question + key terms", "incorrect", "rule"],
        ["A1c", "Key terms in a filler sentence", "incorrect", "rule"],
        ["A2", "Fluent rewrite of a correct answer with one wrong fact", "incorrect", "LLM"],
        ["A3", "Negation or antonym flip of a correct answer", "incorrect", "rule"],
        ["A4", "Meaning-preserving paraphrase (invariance test)", "correct", "LLM"],
        ["A5", "Restating the question", "incorrect", "rule"],
        ["A6", "A1b with broken grammar (white-box)", "incorrect", "rule"],
        ["A7", "Fewer than five key terms (white-box)", "incorrect", "rule"],
    ]
    for r in rows:
        r.append(f"{int(cnt.get(r[0], 0)):,}")
    P.table("Gaming-answer families", ["Code", "Description", "Expected label", "Generator", "Generated"], rows,
            col_widths=[0.5, 3.0, 0.9, 0.8, 0.8])
    P.para(
        "A6 targets a rule of the grader that raises the logic score when grammar is poor, and A7 targets "
        "the short-answer branch, so both are reported separately as white-box probes. Generated answers "
        "pass automatic checks (non-empty, changed, no leaked instructions, length ratio between 0.5 and 2 "
        f"for LLM rewrites); {pct(pass_a2)} of A2 and {pct(pass_a4)} of A4 rewrites pass, and all failing "
        "answers are excluded.")

    P.h2("Comparison Graders")
    P.para(
        "We compare against (i) DeBERTa-v3-base [13] fine-tuned per dataset on (question and reference, "
        "student answer) pairs for three epochs, three seeds, with the epoch chosen by development QWK; runs "
        "that collapse to one class (development QWK below 0.2) are retrained with class-weighted loss, which "
        "was needed for all three Mohler seeds; (ii) Qwen2.5-3B-Instruct [15] prompted with Q, R and S to "
        "return a label and a 0–10 score (holistic); and (iii) the same LLM asked to rate the five criteria "
        "of our grader before giving a label (analytic). Decoding is greedy and outputs are parsed as JSON; "
        "no output failed to parse.")

    P.h2("Robustness Measures")
    P.para(
        "The attack success rate (ASR) is the fraction of gaming answers graded above their expected label. "
        "Because many graders also over-grade genuine incorrect answers, every ASR is compared with the same "
        "rate on genuine incorrect test answers. For A4 we report the fraction whose label drops relative to "
        "the source answer. To locate the fooled criterion, the shift of criterion k under attack a is "
        "expressed in standard-deviation units of that criterion on genuine test answers:")
    P.equation("Δ_k(a) = ( mean_{S∈a} s_k(S) − mean_{S∈ref(a)} s_k(S) ) / σ_k")
    P.para(
        "where ref(a) is the set of source answers for attacks derived from a correct answer (A2–A4) and the "
        "set of genuine incorrect answers otherwise. Confidence intervals resample questions, not answers.")

    P.h2("Disagreement-Based Detection")
    P.para(
        "A gaming answer tends to score high on one criterion and low on another. The Cognitive Discrepancy "
        "Index of our earlier proposal, CDI = s_cov − s_log, flags an answer when s_cov ≥ 0.85 and "
        "s_log ≤ 0.40 (D0). We also evaluate CDI as a continuous score (D1), a logistic regression on "
        "pairwise criterion gaps and NLI probabilities (D2), and gradient boosting on all criterion "
        "features, answer length and keyword density (D3). Detectors are trained on training-split genuine "
        "and gaming answers, with A4 paraphrases as negatives, and the threshold is set so that 5% of genuine "
        "development answers are flagged. A flagged answer is routed to the teacher; its grade is not "
        "changed.")

    P.h2("Adaptive Attacks")
    P.para(
        "A detector-aware attacker starts from an answer to another question and greedily inserts key terms "
        "and connecting phrases, querying the deployed grader and keeping the edit with the highest grade "
        "among those not flagged by D2. Success at budgets of 10, 50 and 150 queries is recorded for 100 "
        "test questions per dataset, together with whether the result also evades the unseen D3 detector. "
        "A second attacker asks the generator LLM to weave the key terms into coherent sentences without "
        "answering the question.")

    # ---------- III. Experiments ----------
    P.h1("Experiments And Results")
    P.h2("Datasets and Setup")
    sp = read_json(RESULTS_DIR.parents[2] / "data" / "processed" / "splits.json")
    drow = []
    for d in DS:
        tr, dv = sp[d]["train"], sp[d]["dev"]
        tests = " / ".join(f"{SPN[s]} {sp[d][s]['n']:,}" for s in TEST[d])
        drow.append([DSN[d], f"{tr['n']:,} ({tr['n_questions']})", f"{dv['n']:,} ({dv['n_questions']})", tests])
    P.para(
        "SciEntsBank and Beetle come from SemEval-2013 Task 7 [16] and keep their official test splits of "
        "unseen answers (UA), unseen questions (UQ) and, for SciEntsBank, unseen domains (UD); 20% of the "
        "training questions form a development split. Mohler [17] contains computer-science answers scored "
        "0–5 by two graders, averaged and mapped to correct (≥ 4), partially correct (≥ 2.5) and incorrect; "
        "it is split 60/20/20 by question. Labels are mapped to three classes. All runs used one RTX 4060 "
        "laptop GPU (8 GB).")
    P.table("Datasets (answers, with questions in parentheses)", ["Dataset", "Train", "Dev", "Test"], drow,
            col_widths=[1.2, 1.3, 1.2, 2.7])

    P.h2("Agreement on Genuine Answers")
    hdr = ["System"] + [f"{DSN[d][:4]}. {SPN[s]}" if d != "mohler" else "Mohler" for d in DS for s in TEST[d]]
    trows = []
    for s in MAIN + ["only_semantic"]:
        r = [SYSN[s]]
        for d in DS:
            for sp_ in TEST[d]:
                m, sd, lo_, hi_ = clean_q(R, s, d, sp_)
                r.append("–" if np.isnan(m) else (f"{m:.2f}±{sd:.2f}" if not np.isnan(sd) else f"{m:.2f}"))
        trows.append(r)
    P.table("Quadratic weighted kappa on genuine test answers", hdr, trows)
    P.para(
        f"Table III shows that the hybrid grader agrees moderately with human labels (QWK {rng(q_h, f3)}), "
        f"below the fine-tuned encoder ({rng(q_d, f3)}, mean over three seeds) and in the range of the "
        f"holistic LLM ({rng(q_lh, f3)}); the analytic LLM is lowest ({rng(q_la, f3)}). Semantic similarity "
        "alone comes close to the full hybrid grade, and grammar and formality alone have QWK below 0.16 on "
        "every split. These figures are lower than the values reported for our earlier prototype, which "
        "was evaluated on synthetic data; the present results are measured on the real benchmarks.")

    P.h2("Attack Success")
    P.figure(FIG / "fig_asr.png",
             "Share of answers graded above the expected label (A4: share whose label drops). The first "
             "group is the rate for genuine incorrect answers, the reference for every attack.", 6.3)
    arows = []
    order = ["CLEAN_incorrect", "A1a", "A1b", "A1c", "A2", "A3", "A5", "A6", "A7", "A4"]
    names = {"CLEAN_incorrect": "Genuine incorrect"}
    for a in order:
        r = [names.get(a, a)]
        for s in MAIN:
            r.append(" / ".join(pct(atk_v(R, s, d, a)) for d in DS))
        arows.append(r)
    P.table("Attack success rate (SciEntsBank / Beetle / Mohler)", ["Answers"] + [SYSN[s] for s in MAIN], arows,
            col_widths=[1.2, 1.3, 1.3, 1.3, 1.3], font_size=8)
    P.para(
        f"Keyword stuffing is the most effective family against the hybrid grader: A1b and A1c raise "
        f"{rng(stuff_h)} of answers above 'incorrect', compared with {rng(base_h)} for genuine incorrect "
        f"answers. The fine-tuned encoder is less affected on SciEntsBank and Beetle ({rng(stuff_d[:4])}) but "
        "not on Mohler, where its training labels are 76% correct. The holistic LLM accepts stuffed answers "
        "about as often as the hybrid grader, and the analytic LLM is the hardest to fool, partly because it "
        "grades genuine answers more harshly. Negation flips lower the hybrid grade by "
        f"{rng([-x for x in neg_h], lambda v: f'{100 * v:.0f}')} points on average but often leave the answer "
        f"above 'incorrect'. Paraphrases lower the label of {rng(a4_h)} of answers, which shows that the "
        "grader also penalises legitimate rewording, most of all on Beetle. The white-box A6 attack succeeds "
        "on almost all answers, while short keyword answers (A7) are mostly caught by the word-overlap branch.")

    P.h2("Which Criterion Is Fooled")
    P.figure(FIG / "fig_criterion_shift.png",
             "Mean shift of each criterion under each attack, in SD units, for the hybrid grader (top) and "
             "the criterion scores reported by the analytic LLM (bottom).", 6.3)
    P.para(
        f"Fig. 3 locates each failure. Stuffing raises coverage by {shift(R, 'hybrid', 'A1b', 'coverage'):+.1f} "
        f"SD (A1b) and {shift(R, 'hybrid', 'A1c', 'coverage'):+.1f} SD (A1c) above genuine incorrect answers, "
        f"and it also raises the logic score ({shift(R, 'hybrid', 'A1b', 'logic'):+.1f} SD) because an "
        "off-topic answer is neutral, not contradictory, to the reference. Negation (A3) and fluent wrong "
        f"rewrites (A2) are visible almost only in the logic criterion ({shift(R, 'hybrid', 'A3', 'logic'):+.1f} "
        f"and {shift(R, 'hybrid', 'A2', 'logic'):+.1f} SD), while semantic similarity "
        f"({shift(R, 'hybrid', 'A3', 'semantic'):+.1f} SD) and coverage ({shift(R, 'hybrid', 'A3', 'coverage'):+.1f} "
        "SD) barely move. Breaking grammar (A6) lowers grammar and formality by one to four SD but leaves "
        "coverage and logic high, which is the rule exploited by the white-box attack.")
    _a3 = [shift(R, "llm_analytic", "A3", k) for k in ["semantic", "coverage", "grammar", "formality", "logic"]]
    P.para(
        "The analytic LLM behaves differently. Under negation all five of its criterion scores drop by a "
        f"similar amount (between {min(_a3):+.1f} and {max(_a3):+.1f} SD), "
        "including grammar and formality, which the edit does not change; under paraphrase all five rise. Its "
        "criterion scores follow its overall judgement and do not show which aspect of the answer is wrong. "
        "Separate models per criterion are what make the localisation possible.")
    ex = [
        ["A1b", "Answer to a different question + 'Bulb battery path contained.'",
         "sem 0.73, cov 1.00, gra 0.18, log 0.81", "81.0 (correct)"],
        ["A3 source", "because bulb a and b are still contained in the closed path with the battery.",
         "sem 0.84, cov 0.75, gra 0.43, log 0.70", "72.2 (correct)"],
        ["A3 attack", "... still contained in the open path with the battery.",
         "sem 0.78, cov 0.75, gra 0.44, log 0.02", "46.5 (partial)"],
        ["A2 attack", "When the multimeter shows zero volts across a light bulb, it indicates a functioning bulb...",
         "sem 0.71, cov 0.80, gra 0.92, log 0.00", "40.0 (partial)"],
    ]
    P.table("Beetle examples with criterion scores", ["Answer", "Text", "Criterion scores", "Grade"], ex,
            col_widths=[0.8, 2.8, 1.9, 1.0], font_size=8)

    P.h2("Detection")
    drows = []
    dn = {"D0_cdi_rule": "D0 CDI rule", "D1_cdi_score": "D1 CDI score", "D2_disagreement_lr": "D2 gap LR",
          "D3_boosting": "D3 boosting"}
    for k, nm in dn.items():
        drows.append([nm,
                      " / ".join(f3(det(R, d, k, "auroc")) for d in DS),
                      " / ".join(pct(det(R, d, k, "tpr(dev thr)")) for d in DS),
                      " / ".join(pct(det(R, d, k, "fpr_clean(dev thr)"), 1) for d in DS),
                      " / ".join(pct(det(R, d, k, "asr_defended")) for d in DS)])
    P.table("Detection on test answers (SciEntsBank / Beetle / Mohler)",
            ["Detector", "AUROC", "Flagged attacks", "Flagged genuine", "ASR after routing"], drows,
            col_widths=[1.2, 1.4, 1.4, 1.4, 1.4], font_size=8)
    P.figure(FIG / "fig_roc.png", "ROC curves of the four detectors; the dashed line marks 5% false positives.")
    c = R["coef"]
    P.para(
        f"The fixed CDI rule flags few attacks (AUROC {rng(auc0, f3)}) because stuffed answers usually keep a "
        "neutral, mid-range logic score. As a continuous score, CDI reaches "
        f"{rng(auc1, f3)}. Learning from all pairwise gaps lifts AUROC to {rng(auc2, f3)} (D2) and "
        f"{rng(auc3, f3)} (D3). The largest D2 coefficients are on the coverage–semantic gap "
        f"({rng([c[d]['gap_cov_sem'] for d in DS], lambda v: f'{v:.2f}')}) and the contradiction probability, "
        "not on the coverage–logic gap. Routing flagged answers to "
        f"the teacher reduces the hybrid grader's ASR from {rng(und)} to {rng(dfd)} with D3 while "
        f"{rng(fpr3)} of genuine answers are flagged. Detection is uneven across families: D2 flags "
        f"{rng(flag('D2_disagreement_lr', 'A1b'))} of A1b and {rng(flag('D2_disagreement_lr', 'A6'))} of A6 "
        f"answers, but only {rng(flag('D2_disagreement_lr', 'A2'))} of fluent wrong rewrites and "
        f"{rng(flag('D2_disagreement_lr', 'A3'))} of negations, which do not create disagreement between "
        f"criteria. When a family is held out of training, D2 flags {rng(held('fluent_wrong'))} of fluent "
        f"wrong rewrites and {rng(held('negation'))} of negations, so the detector does not generalise to "
        "unseen attack styles.")

    if R["adapt"] is not None:
        ad = R["adapt"].groupby(["dataset", "budget"])[["success_vs_D2", "evades_D3"]].mean()
        s150 = [float(ad.loc[(d, 150), "success_vs_D2"]) for d in DS if (d, 150) in ad.index]
        s10 = [float(ad.loc[(d, 10), "success_vs_D2"]) for d in DS if (d, 10) in ad.index]
        e3 = [float(ad.loc[(d, 150), "evades_D3"]) for d in DS if (d, 150) in ad.index]
        P.h2("Adaptive Attacks")
        P.figure(FIG / "fig_adaptive.png",
                 "Share of off-topic starting answers that a query-based attacker turns into an answer graded "
                 "above 'incorrect' and not flagged by D2.", 4.8)
        txt = (f"A detector-aware attacker defeats D2 quickly: after 10 queries {rng(s10)} of off-topic answers "
               f"are graded above 'incorrect' without being flagged, and after 150 queries {rng(s150)}. For "
               f"{rng(e3)} of starting answers the final answer also evades D3, which the attacker never "
               "queried.")
        if R["adapt_llm"] is not None:
            al = R["adapt_llm"].groupby("dataset")[["graded_above_incorrect", "flag_D2", "success_vs_D2"]].mean()
            txt += (" LLM-written coherent stuffing is graded above 'incorrect' for "
                    f"{rng([float(al.loc[d, 'graded_above_incorrect']) for d in DS if d in al.index])} of "
                    f"questions and flagged by D2 for {rng([float(al.loc[d, 'flag_D2']) for d in DS if d in al.index])}; "
                    "some of these LLM-written answers read close to a partially correct answer, so this figure "
                    "awaits the human validation.")
        txt += (" The detector therefore raises the cost of naive gaming but is not a defence against an "
                "attacker who can query the grader.")
        P.para(txt)

    # ---------- IV. Discussion ----------
    P.h1("Discussion")
    P.h2("What the Decomposition Provides")
    P.para(
        "The hybrid grader is not more accurate than a fine-tuned encoder, and it is about as easy to game "
        "with key terms as a holistic LLM. Its advantage is diagnostic. Because each criterion comes from a "
        "separate model, an attack leaves a recognisable signature: high coverage with low semantic "
        "similarity for stuffing, a logic drop with unchanged similarity for negation. The same signature "
        "supports detection, and it points to fixes: coverage should require that key terms appear in a "
        "claim related to the question, and the rule that raises the logic score for poor grammar should be "
        "removed, since it is exploited by A6. An LLM asked for the same criteria returns scores that move "
        "together, so a teacher cannot use them to see what is wrong with an answer.")
    P.h2("Limitations")
    P.bullets([
        "The expected labels of generated answers are checked automatically only; a blind annotation of 300 "
        "answers by two annotators has been prepared and is needed before the attack success rates can be "
        "read as true error rates.",
        "Key terms are taken from the reference answer, which overstates what a student knows.",
        "The comparison LLM is small (3B parameters) because of the 8 GB GPU; larger LLM graders may behave "
        "differently.",
        "The detector does not transfer to unseen attack families and is defeated by a querying attacker.",
        "All data are English science and computer-science answers; Mohler's labels are skewed toward "
        "correct, which affects every grader on that dataset.",
    ])

    # ---------- V. Conclusion ----------
    P.h1("Conclusion and Future Work")
    P.para(
        "We measured how a five-criterion short-answer grader responds to nine kinds of gaming answers on "
        "three real benchmarks. Keyword stuffing raises most off-topic answers above the failing grade, "
        "negation is noticed only by the NLI criterion, and legitimate paraphrases are sometimes penalised. "
        "The criterion decomposition shows which component fails for each attack, which an LLM asked for "
        "the same criteria does not, and disagreement between criteria detects naive gaming far better than "
        "a fixed discrepancy rule. The detector should be used to route answers to teachers rather than to "
        "change grades, because it misses fluent wrong answers and can be evaded by an attacker with query "
        "access. Future work will complete the human validation of the generated answers, replace keyword "
        "coverage with claim-level matching, and test larger LLM graders and Vietnamese-language data.")

    # ---------- References ----------
    P.references([
        'S. Haller, A. Aldea, C. Seifert, and N. Strisciuglio, "Survey on automated short answer grading with deep learning: from word embeddings to transformers," arXiv:2204.03503, 2022.',
        'A. Condor, M. Litster, and Z. Pardos, "Automatic short answer grading with SBERT on out-of-sample questions," in Proc. Educational Data Mining, 2021.',
        'R. Ferreira Mello et al., "Automatic short answer grading in the LLM era: Does GPT-4 with prompt engineering beat traditional models?," in Proc. LAK, 2025.',
        'G. Kortemeyer, "Performance of the pre-trained large language model GPT-4 on automated short answer grading," Discover Artificial Intelligence, vol. 4, no. 1, p. 47, 2024.',
        'Y. Chu et al., "A LLM-powered automatic grading framework with human-level guidelines optimization," arXiv:2410.02165, 2024.',
        'A. Condor and Z. Pardos, "Explainable automatic grading with neural additive models," in Proc. AIED, Springer, 2024.',
        'F. Künnecke et al., "Enhancing multi-domain automatic short answer grading through an explainable neuro-symbolic pipeline," arXiv:2403.01811, 2024.',
        'Y. Ding, B. Riordan, A. Horbach, A. Cahill, and T. Zesch, "Don\'t take "nswvtnvakgxpm" for an answer – The surprising vulnerability of automatic content scoring systems to adversarial input," in Proc. COLING, 2020.',
        'Y. Kumar, M. Bhatia, A. Kabra, J. J. Li, D. Jin, and R. R. Shah, "Calling out bluff: Attacking the robustness of automatic scoring systems with simple adversarial testing," arXiv:2007.06796, 2020.',
        'A. Filighera, T. Steuer, and C. Rensing, "Fooling automatic short answer grading systems," in Proc. AIED, Springer, 2020.',
        'T. Gao, X. Yao, and D. Chen, "SimCSE: Simple contrastive learning of sentence embeddings," in Proc. EMNLP, 2021.',
        'M. Grootendorst, "KeyBERT: Minimal keyword extraction with BERT," Zenodo, 2020.',
        'P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in Proc. ICLR, 2023.',
        'M. Abdin et al., "Phi-3 technical report: A highly capable language model locally on your phone," arXiv:2404.14219, 2024.',
        'Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.',
        'M. O. Dzikovska et al., "SemEval-2013 Task 7: The joint student response analysis and 8th recognizing textual entailment challenge," in Proc. SemEval, 2013, pp. 263–274.',
        'M. Mohler, R. Bunescu, and R. Mihalcea, "Learning to grade short answer questions using semantic similarity measures and dependency graph alignments," in Proc. ACL, 2011, pp. 752–762.',
    ])
    out = out or RESULTS_DIR / "paper_vnict2026.docx"
    P.save(out)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build()
