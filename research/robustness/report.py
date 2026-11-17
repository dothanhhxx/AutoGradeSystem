"""Tables (Markdown + LaTeX) and figures for the robustness study -> experiments/results/robustness/."""

from typing import List

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, git_commit, read_json

FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MAIN_SYSTEMS = ["hybrid", "deberta", "llm_holistic", "llm_analytic"]
SYS_LABEL = {"hybrid": "Hybrid 5+1", "deberta": "DeBERTa-v3 FT", "llm_holistic": "Qwen2.5-3B holistic",
             "llm_analytic": "Qwen2.5-3B analytic", "only_semantic": "Semantic only",
             "only_coverage": "Coverage only", "only_logic": "Logic (NLI) only",
             "hybrid_paperw": "Hybrid (paper weights)", "hybrid_uniform": "Hybrid (uniform weights)"}
ATTACK_LABEL = {"CLEAN_incorrect": "Genuine incorrect (baseline)", "A1a": "A1a keyword list",
                "A1b": "A1b off-topic + keywords", "A1c": "A1c keyword filler", "A2": "A2 fluent but wrong",
                "A3": "A3 negation flip", "A5": "A5 restate question", "A6": "A6 broken grammar (WB)",
                "A7": "A7 <5-word keywords (WB)", "A4": "A4 paraphrase (label drop)"}
ATTACK_ORDER = ["CLEAN_incorrect", "A1a", "A1b", "A1c", "A2", "A3", "A5", "A6", "A7", "A4"]
DATASET_LABEL = {"scientsbank": "SciEntsBank", "beetle": "Beetle", "mohler": "Mohler"}
COLORS = {"hybrid": "#2a78d6", "deberta": "#eb6834", "llm_holistic": "#1baf7a", "llm_analytic": "#eda100"}
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _collapse_seeds(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """deberta_s13/s42/s87 -> 'deberta' with mean and sd columns."""
    df = df.copy()
    df["seed_sys"] = df["system"]
    df["system"] = df["system"].str.replace(r"^deberta_s\d+$", "deberta", regex=True)
    keys = [c for c in df.columns
            if c not in value_cols + ["seed_sys"] and not pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(keys, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [a if not b else (a if b == "mean" else f"{a}_sd") for a, b in agg.columns]
    return agg


def _fmt(m, s=None, pct=False):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    f = (lambda x: f"{100 * x:.0f}") if pct else (lambda x: f"{x:.3f}")
    if s is not None and not np.isnan(s) and s > 0:
        return f"{f(m)}±{f(s)}"
    return f(m)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def table_clean() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "clean_results.csv")
    df = df[df.split != "dev"]
    agg = _collapse_seeds(df[["dataset", "split", "system", "qwk", "accuracy", "macro_f1", "qwk_lo", "qwk_hi",
                              "coverage"]], ["qwk", "accuracy", "macro_f1", "qwk_lo", "qwk_hi", "coverage"])
    order = MAIN_SYSTEMS + ["only_semantic", "only_coverage", "only_logic", "hybrid_paperw", "hybrid_uniform"]
    rows = []
    for s in order:
        r = {"System": SYS_LABEL.get(s, s)}
        for _, x in agg[agg.system == s].iterrows():
            col = f"{DATASET_LABEL[x.dataset]} {x.split.replace('test_', '').upper() if x.split != 'test' else 'test'}"
            ci = "" if s == "deberta" else f" [{x.qwk_lo:.2f},{x.qwk_hi:.2f}]"
            r[col] = _fmt(x.qwk, x.get("qwk_sd")) + ci
        if len(r) > 1:
            rows.append(r)
    return pd.DataFrame(rows).fillna("–")


def table_asr(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "attack_results.csv")
    df = df[df.dataset == dataset]
    df["value"] = np.where(df.attack == "A4", df.get("label_drop"), df.get("asr"))
    agg = _collapse_seeds(df[["dataset", "attack", "system", "value", "n"]], ["value", "n"])
    rows = []
    for a in ATTACK_ORDER:
        sub = agg[agg.attack == a]
        if sub.empty:
            continue
        r = {"Attack": ATTACK_LABEL[a], "n": int(sub[sub.system == "hybrid"]["n"].iloc[0])
             if (sub.system == "hybrid").any() else int(sub["n"].max())}
        for s in MAIN_SYSTEMS + ["only_semantic", "only_coverage", "only_logic"]:
            x = sub[sub.system == s]
            r[SYS_LABEL[s]] = _fmt(x["value"].iloc[0], x["value_sd"].iloc[0], pct=True) if len(x) else "–"
        rows.append(r)
    return pd.DataFrame(rows)


def fig_asr():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(RESULTS_DIR / "attack_results.csv")
    df["value"] = np.where(df.attack == "A4", df.get("label_drop"), df.get("asr"))
    agg = _collapse_seeds(df[["dataset", "attack", "system", "value"]], ["value"])
    datasets = [d for d in DATASET_LABEL if d in set(agg.dataset)]
    attacks = [a for a in ATTACK_ORDER if a in set(agg.attack)]
    systems = [s for s in MAIN_SYSTEMS if s in set(agg.system)]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.1 * len(datasets)), sharex=True, facecolor=SURF)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(systems)
    for ax, d in zip(axes, datasets):
        ax.set_facecolor(SURF)
        for j, s in enumerate(systems):
            vals = [agg[(agg.dataset == d) & (agg.attack == a) & (agg.system == s)]["value"].mean() for a in attacks]
            x = np.arange(len(attacks)) + (j - (len(systems) - 1) / 2) * w
            ax.bar(x, np.nan_to_num(vals), width=w - 0.02, color=COLORS[s], label=SYS_LABEL[s],
                   edgecolor=SURF, linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.set_title(DATASET_LABEL[d], loc="left", fontsize=10, color=INK)
        ax.axvline(0.5, color=BASE, linewidth=0.8, linestyle=":")
        ax.tick_params(axis="x", colors=INK2, labelsize=8, length=0)
    axes[-1].set_xticks(np.arange(len(attacks)))
    axes[-1].set_xticklabels([ATTACK_LABEL[a].replace(" (", "\n(").replace(" ", "\n", 1) for a in attacks],
                             fontsize=7.5, color=INK2)
    axes[0].legend(ncol=len(systems), frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.12),
                   labelcolor=INK2)
    fig.text(0.005, 0.5, "Answers graded above 'incorrect' (A4: label drop)", rotation=90, va="center",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(FIG / "fig_asr.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_criterion_shift():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    c = pd.read_csv(RESULTS_DIR / "criterion_shift.csv")
    if "system" not in c:
        c["system"] = "hybrid"
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8302f"])
    crits = ["semantic", "coverage", "grammar", "formality", "logic"]
    systems = [s for s in ["hybrid", "llm_analytic"] if s in set(c.system)]
    datasets = [d for d in DATASET_LABEL if d in set(c.dataset)]
    attacks = [a for a in ATTACK_ORDER if a in set(c.attack)]
    fig, axes = plt.subplots(len(systems), len(datasets), figsize=(3.6 * len(datasets), 3.4 * len(systems)),
                             facecolor=SURF, squeeze=False, gridspec_kw={"hspace": 0.18})
    for i, s in enumerate(systems):
        for j, d in enumerate(datasets):
            ax = axes[i, j]
            M = np.full((len(attacks), len(crits)), np.nan)
            for a_i, a in enumerate(attacks):
                for k_i, k in enumerate(crits):
                    v = c[(c.system == s) & (c.dataset == d) & (c.attack == a) & (c.criterion == k)]["delta_sd"]
                    if len(v):
                        M[a_i, k_i] = v.iloc[0]
            im = ax.imshow(np.clip(M, -2.5, 2.5), cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
            for a_i in range(len(attacks)):
                for k_i in range(len(crits)):
                    if not np.isnan(M[a_i, k_i]):
                        ax.text(k_i, a_i, f"{M[a_i, k_i]:+.1f}", ha="center", va="center", fontsize=7,
                                color="#ffffff" if abs(M[a_i, k_i]) > 1.6 else INK)
            ax.set_xticks(range(len(crits)))
            ax.set_xticklabels(crits if i == len(systems) - 1 else [], fontsize=7.5, color=INK2, rotation=30)
            ax.set_yticks(range(len(attacks)))
            ax.set_yticklabels(attacks if j == 0 else [], fontsize=7.5, color=INK2)
            ax.set_title(f"{SYS_LABEL[s]} · {DATASET_LABEL[d]}", fontsize=8.5, color=INK, loc="left")
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Shift vs reference answers (SD units)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.savefig(FIG / "fig_criterion_shift.png", dpi=200, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def fig_adaptive():
    p = RESULTS_DIR / "adaptive_search.csv"
    if not p.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(p)
    g = df.groupby(["dataset", "budget"])["success_vs_D2"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for c_, d in zip(cols, [d for d in DATASET_LABEL if d in set(g.dataset)]):
        s = g[g.dataset == d]
        ax.plot(s.budget, s.success_vs_D2, color=c_, linewidth=2, marker="o", markersize=5)
        ax.text(s.budget.iloc[-1] * 1.05, s.success_vs_D2.iloc[-1], DATASET_LABEL[d], fontsize=8, color=INK2,
                va="center")
    ax.set_xscale("log")
    ax.set_xticks([10, 50, 150])
    ax.set_xticklabels(["10", "50", "150"], fontsize=8, color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=8, color=MUTED)
    ax.set_xlabel("Query budget", fontsize=8, color=INK2)
    ax.set_title("Adaptive attack: graded above 'incorrect' and not flagged (D2)", fontsize=8.5,
                 color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.set_xlim(8, 260)
    fig.tight_layout()
    fig.savefig(FIG / "fig_adaptive.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_framework():
    """Fig. 1: study framework (attacks -> five criteria -> grade; criteria gaps -> detector -> teacher)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#ffffff")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#f0efec", ec="#c3c2b7", bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        two = "\n" in title
        ax.text(x + w / 2, y + h / 2 + ((3.2 if two else 2.2) if sub else 0), title, ha="center",
                va="center", fontsize=8.5, color=INK, fontweight="bold" if bold else "normal", family="serif")
        if sub:
            ax.text(x + w / 2, y + h / 2 - (2.6 if two else 1.6), sub, ha="center", va="center", fontsize=7,
                    color=INK2, family="serif")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     color=MUTED, lw=1))

    box(1, 26, 17, 15, "Genuine answer", "SciEntsBank, Beetle,\nMohler (real data)", fc="#ffffff")
    box(1, 4, 17, 17, "Attack generator", "A1 keyword stuffing\nA2 fluent-but-wrong\nA3 negation, A4 paraphrase\nA5 restate, A6-A7 white-box",
        fc="#fbe3d9", ec="#ec835a")
    box(25, 15, 14, 14, "Student answer S", "with question Q and\nreference R")
    arrow(18.5, 33, 25, 24)
    arrow(18.5, 12, 25, 19)
    crit = ["Semantic (SimCSE)", "Coverage (KeyBERT)", "Grammar (CoLA)", "Formality (RoBERTa)", "Logic (DeBERTa NLI)"]
    for i, c in enumerate(crit):
        y = 36 - i * 7.6
        box(44, y, 17, 5.2, c, fc="#e6f0fb", ec="#86b6ef", bold=False)
        arrow(39.5, 22, 43.6, y + 2.6)
        ax.plot([61.5, 64], [y + 2.6, y + 2.6], color=MUTED, lw=1)
    # Collector bus: all five scores feed both the grade and the detector.
    ax.plot([64, 64], [7.2, 38.6], color=MUTED, lw=1)
    arrow(64, 32.5, 68.6, 32.5)
    arrow(64, 11.5, 68.6, 11.5)
    box(69, 27, 16, 11, "Weighted grade", "NNLS weights + caps\n-> 3-way label")
    box(69, 4, 16, 15, "Disagreement\ndetector", "CDI = cov - logic\nD2: LR on gaps\nD3: boosting", fc="#dff3ea",
        ec="#1baf7a")
    box(89, 27, 10, 11, "Grade", "returned", fc="#ffffff")
    box(89, 5, 10, 13, "Flag", "route to\nteacher", fc="#ffffff")
    arrow(85.5, 32.5, 88.6, 32.5)
    arrow(85.5, 11.5, 88.6, 11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_framework.png", dpi=200, facecolor="#ffffff")
    plt.close(fig)


def fig_roc():
    p = RESULTS_DIR / "detector_roc.json"
    if not p.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roc = read_json(p)
    names = {"D0_cdi_rule": "D0 CDI rule", "D1_cdi_score": "D1 CDI score",
             "D2_disagreement_lr": "D2 disagreement LR", "D3_boosting": "D3 boosting"}
    cols = dict(zip(names, ["#eda100", "#eb6834", "#2a78d6", "#1baf7a"]))
    ds = [d for d in DATASET_LABEL if d in roc]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.4 * len(ds), 3.2), facecolor=SURF, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        ax.set_facecolor(SURF)
        ax.plot([0, 1], [0, 1], color=BASE, linewidth=0.8, linestyle=":")
        for k, lab in names.items():
            if k in roc[d]:
                ax.plot(roc[d][k]["fpr"], roc[d][k]["tpr"], color=cols[k], linewidth=2, label=lab)
        ax.axvline(0.05, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_title(DATASET_LABEL[d], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("False-positive rate", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color=GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("True-positive rate", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_roc.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def build(datasets: List[str]):
    fig_framework()
    fig_roc()
    parts = ["# Per-criterion robustness study — results", "",
             f"Generated by `python -m research.run_robustness report` at commit `{git_commit()}`.", ""]
    for name in datasets:
        sp = RESULTS_DIR / name / "sut_config.json"
        if sp.exists():
            h = read_json(sp)["systems"][0]
            parts.append(f"- **{DATASET_LABEL[name]}** Hybrid weights (NNLS on train): "
                         + ", ".join(f"{k}={v:.2f}" for k, v in h["weights"].items())
                         + f"; label thresholds (dev) = {h['thresholds']}")
    parts += ["", "## 1. Clean agreement (QWK, 3-way; 95% CI = question-level bootstrap)", "",
              _md(table_clean()), ""]
    if (RESULTS_DIR / "attack_results.csv").exists():
        fig_asr()
        fig_criterion_shift()
        parts += ["## 2. Attack success (% graded above the expected label; A4 = % whose label drops)", "",
                  "![ASR](figures/fig_asr.png)", ""]
        for name in datasets:
            t = table_asr(name)
            if len(t):
                parts += [f"### {DATASET_LABEL[name]}", "", _md(t), ""]
                t.to_latex(RESULTS_DIR / f"table_asr_{name}.tex", index=False)
        parts += ["## 3. Which criterion is fooled (mean shift in SD units vs. reference answers)", "",
                  "Reference = the source answer for A2/A3/A4, genuine incorrect answers otherwise.", "",
                  "![criterion shift](figures/fig_criterion_shift.png)", ""]
    if (RESULTS_DIR / "detector_overall.csv").exists():
        d = pd.read_csv(RESULTS_DIR / "detector_overall.csv")
        cols = ["dataset", "detector", "auroc", "tpr@5%fpr(oracle)", "tpr(dev thr)", "fpr_clean(dev thr)",
                "fpr_clean_correct", "fpr_A4_paraphrase", "asr_undefended", "asr_defended"]
        parts += ["## 4. Detection (threshold set for 5% flags on genuine dev answers)", "",
                  _md(d[cols].round(3)), ""]
        l = pd.read_csv(RESULTS_DIR / "detector_loao.csv")
        if len(l):
            parts += ["### Leave-one-attack-family-out", "", _md(l.round(3)), ""]
        coefs = read_json(RESULTS_DIR / "detector_lr_coefficients.json")
        parts += ["### D2 logistic-regression coefficients (standardised)", "",
                  _md(pd.DataFrame(coefs).round(2).reset_index().rename(columns={"index": "feature"})), ""]
    if (RESULTS_DIR / "adaptive_search.csv").exists():
        fig_adaptive()
        a = pd.read_csv(RESULTS_DIR / "adaptive_search.csv")
        s = a.groupby(["dataset", "budget"])[["success_vs_D2", "evades_D3", "graded_correct"]].mean().round(3)
        parts += ["## 5. Adaptive attacks", "", "![adaptive](figures/fig_adaptive.png)", "",
                  _md(s.reset_index()), ""]
        if (RESULTS_DIR / "adaptive_llm.csv").exists():
            al = pd.read_csv(RESULTS_DIR / "adaptive_llm.csv")
            parts += ["### LLM-written coherent stuffing", "",
                      _md(al.groupby("dataset")[["graded_above_incorrect", "flag_D2", "flag_D3", "success_vs_D2"]]
                          .mean().round(3).reset_index()), ""]
    vdir = RESULTS_DIR / "validation"
    if (vdir / "auto_checks.csv").exists():
        v = pd.read_csv(vdir / "auto_checks.csv")
        parts += ["## 6. Label validation", "", "Automatic checks (rows failing are excluded):", "",
                  _md(v.groupby("attack")["ok"].agg(["mean", "count"]).round(3).reset_index()), ""]
    if (vdir / "human_agreement.json").exists():
        h = read_json(vdir / "human_agreement.json")
        parts += [f"Human validation: κ_qw = {h['kappa_qw']:.3f} on {h['n_both']} items.", "",
                  _md(pd.DataFrame(h["per_attack"]).round(3)), ""]
    else:
        parts += ["**Human validation pending**: fill `validation/annotator_1.csv` and `annotator_2.csv`, "
                  "then run `python -m research.run_robustness validate agree`.", ""]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", RESULTS_DIR / "REPORT.md")
