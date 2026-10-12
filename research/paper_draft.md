# A Hybrid Framework for Deep Learning-Based Short Answer Assessment with Deterministic Linguistic Indicators

---

> **Abstract** — Automated grading of short student responses often trades interpretability for accuracy, leaving little insight into why a grade was assigned. This paper proposes a hybrid framework for Deep Learning-based Short Answer Assessment (DLSAA) that separates transparent score computation from feedback generation. Five parallel modules extract Deterministic Linguistic Indicators, covering semantic similarity, keyword coverage, grammatical correctness, writing formality, and logical consistency, aggregated through a transparent weighted mechanism into the final grade. A compact instruction-tuned large language model then converts these indicators into coherent, teacher-style feedback without altering the score. Compared with standalone Qwen2.5-1.5B and Qwen2.5-3B baselines, the hybrid model raises accuracy from 75.85% to 83.50%, macro-F1 from 75.50% to 82.90%, and quadratic weighted kappa from 72.40% to 85.50%, while cutting hallucination rate from 10.73% to 1.24%. Case studies show the hybrid model correctly flags logical and factual errors that standalone language model graders overlook and reward with inflated scores, demonstrating more accurate and trustworthy automated grading.
>
> **Keywords** — Automated Short Answer Assessment; Deterministic Linguistic Indicators; Large Language Models; Interpretable Grading; Automated Feedback Generation.

---

## I. INTRODUCTION

Deep learning based short answer assessment (DLSAA) has become an active research area in educational artificial intelligence, offering scalable grading while reducing instructors' workload [1]. However, most existing systems prioritize accuracy over transparency. As illustrated in Fig. 1, the conventional pipeline compresses semantic encoding, response matching, and score prediction into an end-to-end latent representation, leaving little room for interpretation [2]. This opacity undermines teacher confidence, since instructors cannot inspect which part of a response drove a given grade [3], and it weakens formative evaluation, as a single numeric score gives students no indication of which aspect of their answer was weak, a limitation also documented for automated essay scoring [4].

> **Fig. 1.** Conventional end-to-end DLSAA pipeline.

The two dominant paradigms addressing this gap exhibit complementary rather than overlapping strengths, summarized in Table I. Explainable models, typically transformer encoders such as BERT, RoBERTa, and SBERT fine-tuned on grading data, yield deterministic and reproducible scores but operate on internal representations rather than the student's actual response, offering limited criterion-level evidence [2], [3], [7], [4]. Deterministic similarity-based variants, such as SBERT-based grading without any LLM, further improve reproducibility but generalize poorly to out-of-sample questions [8]. LLM-based systems, in contrast, produce rich natural language feedback but are comparatively non-deterministic, and their advantage over fine-tuned models remains inconsistent: some benchmarks find fine-tuned models still outperform GPT-4 despite extensive prompt engineering [5], while others find GPT-4 comparable to hand-engineered baselines but weaker than specialized fine-tuned LLMs, with performance varying by subject [6], [9]. This inconsistency is compounded by domain and language sensitivity, as observed in software engineering coursework [10], medical education [11], and Finnish-language grading, where GPT-4 achieved good agreement in fewer than half of tested settings [12].

<br>

<div align="center">

**TABLE I**
**SUMMARY OF REPRESENTATIVE DLSAA STUDIES**

| Research Problem                | Representative Approach                    | Works           | Year       |
|:--------------------------------|:-------------------------------------------|:----------------|:-----------|
| Deep-learning evolution         | Transformer encoders (BERT/RoBERTa/SBERT)  | [4], [7]        | 2024; 2022 |
| LLM-vs-fine-tuned benchmarking  | GPT-4 prompting vs. fine-tuned models      | [5], [6], [9]   | 2025; 2024 |
| Domain/language sensitivity     | Cross-lingual, technical, medical grading  | [10], [11], [12]| 2024       |
| Rubric-guided grading           | Self-reflective guideline optimization     | [14]            | 2024       |
| Deterministic similarity grading| SBERT retrieval, no LLM                    | [8]             | 2021       |
| Long-answer grading             | Multi-sentence grading pipelines           | [13]            | 2024       |

</div>

<br>

Despite this progress, key challenges remain. Datasets exhibit limited domain diversity, and most systems still provide only a holistic score without criterion-level evidence [1], a limitation more pronounced for longer, multi-sentence responses [13]. Many rubric-based methods still depend on manually crafted guidelines; recent work has begun automating this via self-reflective optimization, in which an LLM agent refines guidelines from its own errors [14], though an initial human-authored rubric remains necessary. Few hybrid approaches explicitly separate deterministic score computation from LLM-based explanation [3], leaving interpretable assessment and consistent feedback generation as an open challenge.

As summarized in Fig. 2, the deterministic but evidence-limited nature of explainable models and the expressive but non-deterministic nature of LLM-based systems point toward the same conclusion: neither paradigm alone is sufficient, and their strengths should instead be combined. A promising direction is to decompose grading into explicit criterion-level linguistic evidence rather than relying solely on latent representations [3]. This study refers to such evidence as Deterministic Linguistic Indicators (DLIs), comprising semantic similarity, keyword coverage, grammatical correctness, writing formality, and logical consistency. These indicators provide explicit and inspectable evidence for individual assessment criteria.

> **Fig. 2.** Motivation for combining Deterministic Linguistic Indicators with LLM synthesis for interpretable grading.

To address this gap, this study proposes a framework that combines deterministic criterion-level scoring with LLM-based feedback synthesis. Five independent modules extract the DLIs, and their resulting scores are aggregated through a transparent weighted scoring mechanism to produce the final grade. A compact instruction-tuned LLM then synthesizes the extracted evidence into teacher-style feedback without modifying the computed score. The contributions are threefold:

1. An interpretable Deterministic Linguistic Indicator framework representing five complementary linguistic dimensions for transparent criterion-level assessment.
2. An LLM synthesis module that transforms deterministic evidence into coherent teacher-style feedback while preserving grading consistency.
3. An interpretable grading architecture combining parallel deterministic assessment with transparent weighted aggregation and evidence-grounded feedback generation.

---

## II. METHODS

### A. System Overview

The proposed DLSAA framework processes an input triplet

$$x = \{Q,\; R,\; S\} \tag{1}$$

where *Q* denotes the question, *R* the reference answer, and *S* the student response. The system produces a final numerical score *y* ∈ [0, 10] together with teacher-style natural-language feedback *F*.

> **Fig. 3.** Architecture of the proposed DLSAA framework.

As illustrated in Fig. 3, the pipeline is organized into three sequential phases. In **Phase 1**, five independent expert models extract a set of Deterministic Linguistic Indicators (DLIs)

$$\mathbf{z} = [\,s_{\text{sem}},\; s_{\text{cov}},\; s_{\text{gra}},\; s_{\text{for}},\; s_{\text{log}}\,] \tag{2}$$

where each component is normalized to the interval [0, 1]. These indicators constitute explicit, inspectable evidence for the subsequent grading decision. In **Phase 2**, the indicators are combined through a transparent weighted aggregation mechanism to obtain the final score *y*. Finally, in **Phase 3**, a compact instruction-tuned large language model (Qwen2.5-3B-Instruct) synthesizes the original triplet, the extracted indicators, and the computed score into coherent teacher-style feedback, without altering the numerical grade.

By strictly separating deterministic score computation from feedback generation, the architecture guarantees both reproducibility of the grade and pedagogical quality of the explanation.

### B. Extraction of Deterministic Linguistic Indicators

Five complementary linguistic dimensions are evaluated independently by specialized pretrained models. Each model outputs a normalized score in [0, 1], ensuring that every assessment criterion remains explicit and interpretable.

#### 1) Semantic Similarity

Semantic similarity quantifies the degree to which the student response conveys the same underlying meaning as the reference answer, irrespective of surface-form differences. To obtain dense sentence representations, we employ SimCSE (Simple Contrastive Learning of Sentence Embeddings), a contrastive learning framework that produces high-quality sentence embeddings by maximizing agreement between differently dropout-augmented views of the same sentence. Both the reference answer *R* and the student answer *S* are encoded by the pretrained SimCSE model:

$$e_R = \text{SimCSE}(R) \tag{3}$$

$$e_S = \text{SimCSE}(S) \tag{4}$$

Their cosine similarity is then linearly mapped to the unit interval:

$$s_{\text{sem}} = \frac{\cos(e_R,\, e_S) + 1}{2} \tag{5}$$

For illustration, consider a student answer that replaces "executes program instructions" with "processes instructions." Although the wording differs, the semantic content remains equivalent, resulting in a high similarity score of approximately *s*_sem = 0.93.

#### 2) Keyword Coverage

Semantic similarity by itself may fail to detect the absence of essential concepts. A second expert module is therefore introduced to verify whether the key ideas present in the reference answer are actually covered by the student response.

KeyBERT first extracts the top-*N* most representative keywords

$$K = \{k_1,\, k_2,\, \ldots,\, k_N\} \tag{6}$$

from the reference answer. Each keyword embedding is then compared with the embedding of the student response via cosine similarity

$$m_i = \cos(k_i,\, e_S) \tag{7}$$

A keyword is judged to be covered whenever

$$m_i > \tau_k \tag{8}$$

where the decision threshold is fixed at *τ*_k = 0.50. This value was selected based on the cosine similarity distribution observed over the training partition, where it corresponds to the elbow point that maximizes the separation between matched and unmatched keywords. A sensitivity analysis over *τ*_k ∈ {0.30, 0.40, 0.50, 0.60, 0.70} (reported in Table IV) confirms that QWK varies by at most ±1.20 percentage points within ±0.10 of the chosen threshold, indicating that the framework is not overly sensitive to this parameter. It should be noted that KeyBERT matching operates on embedding similarity rather than exact string matching, which provides partial robustness to synonyms. However, domain-specific paraphrases (e.g., "central processing unit" vs. "processor") may still fall below the threshold; this limitation is discussed further in Section V.

The final coverage score is the proportion of matched keywords:

$$s_{\text{cov}} = \frac{|\{m_i > \tau_k\}|}{N} \tag{9}$$

In the running example, KeyBERT identifies a set of core concepts such as "instructions," "calculations," and "control." Six out of the ten extracted keywords satisfy the similarity threshold, resulting in a coverage score of *s*_cov = 0.60.

#### 3) Grammatical Correctness

Grammatical quality is assessed by the pretrained RoBERTa-CoLA model, which predicts the linguistic acceptability of a sentence. The corresponding grammar score is defined as

$$s_{\text{gra}} = P(\text{acceptable} \mid S) \tag{10}$$

Because the student response in the running example is free of grammatical errors, the model assigns it a score of *s*_gra = 0.94.

#### 4) Writing Formality

Academic assessment normally requires responses to adopt a formal register. Consequently, a RoBERTa-based formality classifier is employed to estimate the probability that the student answer is written in an appropriate academic style:

$$s_{\text{for}} = P(\text{formal} \mid S) \tag{11}$$

The illustrative response uses suitably formal academic language and therefore receives a formality score of *s*_for = 0.91.

#### 5) Logical Consistency

An answer that is both semantically close to the reference and grammatically well-formed may nevertheless contain internal contradictions. To capture this phenomenon a DeBERTa-v3 Natural Language Inference (NLI) cross-encoder is used to compute the probabilities of entailment *P*_e and contradiction *P*_c between the reference and the student answer:

$$[P_e,\, P_c] = \text{DeBERTa}_{\text{NLI}}(R,\, S) \tag{12}$$

These two probabilities are subsequently transformed into a normalized consistency score

$$s_{\text{log}} = \frac{P_e - P_c + 1}{2} \tag{13}$$

Any response for which the contradiction probability exceeds the fixed threshold *δ*_c = 0.60 is regarded as logically inconsistent. This threshold was calibrated on the training partition by selecting the value that maximizes the F1 score for detecting human-annotated contradictory responses. A sensitivity analysis over *δ*_c ∈ {0.40, 0.50, 0.60, 0.70, 0.80} (Table IV) shows that QWK is robust within ±0.90 percentage points across the tested range.

In the concrete example the NLI model yields *P*_e = 0.92 and *P*_c = 0.02, which translates into a logical-consistency score of *s*_log = 0.95.

Collecting the five individual scores produces the complete criterion-level feature vector

$$\mathbf{z} = [\,s_{\text{sem}},\; s_{\text{cov}},\; s_{\text{gra}},\; s_{\text{for}},\; s_{\text{log}}\,] = [\,0.93,\; 0.60,\; 0.94,\; 0.91,\; 0.95\,] \tag{14}$$

which is passed as input to the weighted aggregation stage described next.

### C. Interpretable Grading and LLM Synthesis

The five Deterministic Linguistic Indicators are aggregated by means of a transparent weighted linear combination to produce the final grade *y*:

$$y = 10 \sum_{i=1}^{5} w_i \cdot s_i \tag{15}$$

The weight vector is defined as

$$\mathbf{w} = [\,w_{\text{sem}},\; w_{\text{cov}},\; w_{\text{gra}},\; w_{\text{for}},\; w_{\text{log}}\,] = [\,0.15,\; 0.30,\; 0.10,\; 0.05,\; 0.40\,] \tag{16}$$

where ∑*w*_i = 1. This weighting assigns the highest importance to logical consistency (*w*_log = 0.40) and keyword coverage (*w*_cov = 0.30), reflecting the pedagogical priority that a correct answer must be both logically sound and substantively complete. The optimization procedure and justification for these weights are described in Section III-C.

> **Fig. 4.** Interpretable grading and LLM-based feedback generation pipeline.

As illustrated in Fig. 4, once the numerical score has been determined, Qwen2.5-3B-Instruct receives a structured prompt that contains the original triplet (*Q*, *R*, *S*), the five criterion-level scores, and the already-computed grade. The model is then responsible solely for generating teacher-style feedback. Because the score itself is fixed before the language model is invoked, the resulting explanation remains fully consistent with the assigned grade while still providing constructive, evidence-grounded suggestions for improvement.

---

## III. EXPERIMENTS AND RESULTS

### A. Datasets and Base Models

Experiments were conducted on two widely used public benchmarks for Deep Learning-based Short Answer Assessment (DLSAA): the SemEval-2013 Task 7 [15] dataset and the Mohler (2011) dataset. The SemEval-2013 corpus comprises two subsets. The Beetle subset contains approximately 4,000 student answers to questions in basic electricity and electronics, while the SciEntsBank subset contains more than 10,000 student answers covering a broader range of science topics. Each sample consists of a question, one or more reference answers, a student response, and a human-assigned score. The Mohler dataset contains 2,442 student answers to 87 questions drawn from an introductory computer-science course on data structures; scores range from 0 to 5 and were averaged from two independent human graders. All scores were normalized to the interval [0, 10] to match the output scale of the proposed pipeline.

The feedback-generation backbone is Qwen2.5-3B-Instruct. The five Deterministic Linguistic Indicator modules employ SimCSE for semantic similarity, KeyBERT for keyword coverage, RoBERTa-CoLA for grammatical correctness, a RoBERTa-based formality classifier, and DeBERTa-v3 for logical consistency (NLI).

### B. Evaluation Metrics

Quantitative performance is measured by Quadratic Weighted Kappa (QWK) for agreement with human scores, Accuracy and Macro-F1 across discrete score categories, JSON Compliance Rate for structural validity of the generated output, and Hallucination Rate for factual inconsistency between the feedback and the extracted indicators. Qualitative analysis further examines pedagogical accuracy, explainability, and actionability of the generated feedback.

**Score Discretization.** Because Accuracy and Macro-F1 require discrete categories, continuous scores in [0, 10] are mapped to three ordinal classes using the following rule:

$$\text{category}(y) = \begin{cases} \textbf{Correct} & \text{if } y \geq 7.5 \\[4pt] \textbf{Partially Correct} & \text{if } 4.5 \leq y < 7.5 \\[4pt] \textbf{Incorrect} & \text{if } y < 4.5 \end{cases} \tag{17}$$

These boundaries mirror the scoring convention used in the SemEval-2013 Task 7 three-way evaluation [15] and are applied consistently to both the model output and the human gold scores before computing Accuracy and Macro-F1.

**Hallucination Evaluation Protocol.** A feedback statement is classified as hallucinated if it meets any of the following criteria: (a) it attributes a claim to the student response that does not appear therein, (b) it misquotes or fabricates content from the reference answer, or (c) it asserts a factual judgment unsupported by the extracted DLI scores. Two of the authors independently annotated a random sample of 200 feedback outputs (100 from the hybrid model, 100 from the Qwen2.5-3B baseline). Inter-rater agreement was measured using Cohen's kappa, yielding κ = 0.82, indicating strong agreement. Disagreements were resolved through discussion. We acknowledge that this preliminary manual evaluation has limited scale and is conducted by the paper authors; future work will incorporate external annotators and a larger sample to strengthen external validity.

### C. Experimental Setup

All experiments were executed on a local workstation equipped with an NVIDIA RTX 4090 GPU (24 GB VRAM) with CUDA acceleration. The five lightweight expert models (SimCSE, KeyBERT, RoBERTa-CoLA, RoBERTa-Formality, DeBERTa-v3-NLI) were run in parallel via Python `ThreadPoolExecutor` with a batch size of 1 per sample. Qwen2.5-3B-Instruct was quantized to 4-bit NF4 using BitsAndBytes, reducing GPU memory from approximately 7 GB (FP16) to approximately 2.5 GB. Peak GPU memory during inference was approximately 5.8 GB, including all five DLI models loaded simultaneously.

**Question-Level Data Split.** To prevent data leakage, the dataset was split at the question level rather than the answer level. All student responses to a given question were assigned exclusively to either the training or the test partition. This ensures that the model is evaluated on questions whose reference answers were never seen during weight optimization. Specifically, 70% of questions were allocated to the training set and 30% to the test set, stratified to maintain approximate label distribution balance.

**Weight Optimization.** Feature weights were optimized on the training partition (70% of questions) using constrained linear regression with non-negativity constraints, followed by normalization so that ∑*w*_i = 1. Random-forest regression was additionally employed as a non-linear baseline to gauge the theoretical upper bound of feature importance. The final weight configuration is:

$$\mathbf{w} = [\,w_{\text{sem}},\; w_{\text{cov}},\; w_{\text{gra}},\; w_{\text{for}},\; w_{\text{log}}\,] = [\,0.15,\; 0.30,\; 0.10,\; 0.05,\; 0.40\,]$$

To mitigate overfitting risk, 5-fold cross-validation was performed within the training partition. The final weight vector represents the mean across folds; the standard deviation across folds was below 0.02 for each weight, indicating stable convergence. Crucially, the test set (30% of questions) was not used at any stage during weight optimization.

**Contradiction Cap.** Whenever the NLI contradiction probability exceeds 0.90, the final score is capped at 4.0. This safety heuristic ensures that when the NLI model assigns very high contradiction probability, the response — which almost certainly contains a fundamental factual or logical error — does not receive a passing grade regardless of surface-level quality. The threshold of 0.90 was chosen conservatively to minimize false positives. A sensitivity analysis (Table V) shows that varying the cap threshold between 0.80 and 0.95 changes QWK by at most ±0.80 percentage points.

### D. Quantitative Results

Table II reports the quantitative results. The Hybrid Model (Full 5+1) obtains the best Accuracy (83.50%), Macro-F1 (82.90%), and QWK (85.50%) among all the studied systems. It also attains the best JSON Compliance Rate (99.12%) and the lowest Hallucination Rate (1.24%). In comparison with the baseline Qwen2.5-3B model, the hybrid model obtains a QWK that is 13.10 percentage points higher and a hallucination rate that is 9.49 percentage points lower, reflecting better agreement with human grading and more accurate feedback generation. The ablation study result also indicates that removing the logic module leads to a reduction in QWK from 85.50% to 75.20%.

<br>

<div align="center">

**TABLE II**
**QUANTITATIVE RESULTS**

| Metric             | Qwen2.5-1.5B | Qwen2.5-3B | Hybrid (−Logic) | Hybrid (Full 5+1) |
|:-------------------|:------------:|:----------:|:---------------:|:-----------------:|
| Accuracy           |    68.42%    |   75.85%   |     78.10%      |    **83.50%**     |
| Macro-F1           |    65.20%    |   75.50%   |     77.54%      |    **82.90%**     |
| QWK                |    60.15%    |   72.40%   |     75.20%      |    **85.50%**     |
| JSON Compliance    |    85.30%    |   92.47%   |     98.85%      |    **99.12%**     |
| Hallucination Rate |    18.50%    |   10.73%   |      4.15%      |     **1.24%**     |

</div>

<br>

### E. Component-Level Ablation Study

To quantify the contribution of each Deterministic Linguistic Indicator, a comprehensive ablation study was conducted. In each variant, one DLI module is removed (its weight set to zero and redistributed proportionally among the remaining modules) while all other components are kept intact. Results are reported on the held-out test set.

<br>

<div align="center">

**TABLE III**
**COMPONENT-LEVEL ABLATION RESULTS**

| Variant                  | Accuracy     | Macro-F1     | QWK          | Δ QWK  |
|:-------------------------|:------------:|:------------:|:------------:|:------:|
| **Full Model (5+1)**     | **83.50%**   | **82.90%**   | **85.50%**   |   —    |
| −Semantic                |    79.30%    |    78.45%    |    80.20%    | −5.30  |
| −Keyword Coverage        |    77.80%    |    77.10%    |    78.60%    | −6.90  |
| −Grammar                 |    82.90%    |    82.35%    |    84.80%    | −0.70  |
| −Formality               |    83.20%    |    82.60%    |    85.20%    | −0.30  |
| −Logic                   |    78.10%    |    77.54%    |    75.20%    | −10.30 |
| −LLM Synthesis           |    83.50%    |    82.90%    |    85.50%    |  0.00  |

</div>

<br>

The ablation results reveal a clear hierarchy of component importance. Removing logical consistency (−Logic) causes the largest QWK drop (−10.30 points), confirming that explicit contradiction detection is critical for reliable grading. Keyword coverage ranks second (−6.90 points), followed by semantic similarity (−5.30 points). Grammar and formality have comparatively smaller effects (−0.70 and −0.30 points respectively), consistent with the lower weights assigned to these dimensions. The −LLM Synthesis variant produces identical scores because the LLM module only generates feedback text and does not alter the computed numerical grade; its contribution is instead measured via the hallucination rate and qualitative feedback quality.

### F. Threshold Sensitivity Analysis

To validate the robustness of the chosen thresholds, sensitivity analyses were conducted for both the keyword matching threshold *τ*_k and the contradiction detection threshold *δ*_c.

<br>

<div align="center">

**TABLE IV**
**THRESHOLD SENSITIVITY ANALYSIS**

</div>

<div align="center">

**(a) Keyword Matching Threshold *τ*_k**

| *τ*_k     | QWK          | Macro-F1     |
|:---------:|:------------:|:------------:|
| 0.30      |    83.80%    |    81.50%    |
| 0.40      |    84.90%    |    82.40%    |
| **0.50**  | **85.50%**   | **82.90%**   |
| 0.60      |    84.80%    |    82.30%    |
| 0.70      |    83.20%    |    80.90%    |

</div>

<br>

<div align="center">

**(b) Contradiction Detection Threshold *δ*_c**

| *δ*_c     | QWK          | Macro-F1     |
|:---------:|:------------:|:------------:|
| 0.40      |    84.70%    |    82.10%    |
| 0.50      |    85.10%    |    82.60%    |
| **0.60**  | **85.50%**   | **82.90%**   |
| 0.70      |    85.20%    |    82.70%    |
| 0.80      |    84.60%    |    82.00%    |

</div>

<br>

Results confirm that performance is robust within ±0.10 of the selected thresholds, with QWK varying by at most 1.70 percentage points for *τ*_k and 0.90 percentage points for *δ*_c. The chosen values (*τ*_k = 0.50, *δ*_c = 0.60) yield the best or near-best performance across both metrics.

<br>

<div align="center">

**TABLE V**
**CONTRADICTION CAP THRESHOLD SENSITIVITY**

| Cap Threshold (*p*_c) | Cap Score | QWK          | Accuracy     |
|:---------------------:|:---------:|:------------:|:------------:|
| 0.80                  |   ≤ 4.0   |    84.70%    |    82.90%    |
| 0.85                  |   ≤ 4.0   |    85.10%    |    83.20%    |
| **0.90**              | **≤ 4.0** | **85.50%**   | **83.50%**   |
| 0.95                  |   ≤ 4.0   |    85.30%    |    83.40%    |

</div>

<br>

### G. Inference Latency

Table VI reports the inference latency per sample, measured as the mean over 100 samples on the test set.

<br>

<div align="center">

**TABLE VI**
**INFERENCE LATENCY BREAKDOWN (PER SAMPLE)**

| Component                              | Latency (ms) |
|:---------------------------------------|:------------:|
| SimCSE (Semantic Similarity)           |     ~45      |
| KeyBERT (Keyword Coverage)             |     ~38      |
| RoBERTa-CoLA (Grammar)                 |     ~32      |
| RoBERTa-Formality (Formality)          |     ~30      |
| DeBERTa-v3 NLI (Logic)                 |     ~55      |
| **DLI Total (parallel execution)**     |   **~62**    |
| Weighted Aggregation                   |      <1      |
| Qwen2.5-3B Feedback Generation (4-bit) |    ~850      |
| **End-to-End Total**                   |  **~915**    |

</div>

<br>

The five DLI modules execute in parallel, so the total DLI extraction time is bounded by the slowest module (DeBERTa-v3 NLI at ~55 ms) plus thread scheduling overhead. The dominant latency bottleneck is Qwen2.5-3B-Instruct feedback generation (~850 ms), which accounts for over 90% of the total inference time. In score-only mode (without feedback generation), the system processes each sample in approximately 65 ms, which is suitable for real-time classroom deployment. Future work will explore quantization and distillation strategies to reduce feedback generation latency.

### H. Cross-Domain Generalization

To evaluate whether the optimized weights generalize beyond the training domain, a cross-domain transfer experiment was conducted: weights optimized on SemEval-2013 (science domain) were applied without retuning to the Mohler dataset (computer science domain), and vice versa.

<br>

<div align="center">

**TABLE VII**
**CROSS-DOMAIN TRANSFER RESULTS**

| Configuration                      | QWK          | Accuracy     | Macro-F1     |
|:-----------------------------------|:------------:|:------------:|:------------:|
| SemEval → SemEval (in-domain)      | **85.50%**   | **83.50%**   | **82.90%**   |
| SemEval → Mohler (cross-domain)    |    82.30%    |    80.80%    |    79.60%    |
| Mohler → Mohler (in-domain)        |    84.80%    |    82.90%    |    82.10%    |
| Mohler → SemEval (cross-domain)    |    81.70%    |    80.20%    |    79.10%    |

</div>

<br>

Cross-domain transfer reduces QWK by 2–4 percentage points compared to in-domain evaluation, suggesting that the learned weights capture general linguistic quality dimensions rather than dataset-specific artifacts. This moderate degradation is expected, as keyword coverage and logical consistency indicators are inherently influenced by domain vocabulary. Nevertheless, the cross-domain configurations still outperform the standalone Qwen2.5-3B baseline (QWK = 72.40%) by a substantial margin.

### I. Qualitative Analysis

Table VIII presents a representative case. Given the question "What is the primary function of red blood cells?", the student incorrectly states that red blood cells help fight infections, which is a factual error. The Base LLM has a tendency to score student answers containing logical or factual errors quite high, scoring 100/100 for the Stack–Queue question and 95/100 for the red-blood-cell question. On the contrary, the hybrid model gives more appropriate scores of 50/100 and 60/100, respectively, as it detects the faulty definition of LIFO and the erroneous fact about the role of red blood cells. Thus, these examples illustrate the advantage of the hybrid model over the Base LLM in terms of more accurate, error-sensitive, and evidence-based feedback.

<br>

<div align="center">

**TABLE VIII**
**QUALITATIVE CASE STUDY**

</div>

<div align="center">

**(a) Example 1 — Computer Science Domain**

| Criteria         | Content                                                                                                                                                                                                    |
|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Question**     | Explain the main difference between a Stack and a Queue.                                                                                                                                                   |
| **Reference**    | A Stack follows the Last-In-First-Out (LIFO) principle, while a Queue follows First-In-First-Out (FIFO).                                                                                                   |
| **Student**      | A Stack uses LIFO, meaning the first item added is the first one removed. A Queue uses FIFO.                                                                                                               |
| **Base LLM**     | [Grade: 100/100] Your answer is perfect. You correctly identified that Stack uses LIFO and Queue uses FIFO.                                                                                                |
| **Hybrid (5+1)** | [Grade: 50/100] Partially Correct. You correctly named LIFO and FIFO. However, your definition of LIFO is a logical error. LIFO means the last item added is removed first, not the first item.            |

</div>

<br>

<div align="center">

**(b) Example 2 — STEM / Biology Domain**

| Criteria         | Content                                                                                                                                                                                     |
|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Question**     | What is the primary function of red blood cells?                                                                                                                                            |
| **Reference**    | Red blood cells carry oxygen from the lungs to the body and bring carbon dioxide back.                                                                                                      |
| **Student**      | Red blood cells carry oxygen around the body and help fight infections.                                                                                                                     |
| **Base LLM**     | [Grade: 95/100] Excellent. You correctly stated that red blood cells carry oxygen and fight infections.                                                                                     |
| **Hybrid (5+1)** | [Grade: 60/100] Partially Correct. While you are right about carrying oxygen, stating they fight infections is a factual error (white blood cells do that).                                 |

</div>

<br>

### J. Results Analysis

> **Fig. 5.** Comprehensive analysis of experimental results.

Fig. 5 provides a comprehensive visualization of the experimental outcomes. The top-left panel summarizes the overall quantitative performance across Accuracy, Macro-F1, QWK, and Hallucination Rate. The top-middle panel shows the correlation between predicted scores and human scores, while the top-right panel compares the score distributions of the hybrid model against both human graders and the Qwen2.5-3B baseline. The bottom-left panel presents the absolute error distribution, demonstrating that the hybrid model yields significantly lower deviation from human scores. Finally, the bottom-right panel illustrates two qualitative cases in which the hybrid model correctly detects logical/factual errors and incomplete answers that the standalone LLM tends to overlook and over-score.

---

## IV. DISCUSSION

### A. Overall Performance and Ablation

The Hybrid Model (Full 5+1) achieves the highest JSON compliance rate (99.12%), lowest hallucination rate (1.24%), and strongest human agreement (QWK = 85.50%) among all examined models, demonstrating that separating deterministic score computation from LLM-generated feedback increases both grading consistency and feedback quality. The component-level ablation analysis (Table III) reveals that logical consistency is the most critical individual indicator, with its removal causing a 10.30-point QWK drop. Keyword coverage is the second most important (−6.90 points), followed by semantic similarity (−5.30 points). Grammar and formality contribute modestly, consistent with their lower optimized weights. These findings confirm that content-oriented and logic-oriented indicators drive grading accuracy, while surface-level quality indicators serve as supplementary refinements.

### B. Latency and Computational Analysis

The five lightweight DLI expert models execute in parallel, with the total DLI extraction time bounded by the slowest module at approximately 62 ms per sample. The dominant computational bottleneck is LLM inference for feedback generation (~850 ms per sample with 4-bit quantization). In score-only mode, the system achieves real-time latency (~65 ms per sample). The total peak GPU memory is approximately 5.8 GB, making the framework deployable on consumer-grade GPUs. Future research should focus on quantization, distillation, and speculative decoding to reduce feedback generation latency for interactive classroom applications.

### C. Error Analysis

As demonstrated in the qualitative case studies (Table VIII), a key advantage of the hybrid model is its ability to detect logical and factual errors at the criterion level. The standalone Qwen2.5-3B baseline tends to assign inflated scores to responses that are fluent and partially correct but contain fundamental conceptual errors. The hybrid model mitigates this failure mode through the logical consistency indicator, which explicitly quantifies the degree of contradiction between the student answer and the reference. This design makes the grading process more conservative and accurate for responses that "sound correct" but contain factual inaccuracies.

### D. Domain Considerations

The DeBERTa-v3 NLI model was trained on general-domain natural language inference data (MNLI, SNLI) and has not been fine-tuned on CS or STEM-specific terminology. While general-domain NLI models have demonstrated reasonable transfer to educational contexts, specialized terms such as "polymorphism," "encapsulation," or "electronegativity" may receive lower entailment scores when semantically equivalent to the reference but expressed in domain-specific vocabulary. This domain gap is partially mitigated by the multi-indicator design: semantic similarity (SimCSE) and keyword coverage (KeyBERT) compensate for NLI misclassifications by capturing meaning overlap through complementary pathways. The cross-domain transfer experiment (Table VII) further demonstrates that the overall framework maintains strong performance across science and computer science domains despite this limitation.

---

## V. LIMITATIONS

Several limitations should be acknowledged:

1. **Static Thresholds.** All decision thresholds (*τ*_k = 0.50, *δ*_c = 0.60, discretization boundaries at 4.5 and 7.5, contradiction cap at *p*_c > 0.90) are fixed and may require re-calibration for new educational domains or grading rubrics. Although the sensitivity analyses (Tables IV–V) show robustness within the tested range, the optimal values could shift for domains with different answer length distributions or vocabulary characteristics.

2. **Heuristic Contradiction Cap.** The rule that caps the final score at 4.0 when *p*_c > 0.90 is a safety heuristic rather than a learned decision boundary. While the sensitivity analysis supports its effectiveness on the current benchmarks, this hard cap may be overly aggressive or insufficient in domains where partial contradictions carry different pedagogical significance.

3. **KeyBERT Synonym and Paraphrase Limitation.** KeyBERT relies on embedding-based similarity for keyword matching, which provides partial robustness to lexical variation. However, valid domain-specific paraphrases, non-standard abbreviations, or multi-word expressions that differ substantially in surface form from the reference keywords may be penalized. For example, a student writing "CPU" instead of "central processing unit" may receive a lower coverage score despite expressing the same concept. Incorporating domain-specific synonym expansion or contextualized keyword extraction could address this limitation.

4. **NLI Domain Mismatch.** The DeBERTa-v3 NLI model is a general-domain model that may misclassify entailment relationships involving specialized terminology. In particular, domain-specific definitions (e.g., "a stack is a LIFO data structure") may receive lower entailment confidence than general-knowledge statements, potentially leading to underestimation of logical consistency for technically precise but domain-specific answers.

5. **Computational Cost.** Running five transformer-based expert models plus an instruction-tuned LLM requires dedicated GPU hardware. Although the 4-bit quantization of Qwen2.5-3B-Instruct reduces memory requirements to approximately 5.8 GB peak, real-time feedback generation (~850 ms per sample) may be insufficient for high-throughput scenarios such as live exam grading with hundreds of concurrent students. Score-only mode (~65 ms) is feasible for real-time use.

6. **Manual Hallucination Evaluation.** The hallucination rate was assessed through a preliminary manual evaluation conducted by two of the paper authors on 200 sampled feedback outputs. While inter-rater agreement was strong (κ = 0.82), the limited sample size and potential author bias reduce the external validity of this metric. Scaling the evaluation to independent, domain-expert annotators with a larger and more diverse sample is a necessary step for future validation.

7. **Dataset Scope.** The current evaluation is limited to two benchmarks covering science (SemEval-2013) and introductory computer science (Mohler). The framework's performance on other educational domains (e.g., humanities, social sciences, mathematics), non-English languages, and longer free-text responses remains to be validated.

8. **Weight Generalization.** Although the cross-domain transfer experiment (Table VII) shows moderate generalization, the optimized weights are derived from a constrained linear regression on the training data, which may not capture non-linear interactions between DLI dimensions. Non-linear aggregation methods or attention-based weighting could potentially improve performance but would reduce interpretability.

---

## VI. CONCLUSION AND FUTURE WORK

Current DLSAA systems often trade off feedback quality and transparency for accuracy. This study proposes HybridASAG, which scores responses through five deterministic linguistic dimensions with transparent weighted aggregation, while Qwen2.5-3B-Instruct generates evidence-grounded teacher-style feedback. By separating deterministic score computation from feedback synthesis, the framework enables criterion-level interpretability while improving reliability, achieving a QWK of 85.50%, a JSON compliance rate of 99.12%, and a hallucination rate of 1.24%. These results position HybridASAG as a practical and transparent alternative to monolithic LLM-based grading. The component-level ablation results highlight the importance of the logical consistency module, as removing this module reduces QWK from 85.50% to 75.20%, demonstrating that semantic similarity alone is insufficient for reliable short-answer assessment. This finding shows that explicitly modeling logical consistency plays an important role in building a transparent grading system.

The current evaluation is limited to the selected ASAG benchmarks and focuses primarily on responses in science and computer science. Therefore, future studies will focus on reducing LLM inference latency through quantization and distillation, extending the framework to broader datasets and different educational domains, and incorporating domain-specific keyword expansion and fine-tuned NLI models to further improve criterion-level accuracy. In addition, further work could explore non-linear aggregation methods, external hallucination evaluation with independent annotators, and real-time deployment studies in live classroom settings to validate the system's practical utility and robustness.

---

## REFERENCES

[1] D. Aggarwal *et al.*, "'I understand why I got this grade': Automatic Short Answer Grading (ASAG) with Feedback," in *Proc. Int. Conf. Artificial Intelligence in Education (AIED)*. Cham: Springer, 2025.

[2] A. Condor and Z. Pardos, "Explainable automatic grading with neural additive models," in *Proc. Int. Conf. Artificial Intelligence in Education (AIED)*. Cham: Springer, 2024.

[3] F. Künnecke *et al.*, "Enhancing multi-domain automatic short answer grading through an explainable neuro-symbolic pipeline," *arXiv preprint arXiv:2403.01811*, 2024.

[4] H. Misgna *et al.*, "A survey on deep learning-based automated essay scoring and feedback generation," *Artificial Intelligence Review*, vol. 58, no. 2, p. 36, 2024.

[5] R. Ferreira Mello *et al.*, "Automatic short answer grading in the LLM era: Does GPT-4 with prompt engineering beat traditional models?," in *Proc. 15th Int. Learning Analytics and Knowledge Conf. (LAK)*, 2025.

[6] G. Kortemeyer, "Performance of the pre-trained large language model GPT-4 on automated short answer grading," *Discover Artificial Intelligence*, vol. 4, no. 1, p. 47, 2024.

[7] S. Haller *et al.*, "Survey on automated short answer grading with deep learning: From word embeddings to transformers," *arXiv preprint arXiv:2204.03503*, 2022.

[8] A. Condor, M. Litster, and Z. Pardos, "Automatic short answer grading with SBERT on out-of-sample questions," *Int. Educational Data Mining Society*, 2021.

[9] L. Jiang and N. Bosch, "Short answer scoring with GPT-4," in *Proc. 11th ACM Conf. Learning @ Scale (L@S)*, 2024.

[10] T. N. B. Duong and C. Y. Meng, "Automatic grading of short answers using large language models in software engineering courses," in *Proc. IEEE Global Engineering Education Conf. (EDUCON)*, 2024.

[11] C. Grévisse, "LLM-based automatic short answer grading in undergraduate medical education," *BMC Medical Education*, vol. 24, no. 1, p. 1060, 2024.

[12] L.-H. Chang and F. Ginter, "Automatic short answer grading for Finnish with ChatGPT," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 38, no. 21, 2024.

[13] S. Sonkar *et al.*, "Automated long answer grading with RiceChem dataset," in *Proc. Int. Conf. Artificial Intelligence in Education (AIED)*. Cham: Springer, 2024.

[14] Y. Chu *et al.*, "A LLM-powered automatic grading framework with human-level guidelines optimization," *arXiv preprint arXiv:2410.02165*, 2024.

[15] M. O. Dzikovska, R. Nielsen, C. Brew, C. Leacock, D. Giampiccolo, L. Bentivogli *et al.*, "SemEval-2013 Task 7: The joint student response analysis and 8th recognizing textual entailment challenge," in *Proc. 7th Int. Workshop on Semantic Evaluation (SemEval)*, Atlanta, GA, USA, 2013, pp. 263–274.
