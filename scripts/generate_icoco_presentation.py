"""Generate the upgraded IEEE ICOCO presentation directly reflecting IEEE_G13_ICOCO_29.8.docx.

Inherits:
- Exact paper title, terminology, equations, weights, and constraints
- Extracted official figures: image1.png (Conventional ASAG), image2.png (HybridASAG 3-Phase Architecture), image3.png (Aggregation Process)
- Official quantitative tables: Table II-A, Table III (Per-Dataset), Table IV (Qualitative Case Study)
- Live UI demonstration
- Academic presentation best practices (Action titles, ghost deck, clear visual hierarchy, speaker notes)
"""
import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

TEMPLATE_PATH = "ICOCO2026-presentation-template.pptx"
OUTPUT_PATH = "ICOCO2026_HybridASAG_Presentation.pptx"

# Extracted media from docx
IMG_CONVENTIONAL = "extracted_doc_media/word/media/image1.png"
IMG_ARCHITECTURE = "extracted_doc_media/word/media/image2.png"
IMG_AGGREGATION = "extracted_doc_media/word/media/image3.png"
IMG_UI_DASHBOARD = "docs/grading_ui_screenshot.png"
IMG_ABLATION_FIG = "evaluation/evaluation_results/figures/fig11_ablation_significance.png"

# Color Palette (Matching IEEE Theme: Navy Blue, IEEE Blue, Amber, Slate Dark)
COLOR_NAVY = RGBColor(20, 33, 61)       # #14213D
COLOR_IEEE_BLUE = RGBColor(27, 93, 170) # #1B5DAA
COLOR_AMBER = RGBColor(243, 146, 0)     # #F39200
COLOR_DARK_TEXT = RGBColor(30, 41, 59)  # #1E293B
COLOR_MUTED = RGBColor(107, 122, 144)   # #6B7A90
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BG = RGBColor(243, 246, 251)
COLOR_CARD_BORDER = RGBColor(203, 219, 237)
COLOR_GREEN = RGBColor(34, 139, 34)
COLOR_ALERT_RED = RGBColor(180, 40, 40)

FONT_FAMILY = "Segoe UI"

def clear_existing_slides(prs):
    for i in range(len(prs.slides) - 1, -1, -1):
        slide = prs.slides[i]
        for sldId in list(prs.slides._sldIdLst):
            if sldId.id == slide.slide_id:
                prs.part.drop_rel(sldId.rId)
                prs.slides._sldIdLst.remove(sldId)
                break

def format_title(slide, action_title, font_size=23, color=COLOR_NAVY):
    try:
        title_ph = slide.placeholders[0]
        title_ph.text = action_title
        tf = title_ph.text_frame
        tf.word_wrap = True
        for p in tf.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.size = Pt(font_size)
            p.font.bold = True
            p.font.color.rgb = color
            p.alignment = PP_ALIGN.LEFT
    except KeyError:
        pass

def format_footer_and_number(slide, slide_num, total_slides=13):
    for ph in slide.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 11:
            ph.text = "IEEE ICOCO 2026  ·  Deconstructing ASAG (HybridASAG)"
            for p in ph.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(10)
                p.font.color.rgb = COLOR_MUTED
        elif idx == 12:
            ph.text = f"{slide_num} / {total_slides}"
            for p in ph.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(10)
                p.font.color.rgb = COLOR_MUTED

def set_speaker_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def add_card_bullet(text_frame, title, desc, title_color=COLOR_IEEE_BLUE, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=12, desc_size=11.5, space_after=6):
    p = text_frame.add_paragraph()
    p.space_after = Pt(space_after)
    r1 = p.add_run()
    r1.text = prefix + title
    r1.font.name = FONT_FAMILY
    r1.font.bold = True
    r1.font.size = Pt(title_size)
    r1.font.color.rgb = title_color
    
    r2 = p.add_run()
    r2.text = desc
    r2.font.name = FONT_FAMILY
    r2.font.bold = False
    r2.font.size = Pt(desc_size)
    r2.font.color.rgb = text_color
    return p

def main():
    prs = Presentation(TEMPLATE_PATH)
    clear_existing_slides(prs)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide (Layout 0)
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    title_ph = slide1.placeholders[0]
    title_ph.text = "Deconstructing ASAG: Combining Deterministic NLP Tags\nwith LLM Synthesis for Interpretable Grading"
    for p in title_ph.text_frame.paragraphs:
        p.font.name = FONT_FAMILY
        p.font.size = Pt(27)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    sub_ph = slide1.placeholders[1]
    sub_ph.text = (
        "2026 IEEE International Conference on Computing (ICOCO)\n\n"
        "Authors: Do Thanh Hai¹ · Research Team²\n"
        "¹Department of Computer Science & Engineering\n"
        "Keywords: ASAG · Deterministic Linguistic Indicators · LLMs · Interpretable Assessment"
    )
    for p in sub_ph.text_frame.paragraphs:
        p.font.name = FONT_FAMILY
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(220, 235, 252)

    format_footer_and_number(slide1, 1)
    set_speaker_notes(slide1, (
        "[Timing: 0:00 - 1:00]\n"
        "Good morning, session chairs and fellow researchers. Today I present our paper: "
        "'Deconstructing ASAG: Combining Deterministic NLP Tags with LLM Synthesis for Interpretable Grading'.\n"
        "Short answer grading is critical for STEM and computing education. But today's AI graders face a major dilemma: "
        "traditional deep models are opaque black boxes, while modern LLMs hallucinate grades and reward verbosity.\n"
        "In this work, we propose HybridASAG: deconstructing the grading process into 5 deterministic NLP tags, "
        "coupled with LLM synthesis strictly for pedagogical feedback."
    ))

    # -------------------------------------------------------------
    # SLIDE 2: Background & Motivation (Layout 5: Title Only + Cards)
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide2, "Automated Grading Demands Criterion-Level Evidence, Not Just Holistic Predictions")
    format_footer_and_number(slide2, 2)

    col_w = Inches(3.6)
    col_gap = Inches(0.35)
    top_pos = Inches(1.9)
    card_h = Inches(4.7)

    # Card 1: Educational Context
    c1 = add_card(slide2, Inches(0.92), top_pos, col_w, card_h)
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Scalability in STEM & CS"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_IEEE_BLUE

    bullets1 = [
        "Short-answer questions assess deep conceptual synthesis and problem solving far better than multiple choice.",
        "Massive course enrollments (e.g., Computer Science Data Structures, basic Physics) overwhelm human instructor grading capacity.",
        "Human evaluation suffers from intra-rater inconsistency, fatigue, and turnaround delays."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    # Card 2: The Evidence Gap
    c2 = add_card(slide2, Inches(0.92) + col_w + col_gap, top_pos, col_w, card_h)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. The 'Criterion Evidence' Gap"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_IEEE_BLUE

    bullets2 = [
        "Current ASAG models primarily optimize holistic accuracy (e.g., predicting an aggregate score 7.5/10).",
        "They fail to provide criterion-level evidence: Was the penalty due to missing keywords, flawed logic, or grammatical phrasing?",
        "Lack of granular evidence prevents formative learning and hinders institutional adoption."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    # Card 3: The Interpretability Mandate
    c3 = add_card(slide2, Inches(0.92) + (col_w + col_gap) * 2, top_pos, col_w, card_h)
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Trust & Auditability"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_IEEE_BLUE

    bullets3 = [
        "High-stakes exam grading requires auditable decisions that students and instructors can dispute and inspect.",
        "Post-hoc explanations (e.g., SHAP, attention heatmaps) do not map directly to pedagogical grading rubrics.",
        "Deterministic scoring guarantees that identical submissions receive identical grades under repeat evaluation."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    set_speaker_notes(slide2, (
        "[Timing: 1:00 - 2:00]\n"
        "To motivate our work: ASAG systems are expanding across universities worldwide. "
        "However, as highlighted in Section I of our paper, existing systems prioritize a single numeric score over explainability.\n"
        "Students are left asking: 'Why did I get this grade?' Was it bad grammar? A missing key concept? Or invalid reasoning?\n"
        "Without criterion-level evidence, AI assessment cannot be trusted in formative or summative education."
    ))

    # -------------------------------------------------------------
    # SLIDE 3: Problem: Conventional Black-Box vs Prompted LLMs (with Fig 1 from docx)
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide3, "Conventional End-to-End Models and Prompted LLMs Fail the Interpretability Test")
    format_footer_and_number(slide3, 3)

    # Embed Figure 1 from docx (Conventional End-to-End ASAG Pipeline)
    # image1 aspect ratio = 4.84. width=9.8 inches -> height=2.02 inches.
    if os.path.exists(IMG_CONVENTIONAL):
        slide3.shapes.add_picture(IMG_CONVENTIONAL, Inches(1.76), Inches(1.68), width=Inches(9.8))

    # Two comparison cards below Figure 1 (top=3.95, height=2.90)
    c_left = add_card(slide3, Inches(0.92), Inches(3.92), Inches(5.6), Inches(2.92),
                      bg_color=RGBColor(254, 246, 246), border_color=COLOR_ALERT_RED)
    tl = c_left.text_frame
    tl.word_wrap = True
    p = tl.paragraphs[0]
    p.text = "Conventional Deep Learning (Fig. 1)"
    p.font.name = FONT_FAMILY
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_ALERT_RED

    deep_pts = [
        ("Latent Space Compression: ", "Encodes Reference and Student answers into hidden vectors before predicting a scalar score."),
        ("Opaque Black Box: ", "No explicit intermediate linguistic variables; impossible to verify which rubric criterion was violated."),
        ("No Formative Feedback: ", "Cannot generate actionable suggestions for formative student improvement.")
    ]
    for title, desc in deep_pts:
        add_card_bullet(tl, title, desc, title_color=COLOR_ALERT_RED, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=11.5, desc_size=11, space_after=4)

    c_right = add_card(slide3, Inches(6.8), Inches(3.92), Inches(5.6), Inches(2.92),
                       bg_color=RGBColor(255, 250, 242), border_color=COLOR_AMBER)
    tr = c_right.text_frame
    tr.word_wrap = True
    p = tr.paragraphs[0]
    p.text = "Prompted Generative LLMs (e.g., GPT-4)"
    p.font.name = FONT_FAMILY
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_AMBER

    llm_pts = [
        ("Length & Eloquence Bias: ", "Consistently assigns high marks (90-100%) to verbose, fluent answers containing factual errors!"),
        ("Stochastic Score Drift: ", "Non-zero temperature or minor prompt formatting causes identical student answers to get different grades."),
        ("Post-Hoc Hallucinations: ", "10-20% hallucination rate in justifications, referencing phantom rubric points or invented statements.")
    ]
    for title, desc in llm_pts:
        add_card_bullet(tr, title, desc, title_color=COLOR_AMBER, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=11.5, desc_size=11, space_after=4)

    set_speaker_notes(slide3, (
        "[Timing: 2:00 - 3:15]\n"
        "In Section I & II, we formalize the fundamental limitation of existing paradigms.\n"
        "As shown in Figure 1 taken from our paper: conventional models collapse student text into a latent representation. "
        "It's completely opaque.\n"
        "On the other hand, purely prompted LLMs swing to the opposite extreme: they produce fluent text, but they hallucinate! "
        "They are easily fooled by superficial fluency—giving high scores to answers that sound good but are factually wrong.\n"
        "This motivates our HybridASAG framework: we must separate deterministic scoring from linguistic synthesis."
    ))

    # -------------------------------------------------------------
    # SLIDE 4: Proposed HybridASAG Architecture (with Fig 3 from docx)
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide4, "HybridASAG: Deconstructing Grading into 3 Transparent, Decoupled Phases")
    format_footer_and_number(slide4, 4)

    # Subtitle highlight pill right below the title
    sub_pill = add_card(slide4, Inches(0.92), Inches(1.58), Inches(11.5), Inches(0.42),
                        bg_color=RGBColor(240, 246, 254), border_color=COLOR_IEEE_BLUE)
    sptf = sub_pill.text_frame
    sptf.word_wrap = True
    p_sub = sptf.paragraphs[0]
    p_sub.text = "Figure 3 from Paper: Decoupled Workflow Separating Deterministic Scoring (Phases 1 & 2) from LLM Synthesis (Phase 3)"
    p_sub.font.name = FONT_FAMILY
    p_sub.font.bold = True
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = COLOR_IEEE_BLUE
    p_sub.alignment = PP_ALIGN.CENTER

    # Embed Figure 3 from docx (Overall architecture of HybridASAG)
    # image2 aspect ratio = 1.99. width=10.2 -> height=5.12.
    if os.path.exists(IMG_ARCHITECTURE):
        slide4.shapes.add_picture(IMG_ARCHITECTURE, Inches(1.56), Inches(2.08), width=Inches(10.2))

    set_speaker_notes(slide4, (
        "[Timing: 3:15 - 4:30]\n"
        "Here is the core technical architecture of HybridASAG, presented in Figure 3 of our paper.\n"
        "Notice the three distinct phases:\n"
        "In Phase 1, we extract a 5-dimensional feature vector using specialized expert models.\n"
        "In Phase 2, the final grade is calculated using a transparent weighted aggregation formula. "
        "The model cannot invent or hallucinate the score.\n"
        "In Phase 3, Qwen2.5-3B acts purely as a synthesizer: it reads the 5 tags and writes pedagogical teacher feedback. "
        "This completely eliminates grade hallucinations."
    ))


    # -------------------------------------------------------------
    # SLIDE 5: Phase 1: Five Deterministic NLP Tags (Structured Table & Equations)
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide5, "Phase 1: Five Deterministic NLP Indicators Grounded in Linguistic Theory")
    format_footer_and_number(slide5, 5)

    # Table of 5 indicators from Section III-B
    tb_shape = slide5.shapes.add_table(6, 5, Inches(0.92), Inches(1.9), Inches(11.5), Inches(4.8))
    tb = tb_shape.table
    tb.columns[0].width = Inches(2.2) # Indicator
    tb.columns[1].width = Inches(2.8) # Underlying Model
    tb.columns[2].width = Inches(1.2) # Size
    tb.columns[3].width = Inches(3.3) # Mathematical Formulation
    tb.columns[4].width = Inches(2.0) # Pedagogical Metric

    headers = ["Linguistic Tag", "Specialized Model", "Size", "Formulation & Mapping", "Target Competence"]
    for ci, h in enumerate(headers):
        cell = tb.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    data_tags = [
        ("1. Semantic Similarity (s_sem)", "princeton-nlp/sup-simcse-roberta-large", "~1.3 GB", "s_sem = (cos(e_R, e_S) + 1) / 2\nDense sentence contrastive embedding", "Paraphrastic meaning preservation"),
        ("2. Keyword Coverage (s_cov)", "all-MiniLM-L6-v2 (KeyBERT backend)", "~80 MB", "s_cov = |{k in K | cos(e_k, e_S) >= 0.5}| / N\nThreshold theta_k = 0.5", "Core domain terminology recall"),
        ("3. Grammatical Quality (s_gra)", "textattack/roberta-base-CoLA", "~500 MB", "s_gra = P_acceptable(S)\nCorpus of Linguistic Acceptability", "Syntactic acceptability & fluency"),
        ("4. Writing Formality (s_for)", "cointegrated/roberta-base-formality", "~500 MB", "s_for = P_formal(S)\nGYAFC formality corpus register", "Scholarly tone vs. colloquial slang"),
        ("5. Logical Consistency (s_log)", "cross-encoder/nli-deberta-v3-base", "~700 MB", "s_log = (P_entail - P_contra + 1) / 2\nPremise R -> Hypothesis S", "Entailment vs contradiction detection")
    ]

    for ri, row in enumerate(data_tags, start=1):
        bg = RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_DARK_TEXT
                if ci == 0:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_IEEE_BLUE
                elif ci in (2,):
                    p.alignment = PP_ALIGN.CENTER

    set_speaker_notes(slide5, (
        "[Timing: 4:30 - 5:30]\n"
        "Section III-B details each linguistic indicator:\n"
        "1. Semantic Similarity uses SimCSE to map sentence embeddings, scoring paraphrases fairly.\n"
        "2. Keyword Coverage uses KeyBERT with an empirical threshold of 0.5 to check essential concepts.\n"
        "3. Grammar uses RoBERTa trained on CoLA for linguistic acceptability.\n"
        "4. Formality detects informal registers.\n"
        "5. Most importantly, Logical Consistency uses DeBERTa-v3 NLI cross-encoder to compute P(entailment) and P(contradiction). "
        "Together, these 5 tags total under 3GB of memory and execute in parallel."
    ))

    # -------------------------------------------------------------
    # SLIDE 6: Phase 2: Transparent Score Aggregation (with Fig from docx + learned weights)
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide6, "Phase 2: Transparent Score Aggregation with NLI Contradiction Hard-Cap")
    format_footer_and_number(slide6, 6)

    # Embed Figure from docx (Score aggregation process)
    # image3 aspect ratio = 1.83. width=5.8 -> height=3.17
    if os.path.exists(IMG_AGGREGATION):
        slide6.shapes.add_picture(IMG_AGGREGATION, Inches(0.92), Inches(1.85), width=Inches(5.8))

    # Summary card under aggregation image
    ag_summary = add_card(slide6, Inches(0.92), Inches(5.12), Inches(5.8), Inches(1.68),
                          bg_color=RGBColor(240, 246, 254), border_color=COLOR_IEEE_BLUE)
    astf = ag_summary.text_frame
    astf.word_wrap = True
    p_as = astf.paragraphs[0]
    p_as.text = "Score Aggregation Formula (Section III-C)"
    p_as.font.name = FONT_FAMILY
    p_as.font.bold = True
    p_as.font.size = Pt(13)
    p_as.font.color.rgb = COLOR_IEEE_BLUE

    add_card_bullet(astf, "Feature Vector: ", "[s_sem, s_cov, s_for, s_gra, s_log] ∈ [0, 1]^5",
                    title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=11, desc_size=10.5, space_after=2)
    add_card_bullet(astf, "Aggregate Score: ", "Final Score = 10 · ∑(w_i · s_i) ∈ [0, 10]",
                    title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=11, desc_size=10.5, space_after=2)
    add_card_bullet(astf, "Deterministic: ", "Zero temperature drift; identical answers receive identical grades.",
                    title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=11, desc_size=10.5, space_after=2)

    # Right side: Mathematical details & constraints
    r_card = add_card(slide6, Inches(7.0), Inches(1.85), Inches(5.4), Inches(4.95))
    rtf = r_card.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "Mathematical Formulation (Eq. 16-18)"
    p.font.name = FONT_FAMILY
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_NAVY

    eq_pts = [
        ("Linear Weighted Formulation: ", "y = 10 · ∑(w_i · s_i) with simplex constraint: ∑ w_i = 1, w_i ≥ 0."),
        ("Supervised Optimal Weights (w*): ", "w = [w_sem: 0.15, w_cov: 0.30, w_gra: 0.10, w_for: 0.05, w_log: 0.40] fitted via linear regressors on training splits."),
        ("Key Finding: Logic & Coverage Dominate: ", "Logical consistency (40%) and Keyword recall (30%) account for 70% of total score variance."),
        ("Strict NLI Hard-Cap Constraint: ", "If P_contra(R, S) > 0.90, final grade is HARD-CAPPED at 4.0 / 10 (40/100) regardless of high semantic or grammar scores!"),
        ("Guaranteed Determinism: ", "Same student input always yields the exact same numeric grade—100% auditable.")
    ]
    for title, desc in eq_pts:
        add_card_bullet(rtf, title, desc,
                        title_color=COLOR_ALERT_RED if "Hard-Cap" in title else COLOR_IEEE_BLUE,
                        text_color=COLOR_DARK_TEXT, prefix="✔ ", title_size=11.5, desc_size=10.5, space_after=4)

    set_speaker_notes(slide6, (
        "[Timing: 5:30 - 6:30]\n"
        "Section III-C describes our aggregation mechanism.\n"
        "Our weight vector was derived empirically on training data: Logic is assigned 40%, Coverage 30%, Semantic 15%, Grammar 10%, and Formality 5%.\n"
        "Look at the red rule: our NLI Contradiction Hard-Cap. "
        "If the contradiction probability exceeds 0.90, the grade is capped at 4.0 out of 10. "
        "This prevents a critical failure mode where a student writes a beautiful, grammatically flawless paragraph that says the exact opposite of the truth!"
    ))

    # -------------------------------------------------------------
    # SLIDE 7: Experimental Setup & Datasets (SemEval + Mohler)
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide7, "Experimental Setup: Multi-Domain Benchmarks and Dual Quality Metrics")
    format_footer_and_number(slide7, 7)

    # Left Card: Datasets
    c_left = add_card(slide7, Inches(0.92), Inches(1.85), Inches(5.6), Inches(4.9))
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    p0 = tf_l.paragraphs[0]
    p0.text = "Benchmark Datasets (Section IV-A)"
    p0.font.name = FONT_FAMILY
    p0.font.bold = True
    p0.font.size = Pt(15)
    p0.font.color.rgb = COLOR_IEEE_BLUE

    dsets = [
        ("SemEval-2013 Task 7: ", "Golden ASAG benchmark comprising two distinct scientific domains:"),
        ("Beetle Domain: ", "~4,000 student answers in basic electricity and electronics (Unseen Answers & Questions splits)."),
        ("SciEntsBank Domain: ", ">10,000 student answers across physics, biology, and chemistry (broad scientific evaluation)."),
        ("Mohler CS Benchmark: ", "2,442 student answers across 87 questions from college Data Structures & Algorithms (normalized to [0, 10])."),
        ("Rigorous Test Splits: ", "Evaluated on Beetle Unseen Answers (UA), Beetle Unseen Questions (UQ), SciEntsBank Test, and Mohler 5-Fold Cross-Validation.")
    ]
    for title, desc in dsets:
        add_card_bullet(tf_l, title, desc, title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="• ", title_size=12, desc_size=11, space_after=6)

    # Right Card: Evaluation Criteria
    c_right = add_card(slide7, Inches(6.8), Inches(1.85), Inches(5.6), Inches(4.9))
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    p0_r = tf_r.paragraphs[0]
    p0_r.text = "Evaluation Criteria (Section IV-B)"
    p0_r.font.name = FONT_FAMILY
    p0_r.font.bold = True
    p0_r.font.size = Pt(15)
    p0_r.font.color.rgb = COLOR_IEEE_BLUE

    eval_metrics = [
        ("Quadratic Weighted Kappa (QWK): ", "Standard conference agreement metric; quadratically penalizes severe grade misclassifications (Correct vs Contradictory)."),
        ("Accuracy & Macro-F1: ", "Measures balanced classification across score categories."),
        ("JSON Compliance Rate: ", "Measures structural output reliability for automated LMS integration."),
        ("Hallucination Rate: ", "Identifies unsupported factual claims in feedback using NLI filtering followed by author verification."),
        ("Hardware Setup: ", "Local NVIDIA GPU workstation with CUDA acceleration; Qwen2.5-3B quantized to 4-bit NF4 (~450ms CPU / ~120ms GPU).")
    ]
    for title, desc in eval_metrics:
        add_card_bullet(tf_r, title, desc, title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="✔ ", title_size=12, desc_size=11, space_after=6)

    set_speaker_notes(slide7, (
        "[Timing: 6:30 - 7:30]\n"
        "In Section IV, we benchmark across two premier ASAG datasets:\n"
        "SemEval-2013 Task 7 (covering electricity and broader science) and the Mohler dataset (computer science data structures).\n"
        "We evaluate not only classical grading agreement (QWK, Accuracy, Macro F1), but also output reliability: "
        "JSON compliance rate and Hallucination rate in generated feedback."
    ))

    # -------------------------------------------------------------
    # SLIDE 8: Table II-A & II-B: Quantitative Performance
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide8, "Quantitative Results: HybridASAG Outperforms Base LLMs with 85.5% QWK")
    format_footer_and_number(slide8, 8)

    # Table II-A from docx
    t_shape = slide8.shapes.add_table(5, 6, Inches(0.92), Inches(1.9), Inches(11.5), Inches(2.7))
    t = t_shape.table
    t.columns[0].width = Inches(3.0) # Model
    t.columns[1].width = Inches(1.7) # Accuracy
    t.columns[2].width = Inches(1.7) # Macro F1
    t.columns[3].width = Inches(1.7) # QWK
    t.columns[4].width = Inches(1.7) # JSON Compl.
    t.columns[5].width = Inches(1.7) # Hallucination

    h_titles = ["Model Architecture", "Accuracy", "Macro F1", "QWK", "JSON Compl.", "Hallucination"]
    for ci, h in enumerate(h_titles):
        cell = t.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    results_table = [
        ("Qwen2.5-1.5B (Direct Prompt)", "68.42%", "65.20%", "60.15%", "85.30%", "18.50%"),
        ("Qwen2.5-3B (Direct Prompt)", "75.85%", "75.50%", "72.40%", "92.47%", "10.73%"),
        ("Hybrid (-Logic)", "78.10%", "77.54%", "75.20%", "98.85%", "4.15%"),
        ("HybridASAG (Full 5+1 Proposed)", "83.50%", "82.90%", "85.50%", "99.12%", "1.24%")
    ]

    for ri, r in enumerate(results_table, start=1):
        is_hybrid = ri == 4
        bg = RGBColor(230, 245, 235) if is_hybrid else (RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE)
        for ci, val in enumerate(r):
            cell = t.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(12)
                if is_hybrid:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_GREEN if ci in (1, 2, 3, 4, 5) else COLOR_NAVY
                else:
                    p.font.color.rgb = COLOR_DARK_TEXT
                if ci > 0:
                    p.alignment = PP_ALIGN.CENTER

    # Findings Box
    f_card = add_card(slide8, Inches(0.92), Inches(4.8), Inches(11.5), Inches(1.9))
    ftf = f_card.text_frame
    ftf.word_wrap = True
    p = ftf.paragraphs[0]
    p.text = "Key Analytical Takeaways (TABLE II-A & II-B):"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_IEEE_BLUE

    res_pts = [
        ("Substantial Agreement Boost: ", "HybridASAG achieves 85.50% QWK, outperforming direct Qwen2.5-3B prompting by +13.10% QWK and +7.65% Accuracy."),
        ("Virtually Zero Hallucination: ", "Constraining the LLM with deterministic tags slashes Hallucination Rate from 10.73% down to only 1.24%."),
        ("99.12% Format Compliance: ", "Decoupled pipeline ensures stable, parseable JSON output, rendering the system enterprise-ready for automated exam workflows.")
    ]
    for title, desc in res_pts:
        add_card_bullet(ftf, title, desc, title_color=COLOR_IEEE_BLUE, text_color=COLOR_DARK_TEXT, prefix="✔ ", title_size=12, desc_size=11, space_after=4)

    set_speaker_notes(slide8, (
        "[Timing: 7:30 - 8:45]\n"
        "Here are the quantitative results from Table II-A and II-B in our paper.\n"
        "Comparing HybridASAG to direct LLM prompting:\n"
        "Qwen2.5-3B prompted directly achieves 72.4% QWK with a 10.7% hallucination rate.\n"
        "HybridASAG achieves 85.5% QWK—a 13.1% jump—while cutting hallucinations by nearly 90%, down to 1.24%.\n"
        "JSON compliance also hits 99.12%, ensuring seamless integration into learning management systems."
    ))

    # -------------------------------------------------------------
    # SLIDE 9: Table III: Per-Dataset Cross-Domain Performance
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide9, "Cross-Domain Generalization Across Electronics, Science, and Computer Science")
    format_footer_and_number(slide9, 9)

    # Table III from docx
    t3_shape = slide9.shapes.add_table(5, 5, Inches(0.92), Inches(1.9), Inches(11.5), Inches(2.7))
    t3 = t3_shape.table
    t3.columns[0].width = Inches(3.5) # Dataset
    t3.columns[1].width = Inches(2.0) # Test Samples
    t3.columns[2].width = Inches(2.0) # Accuracy
    t3.columns[3].width = Inches(2.0) # Macro F1
    t3.columns[4].width = Inches(2.0) # QWK

    h3_titles = ["Benchmark Dataset (Domain)", "Test Samples", "Accuracy", "Macro F1", "QWK"]
    for ci, h in enumerate(h3_titles):
        cell = t3.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    dataset_rows = [
        ("Beetle — Unseen Answers (Electronics)", "~700", "84.20%", "83.50%", "86.10%"),
        ("Beetle — Unseen Questions (Electronics)", "~700", "82.80%", "82.10%", "84.70%"),
        ("SciEntsBank — Test (General Science)", "~3,500", "83.90%", "83.20%", "85.80%"),
        ("Mohler — 5-Fold Cross Validation (CS Data Structures)", "2,442", "82.60 ± 2.1%", "81.90 ± 2.4%", "84.50 ± 2.8%")
    ]

    for ri, r in enumerate(dataset_rows, start=1):
        bg = RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE
        for ci, val in enumerate(r):
            cell = t3.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(12)
                if ci == 0:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_IEEE_BLUE
                else:
                    p.alignment = PP_ALIGN.CENTER

    # Robustness takeaways card
    r_card = add_card(slide9, Inches(0.92), Inches(4.8), Inches(11.5), Inches(1.9))
    rtf = r_card.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "Generalization & Out-of-Domain Robustness (Section IV-D):"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_IEEE_BLUE

    rob_pts = [
        ("Robust to Unseen Questions: ", "On Beetle Unseen Questions (UQ), QWK maintains a solid 84.70%, proving the model does not overfit to seen prompt structures."),
        ("Large-Scale Science Validation: ", "Across 3,500 test questions in SciEntsBank, performance remains consistent at 85.80% QWK and 83.90% Accuracy."),
        ("Multi-Annotator CS Agreement: ", "On Mohler (2,442 university CS answers), 5-fold CV achieves 84.50 ± 2.8% QWK, showing exceptional alignment with human professors.")
    ]
    for title, desc in rob_pts:
        add_card_bullet(rtf, title, desc, title_color=COLOR_IEEE_BLUE, text_color=COLOR_DARK_TEXT, prefix="✔ ", title_size=12, desc_size=11, space_after=4)

    set_speaker_notes(slide9, (
        "[Timing: 8:45 - 9:45]\n"
        "Table III highlights our model's cross-domain robustness.\n"
        "Notice that on Beetle Unseen Questions—where the system is evaluated on prompts it has never encountered during training—it retains 84.7% QWK.\n"
        "And on the Mohler Computer Science dataset across 2,442 samples, 5-fold cross-validation achieves 84.5% QWK. "
        "This confirms that our 5 deterministic linguistic tags transfer reliably across diverse STEM and CS disciplines."
    ))

    # -------------------------------------------------------------
    # SLIDE 10: Ablation Study & Component Contributions (Section IV-E)
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide10, "Ablation Study: NLI Logic is the Primary Linchpin Against Hallucinatory Grades")
    format_footer_and_number(slide10, 10)

    # Insert Fig 11 if available
    if os.path.exists(IMG_ABLATION_FIG):
        slide10.shapes.add_picture(IMG_ABLATION_FIG, Inches(0.92), Inches(1.85), width=Inches(6.8))

    # Right side: Analytical takeaways from Section IV-E
    rc = add_card(slide10, Inches(7.9), Inches(1.85), Inches(4.5), Inches(4.8))
    rtf = rc.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "Ablation Findings (Section IV-E)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY

    abl_data = [
        ("Removing Logic Module (-Logic): ", "Causes a catastrophic drop in QWK from 85.50% down to 75.20% (Δ = -10.30%)! Accuracy falls from 83.50% to 78.10%."),
        ("Hallucination Triples Without Logic: ", "Without DeBERTa NLI, Hallucination Rate spikes from 1.24% to 4.15%. Logic is essential to penalize factual inversions."),
        ("Semantic Embeddings Anchor Baseline: ", "SimCSE drives initial paraphrase recognition (w_sem = 0.15); removing it impairs synonym recall."),
        ("Grammar & Formality are Non-Destructive: ", "Grammar (10%) and Formality (5%) contribute fine-grained calibration without unfairly penalizing ESL students.")
    ]
    for title, desc in abl_data:
        p = rtf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_ALERT_RED if "Logic" in title else COLOR_IEEE_BLUE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    set_speaker_notes(slide10, (
        "[Timing: 9:45 - 10:45]\n"
        "Section IV-E reports our ablation study, answering: Which component is most essential?\n"
        "Look at the impact of removing the Logic module:\n"
        "QWK plunges from 85.5% to 75.2%—a massive 10.3% loss.\n"
        "Even worse: Hallucination rate more than triples, from 1.24% to 4.15%.\n"
        "Why? Because when students write erroneous claims that share vocabulary with the question, "
        "only DeBERTa NLI recognizes the logical contradiction. Without logic, automated grading is blind to factual errors."
    ))

    # -------------------------------------------------------------
    # SLIDE 11: Qualitative Case Study: Catching What LLMs Miss (TABLE IV from docx!)
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide11, "Qualitative Case Study: Catching Factual Errors that Fool Base LLMs")
    format_footer_and_number(slide11, 11)

    # 2 Cards for the 2 cases in Table IV of docx
    w_card = Inches(5.6)
    h_card = Inches(4.8)

    # Case 1: Stack vs Queue (CS)
    c_cs = add_card(slide11, Inches(0.92), Inches(1.9), w_card, h_card)
    tf_cs = c_cs.text_frame
    tf_cs.word_wrap = True
    p = tf_cs.paragraphs[0]
    p.text = "Case 1: Computer Science Domain"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_IEEE_BLUE

    cs_content = [
        ("Question: ", "Explain the main difference between a Stack and a Queue."),
        ("Reference: ", "Stack = LIFO (Last-In-First-Out); Queue = FIFO (First-In-First-Out)."),
        ("Student Answer: ", "\"A Stack uses LIFO, meaning the first item added is the first one removed. A Queue uses FIFO.\" [INVERTED LOGIC!]"),
        ("Base LLM (Qwen2.5-3B): ", "[Grade: 100/100] 'Your answer is perfect. You correctly identified that Stack uses LIFO and Queue uses FIFO.' [CRITICAL FAILURE!]"),
        ("HybridASAG (Proposed): ", "[Grade: 50/100] 'Partially Correct. You correctly named LIFO and FIFO. However, your definition of LIFO is a logical error: the last item added is removed first.'")
    ]
    for label, text in cs_content:
        p = tf_cs.add_paragraph()
        r1 = p.add_run()
        r1.text = label
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = COLOR_ALERT_RED if "Base LLM" in label else (COLOR_GREEN if "Hybrid" in label else COLOR_NAVY)
        r2 = p.add_run()
        r2.text = text
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(4)

    # Case 2: Red Blood Cells (STEM)
    c_bio = add_card(slide11, Inches(6.8), Inches(1.9), w_card, h_card)
    tf_bio = c_bio.text_frame
    tf_bio.word_wrap = True
    p = tf_bio.paragraphs[0]
    p.text = "Case 2: Biology / STEM Domain"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_IEEE_BLUE

    bio_content = [
        ("Question: ", "What is the primary function of red blood cells?"),
        ("Reference: ", "Carry oxygen from lungs to the body and bring carbon dioxide back."),
        ("Student Answer: ", "\"Red blood cells carry oxygen around the body and help fight infections.\" [FACTUAL ERROR!]"),
        ("Base LLM (Qwen2.5-3B): ", "[Grade: 95/100] 'Excellent. You correctly stated that red blood cells carry oxygen and fight infections.' [HALLUCINATION!]"),
        ("HybridASAG (Proposed): ", "[Grade: 60/100] 'Partially Correct. While you are right about carrying oxygen, fighting infections is a factual error (white blood cells do that).'")
    ]
    for label, text in bio_content:
        p = tf_bio.add_paragraph()
        r1 = p.add_run()
        r1.text = label
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = COLOR_ALERT_RED if "Base LLM" in label else (COLOR_GREEN if "Hybrid" in label else COLOR_NAVY)
        r2 = p.add_run()
        r2.text = text
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(4)

    set_speaker_notes(slide11, (
        "[Timing: 10:45 - 12:00]\n"
        "Table IV from our paper provides this compelling qualitative demonstration.\n"
        "Look at Case 1: The student wrote that LIFO means 'the first item added is the first one removed'. "
        "That is 100% wrong!\n"
        "Yet what did the base LLM do? It awarded 100/100 and called it 'perfect' because it saw the keywords LIFO and FIFO!\n"
        "HybridASAG, by contrast, caught the contradiction via DeBERTa NLI, deducted 50 marks, "
        "and Qwen generated the exact explanation: 'LIFO means the last item added is removed first'.\n"
        "This proves why criterion-level deconstruction is essential."
    ))

    # -------------------------------------------------------------
    # SLIDE 12: Interactive Dashboard & Production System
    # -------------------------------------------------------------
    slide12 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide12, "Production Implementation: Real-Time Web Interface for Teachers and Students")
    format_footer_and_number(slide12, 12)

    # Insert Live UI Screenshot
    if os.path.exists(IMG_UI_DASHBOARD):
        slide12.shapes.add_picture(IMG_UI_DASHBOARD, Inches(0.92), Inches(1.85), width=Inches(7.2))

    # Right side: System highlights
    ui_card = add_card(slide12, Inches(8.3), Inches(1.85), Inches(4.1), Inches(4.8))
    uitf = ui_card.text_frame
    uitf.word_wrap = True
    p = uitf.paragraphs[0]
    p.text = "Operational Architecture"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY

    ui_pts = [
        ("Full-Stack Implementation: ", "FastAPI asynchronous backend with PyTorch/HuggingFace + React/Vite interactive frontend."),
        ("Sub-Second Inference: ", "Parallelized feature extractors compute 5 tags in ~450ms on CPU or ~120ms with CUDA."),
        ("Diagnostic Radar Breakdown: ", "Instructors visually inspect scores across all 5 dimensions simultaneously."),
        ("Missing Vocabulary Extraction: ", "Automatically flags missing essential keywords (e.g., 'photosynthesis', 'carbon dioxide')."),
        ("Rule-Based Fallback: ", "If LLM generation is bypassed (`skip_llm=True`), templated feedback executes in 5ms with zero VRAM.")
    ]
    for title, desc in ui_pts:
        p = uitf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_IEEE_BLUE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    set_speaker_notes(slide12, (
        "[Timing: 12:00 - 13:00]\n"
        "HybridASAG is not just a theoretical proposal; it is a fully functioning production system.\n"
        "Here is our live dashboard running on a local workstation.\n"
        "Teachers and students receive an overall score (e.g. 67/100), visual bars for all 5 dimensions, "
        "and a list of missing vocabulary.\n"
        "With 4-bit NF4 quantization, the entire framework operates on standard hardware, with a fallback engine that runs in under 5 milliseconds."
    ))


    # -------------------------------------------------------------
    # SLIDE 13: Conclusions, Limitations & Q&A Anchor (Section V)
    # -------------------------------------------------------------
    slide13 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide13, "Conclusion: A Trustworthy, Explainable, and Practical Path Forward for ASAG")
    format_footer_and_number(slide13, 13)

    # Left Card: Contributions
    c_left = add_card(slide13, Inches(0.92), Inches(1.85), Inches(5.6), Inches(3.6))
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    p0 = tf_l.paragraphs[0]
    p0.text = "Core Contributions & Findings"
    p0.font.name = FONT_FAMILY
    p0.font.bold = True
    p0.font.size = Pt(16)
    p0.font.color.rgb = COLOR_IEEE_BLUE

    contribs = [
        ("Decoupled Paradigm: ", "Separated deterministic criterion-level scoring from LLM feedback synthesis, eliminating grade hallucinations."),
        ("Empirical Validation: ", "Achieved 85.50% QWK and 83.50% Accuracy on SemEval and Mohler benchmarks, outperforming base LLMs."),
        ("Logic is Crucial: ", "Proved that NLI logic is the essential linchpin preventing factual inversion errors in automated assessment.")
    ]
    for title, desc in contribs:
        add_card_bullet(tf_l, title, desc, title_color=COLOR_NAVY, text_color=COLOR_DARK_TEXT, prefix="✔ ", title_size=12, desc_size=11, space_after=6)

    # Right Card: Future Work & Open Science
    c_right = add_card(slide13, Inches(6.8), Inches(1.85), Inches(5.6), Inches(3.6))
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    p0_r = tf_r.paragraphs[0]
    p0_r.text = "Future Work & Open Science"
    p0_r.font.name = FONT_FAMILY
    p0_r.font.bold = True
    p0_r.font.size = Pt(16)
    p0_r.font.color.rgb = COLOR_AMBER

    fw = [
        ("Cross-Lingual Generalization: ", "Extending expert encoders to Vietnamese and low-resource Southeast Asian languages."),
        ("Multiple Reference Answers: ", "Calibrating dynamic weights for questions with divergent valid solution paths."),
        ("Open-Source Code & Artifacts: ", "github.com/dothanhhxx/AutoGradeSystem"),
        ("Corresponding Author: ", "Do Thanh Hai · IEEE ICOCO 2026")
    ]
    for title, desc in fw:
        add_card_bullet(tf_r, title, desc, title_color=COLOR_NAVY, text_color=COLOR_IEEE_BLUE if "github" in desc else COLOR_DARK_TEXT, prefix="• ", title_size=12, desc_size=11, space_after=5)

    # Clean bottom card for Q&A across full width
    qa_card = add_card(slide13, Inches(0.92), Inches(5.65), Inches(11.5), Inches(1.15),
                       bg_color=RGBColor(240, 245, 252), border_color=COLOR_IEEE_BLUE)
    qtf = qa_card.text_frame
    qtf.word_wrap = True
    p_qa = qtf.paragraphs[0]
    p_qa.text = "Thank you for your attention!  ·  Q&A Session"
    p_qa.font.name = FONT_FAMILY
    p_qa.font.bold = True
    p_qa.font.size = Pt(15)
    p_qa.font.color.rgb = COLOR_IEEE_BLUE
    p_qa.alignment = PP_ALIGN.CENTER

    p_qa2 = qtf.add_paragraph()
    p_qa2.text = "We welcome questions and discussion from the session chairs and audience."
    p_qa2.font.name = FONT_FAMILY
    p_qa2.font.size = Pt(12)
    p_qa2.font.color.rgb = COLOR_NAVY
    p_qa2.alignment = PP_ALIGN.CENTER


    set_speaker_notes(slide13, (
        "[Timing: 13:00 - 15:00+ (Q&A Session Anchor)]\n"
        "Key Point: Leave this slide on screen during the 2-3 minute Q&A session.\n"
        "Anticipated Q&A Defenses:\n"
        "1. Q: Why not fine-tune a single 7B model instead of 5 separate models?\n"
        "   A: A single fine-tuned model remains a black box. You cannot audit criterion weights, and you cannot enforce hard constraints like our NLI contradiction cap.\n"
        "2. Q: How do you handle multiple valid reference answers?\n"
        "   A: In Section V, we discuss our ongoing extension: taking the maximum semantic score across alternative reference formulations.\n"
        "3. Q: How fast is inference?\n"
        "   A: The 5 expert models run concurrently in ~120ms on GPU or ~450ms on CPU. Pure LLM generation takes 5-15 seconds."
    ))

    prs.save(OUTPUT_PATH)
    print(f"Successfully generated updated {OUTPUT_PATH} with {len(prs.slides)} slides.")

if __name__ == "__main__":
    main()
