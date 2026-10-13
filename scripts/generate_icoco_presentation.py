"""Generate the complete 13-slide academic presentation for IEEE ICOCO 2026.

Uses ICOCO2026-presentation-template.pptx as the master template.
Applies academic presentation standards (Action titles, ghost deck test,
proper layout mapping, figures, tables, placeholder markers, speaker notes).
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Paths
TEMPLATE_PATH = "ICOCO2026-presentation-template.pptx"
OUTPUT_PATH = "ICOCO2026_HybridASAG_Presentation.pptx"
FIG1_PATH = "evaluation/evaluation_results/figures/fig1_performance_comparison.png"
FIG11_PATH = "evaluation/evaluation_results/figures/fig11_ablation_significance.png"
UI_SCREENSHOT_PATH = "docs/grading_ui_screenshot.png"

# Color Palette (Matching IEEE ICOCO Theme)
COLOR_NAVY = RGBColor(20, 33, 61)       # #14213D - Dark navy for primary headers
COLOR_IEEE_BLUE = RGBColor(27, 93, 170) # #1B5DAA - Theme accent blue
COLOR_AMBER = RGBColor(243, 146, 0)     # #F39200 - IEEE Accent amber
COLOR_DARK_TEXT = RGBColor(30, 41, 59)  # #1E293B - Slate dark body
COLOR_MUTED = RGBColor(107, 122, 144)   # #6B7A90 - Muted gray citations
COLOR_WHITE = RGBColor(255, 255, 255)   # #FFFFFF
COLOR_CARD_BG = RGBColor(243, 246, 251) # #F3F6FB - Soft container bg
COLOR_CARD_BORDER = RGBColor(203, 219, 237) # #CBDBED - Card border
COLOR_GREEN = RGBColor(34, 139, 34)     # #228B22 - Success green

FONT_FAMILY = "Segoe UI"

def clear_existing_slides(prs):
    """Cleanly remove existing slides without corrupting relationships."""
    for i in range(len(prs.slides) - 1, -1, -1):
        slide = prs.slides[i]
        for sldId in list(prs.slides._sldIdLst):
            if sldId.id == slide.slide_id:
                prs.part.drop_rel(sldId.rId)
                prs.slides._sldIdLst.remove(sldId)
                break

def format_title(slide, action_title, font_size=24, color=COLOR_NAVY):
    """Format the slide action title in placeholder 0."""
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
    """Set slide footer and slide number."""
    for ph in slide.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 11:  # Footer
            ph.text = "IEEE ICOCO 2026  ·  Beyond Single LLMs: HybridASAG"
            for p in ph.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(10)
                p.font.color.rgb = COLOR_MUTED
        elif idx == 12:  # Slide Number
            ph.text = f"{slide_num} / {total_slides}"
            for p in ph.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(10)
                p.font.color.rgb = COLOR_MUTED

def set_speaker_notes(slide, notes_text):
    """Attach detailed speaking notes to the slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
    """Create a rounded rectangle container card."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def add_placeholder_badge(slide, left=Inches(8.5), top=Inches(6.8), width=Inches(4.2), height=Inches(0.35)):
    """Add subtle annotation marking preliminary/pilot data as agreed."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "[NOTE: Benchmark results from pilot study; update for camera-ready]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9)
    p.font.italic = True
    p.font.color.rgb = COLOR_AMBER
    p.alignment = PP_ALIGN.RIGHT

def main():
    prs = Presentation(TEMPLATE_PATH)
    clear_existing_slides(prs)

    # Copy browser UI screenshot to docs if available
    brain_screenshot = r"C:\Users\lenovo\.gemini\antigravity-ide\brain\979caa21-afe2-43e1-993b-c970de86c418\grading_results_top_1791888272453.png"
    if os.path.exists(brain_screenshot) and not os.path.exists(UI_SCREENSHOT_PATH):
        os.makedirs("docs", exist_ok=True)
        import shutil
        shutil.copyfile(brain_screenshot, UI_SCREENSHOT_PATH)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide (Layout 0)
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    title_ph = slide1.placeholders[0]
    title_ph.text = "Beyond Single LLMs: A Hybrid Multi-Model Framework\nfor Explainable Short Answer Grading"
    for p in title_ph.text_frame.paragraphs:
        p.font.name = FONT_FAMILY
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    sub_ph = slide1.placeholders[1]
    sub_ph.text = (
        "2026 IEEE International Conference on Computing (ICOCO)\n\n"
        "Do Thanh Hai¹ · Research Team²\n"
        "¹Department of Computer Science & Engineering\n"
        "Paper Track: AI in Education & Natural Language Processing"
    )
    for p in sub_ph.text_frame.paragraphs:
        p.font.name = FONT_FAMILY
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(220, 235, 252)

    format_footer_and_number(slide1, 1)
    set_speaker_notes(slide1, (
        "[Timing: 0:00 - 1:00]\n"
        "Opening Hook: Good morning, session chair and esteemed audience. Today I present our paper: "
        "'Beyond Single LLMs: A Hybrid Multi-Model Framework for Explainable Short Answer Grading'.\n"
        "While Large Language Models have taken NLP by storm, deploying them as monolithic black-box "
        "graders introduces severe risks: hallucinated grades, superficial bias, and zero auditability.\n"
        "In this talk, we present HybridASAG: a 5+1 decoupled architecture that achieves superior grading agreement "
        "while providing crystal-clear pedagogical explanations at a fraction of the compute cost."
    ))

    # -------------------------------------------------------------
    # SLIDE 2: Background & Motivation (Layout 5: Title Only + Cards)
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide2, "Automated Grading in Modern Education Demands Both High Precision and Granular Explainability")
    format_footer_and_number(slide2, 2)

    # 3 Column Cards
    col_w = Inches(3.6)
    col_gap = Inches(0.3)
    top_pos = Inches(2.0)
    card_h = Inches(4.5)

    # Card 1: Scale
    c1 = add_card(slide2, Inches(0.92), top_pos, col_w, card_h)
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Scale & Operational Bottlenecks"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE
    p.font.name = FONT_FAMILY

    bullets1 = [
        "Rapid expansion of STEM, CS courses & MOOCs yields tens of thousands of student submissions.",
        "Manual grading introduces human fatigue, inconsistent intra-grader variance, and grading delays.",
        "Short-answer questions (ASAG) are essential: they test conceptual synthesis better than multiple-choice."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.font.name = FONT_FAMILY
        p.space_after = Pt(10)

    # Card 2: Explainability
    c2 = add_card(slide2, Inches(0.92) + col_w + col_gap, top_pos, col_w, card_h)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. The Explainability Imperative"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE
    p.font.name = FONT_FAMILY

    bullets2 = [
        "A single numeric mark (e.g., '7/10') is pedagogically unhelpful for student mastery.",
        "Students need to know: Did they miss key keywords? Was their logic flawed? Or was phrasing informal?",
        "Educators need defensible evidence to resolve grading appeals and maintain academic standards."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.font.name = FONT_FAMILY
        p.space_after = Pt(10)

    # Card 3: Determinism
    c3 = add_card(slide2, Inches(0.92) + (col_w + col_gap) * 2, top_pos, col_w, card_h)
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Practical Deployment Needs"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE
    p.font.name = FONT_FAMILY

    bullets3 = [
        "Real-time feedback during practice exams requires sub-second inference response.",
        "Institution constraints: Low-cost server deployment without clusters of $10,000 GPUs.",
        "Determinism: Identical student answers must reliably receive identical scores under audit."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.font.name = FONT_FAMILY
        p.space_after = Pt(10)

    set_speaker_notes(slide2, (
        "[Timing: 1:00 - 2:15]\n"
        "Key Point: Short Answer Grading (ASAG) sits at the frontier of EdTech. "
        "Unlike multiple choice, open-ended responses test true student comprehension.\n"
        "However, grading at scale hits three hard walls: extreme teacher workload, the strict pedagogical requirement "
        "for actionable diagnostic feedback, and institutional demand for deterministic, low-cost scoring."
    ))

    # -------------------------------------------------------------
    # SLIDE 3: Problems of Monolithic LLMs (Layout 3: Two Content)
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(prs.slide_layouts[3])
    format_title(slide3, "Monolithic LLMs Suffer from Opacity, Hallucinatory Bias, and Compute Inefficiency")
    format_footer_and_number(slide3, 3)

    # Left content
    c_left = slide3.placeholders[1]
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.text = "Black-Box Pitfalls & Hallucinations"
    p0 = tf_l.paragraphs[0]
    p0.font.bold = True
    p0.font.size = Pt(17)
    p0.font.color.rgb = COLOR_AMBER
    p0.font.name = FONT_FAMILY

    pts_l = [
        ("Length & Fluency Bias: ", "Prompted LLMs reward verbose, eloquently written submissions even when factual errors or contradictions exist."),
        ("Stochastic Score Drift: ", "Non-zero temperature or minor prompt formatting changes cause the same submission to oscillate across grade boundaries."),
        ("Post-Hoc Hallucinations: ", "When prompted to justify scores, single LLMs frequently invent missing criteria or reference absent rubric points."),
        ("Vulnerability to Adversarial Text: ", "Students appending 'Grade: 10/10' or injection cues can compromise grading consistency.")
    ]
    for title, desc in pts_l:
        p = tf_l.add_paragraph()
        run1 = p.add_run()
        run1.text = title
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = COLOR_DARK_TEXT
        run2 = p.add_run()
        run2.text = desc
        run2.font.size = Pt(13)
        run2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    # Right content
    c_right = slide3.placeholders[2]
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.text = "Operational & Infrastructure Limits"
    p0_r = tf_r.paragraphs[0]
    p0_r.font.bold = True
    p0_r.font.size = Pt(17)
    p0_r.font.color.rgb = COLOR_AMBER
    p0_r.font.name = FONT_FAMILY

    pts_r = [
        ("Prohibitive VRAM Demands: ", "Serving 7B+ parameter models requires 16GB+ VRAM, multiplying infrastructure costs for schools and universities."),
        ("High Latency Bottleneck: ", "Autoregressive token generation takes 5-15 seconds per single sentence, making concurrent mass exam grading impossible."),
        ("Unreliable JSON Compliance: ", "Generalist LLMs exhibit 10-30% JSON syntax failures under zero-shot prompting, crashing automated pipelines."),
        ("Complete Model Retraining: ", "Adjusting rubric criteria requires expensive re-prompting or LoRA fine-tuning rather than simple weight adjustments.")
    ]
    for title, desc in pts_r:
        p = tf_r.add_paragraph()
        run1 = p.add_run()
        run1.text = title
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = COLOR_DARK_TEXT
        run2 = p.add_run()
        run2.text = desc
        run2.font.size = Pt(13)
        run2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    set_speaker_notes(slide3, (
        "[Timing: 2:15 - 3:30]\n"
        "Key Point: Recent papers simply prompted GPT-4 or Llama to 'Grade this answer 1 to 5'. Why is this dangerous?\n"
        "First, length bias: LLMs fall in love with fluent English, even if the student is totally wrong.\n"
        "Second, latency & compute: universities cannot afford GPU clusters to grade thousands of midterm exams.\n"
        "This motivates our core research question: How do we get the reasoning depth of AI without trusting a black box?"
    ))

    # -------------------------------------------------------------
    # SLIDE 4: Proposed Hybrid Architecture (Layout 5: Title Only + Flow)
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide4, "HybridASAG Decouples Quantitative Scoring from Qualitative Feedback Generation")
    format_footer_and_number(slide4, 4)

    # Top banner: Core concept
    banner = add_card(slide4, Inches(0.92), Inches(1.8), Inches(11.5), Inches(0.9), bg_color=RGBColor(235, 243, 252), border_color=COLOR_IEEE_BLUE)
    btf = banner.text_frame
    btf.word_wrap = True
    bp = btf.paragraphs[0]
    bp.text = "Core Thesis: Do NOT let the LLM invent scores. Use 5 specialized, lightweight models for deterministic measurement, and restrict the LLM strictly to translating scores into natural language feedback."
    bp.font.size = Pt(14)
    bp.font.bold = True
    bp.font.color.rgb = COLOR_IEEE_BLUE
    bp.font.name = FONT_FAMILY

    # Layer 1 Card
    l1_card = add_card(slide4, Inches(0.92), Inches(2.9), Inches(5.6), Inches(3.8))
    l1_tf = l1_card.text_frame
    l1_tf.word_wrap = True
    p = l1_tf.paragraphs[0]
    p.text = "Layer 1: Deterministic Feature Extraction"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY

    l1_items = [
        ("SimCSE (RoBERTa-large): ", "Captures deep semantic equivalence"),
        ("KeyBERT (MiniLM-L6): ", "Extracts and checks core concept coverage"),
        ("RoBERTa Formality: ", "Measures academic vs. casual tone"),
        ("RoBERTa CoLA: ", "Evaluates grammatical acceptability"),
        ("DeBERTa-v3 NLI: ", "Detects factual entailment vs contradiction")
    ]
    for name, desc in l1_items:
        p = l1_tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "✔ " + name
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_IEEE_BLUE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    p_sub = l1_tf.add_paragraph()
    p_sub.text = "Output: 5 bounded sub-scores [0.0 - 1.0] (Memory: < 3 GB)"
    p_sub.font.bold = True
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = COLOR_GREEN

    # Layer 2 Card
    l2_card = add_card(slide4, Inches(6.8), Inches(2.9), Inches(5.6), Inches(3.8))
    l2_tf = l2_card.text_frame
    l2_tf.word_wrap = True
    p = l2_tf.paragraphs[0]
    p.text = "Layer 2: Aggregation & LLM Synthesis"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY

    l2_items = [
        ("Configurable Weighted Sum: ", "Final score = Σ (w_i × s_i) × 100. Fully transparent and auditable."),
        ("Adaptive Presets: ", "Instant recalibration across CS, Science, and Humanities without re-running models."),
        ("Constrained LLM (Qwen2.5-3B): ", "Fed ONLY the prompt, answer, and 5 scores. Instructed to synthesize targeted suggestions."),
        ("Rule-Based Zero-Downtime Fallback: ", "If LLM times out or is bypassed (`skip_llm=True`), deterministic templates generate full feedback in 5ms.")
    ]
    for name, desc in l2_items:
        p = l2_tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "✔ " + name
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_AMBER
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    p_sub2 = l2_tf.add_paragraph()
    p_sub2.text = "Output: Grade 0-100 + Radar Chart + Actionable Diagnostics"
    p_sub2.font.bold = True
    p_sub2.font.size = Pt(12)
    p_sub2.font.color.rgb = COLOR_GREEN

    set_speaker_notes(slide4, (
        "[Timing: 3:30 - 4:45]\n"
        "Key Point: This slide illustrates the central architecture of HybridASAG.\n"
        "Notice the fundamental separation of concerns:\n"
        "Layer 1 extracts 5 orthogonal linguistic features using compact, purpose-built transformers.\n"
        "Layer 2 performs a deterministic weighted sum to produce the grade.\n"
        "The LLM (Qwen2.5-3B) is strictly an explainer—it translates the 5 diagnostic numbers into teacher-like advice."
    ))

    # -------------------------------------------------------------
    # SLIDE 5: Layer 1 Component Deep-Dive (Layout 5: Title Only + Table)
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide5, "Five Lightweight NLP Modules Target Orthogonal Linguistic & Semantic Dimensions")
    format_footer_and_number(slide5, 5)

    # Create Table
    rows = 6
    cols = 5
    tbl_left = Inches(0.92)
    tbl_top = Inches(2.0)
    tbl_width = Inches(11.5)
    tbl_height = Inches(4.7)

    tbl_shape = slide5.shapes.add_table(rows, cols, tbl_left, tbl_top, tbl_width, tbl_height)
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.8) # Dimension
    tbl.columns[1].width = Inches(3.2) # Model Architecture
    tbl.columns[2].width = Inches(1.2) # Size
    tbl.columns[3].width = Inches(3.8) # Pedagogical Purpose
    tbl.columns[4].width = Inches(1.5) # Metric Range

    headers = ["Criterion", "Underlying Model", "Memory", "Target Competence", "Evaluation"]
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    data = [
        ("Semantic Similarity", "princeton-nlp/sup-simcse-roberta-large", "~1.3 GB", "Paraphrase & conceptual synonym alignment", "Cosine Sim [0, 1]"),
        ("Keyword Coverage", "all-MiniLM-L6-v2 (KeyBERT)", "~80 MB", "Verification of required domain terms", "Recall Ratio [0, 1]"),
        ("Formality", "cointegrated/roberta-base-formality", "~500 MB", "Academic tone vs. colloquial slang", "P(formal) [0, 1]"),
        ("Grammar & Fluency", "textattack/roberta-base-CoLA", "~500 MB", "Syntactic validity & linguistic acceptability", "P(acceptable) [0, 1]"),
        ("NLI / Logical Rigor", "cross-encoder/nli-deberta-v3-base", "~700 MB", "Premise-hypothesis entailment vs contradiction", "Softmax Entail [0, 1]")
    ]

    for ri, row in enumerate(data, start=1):
        bg = RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_DARK_TEXT
                if ci == 0:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_IEEE_BLUE
                elif ci in (2, 4):
                    p.alignment = PP_ALIGN.CENTER

    set_speaker_notes(slide5, (
        "[Timing: 4:45 - 5:45]\n"
        "Key Point: Why these five? Because they capture orthogonal facets of student answers.\n"
        "SimCSE handles semantic paraphrasing so students aren't punished for using different wording.\n"
        "KeyBERT ensures required terminology is present.\n"
        "Crucially, DeBERTa NLI checks logical consistency: if a student says 'Plants absorb oxygen to produce CO2', "
        "SimCSE might score high because of word overlap, but DeBERTa immediately flags the Contradiction!"
    ))

    # -------------------------------------------------------------
    # SLIDE 6: Layer 2 - Weight Presets & ML Optimizer (Layout 3: Two Content)
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(prs.slide_layouts[3])
    format_title(slide6, "Flexible Aggregation Balances Pedagogical Presets with Machine-Learned Weights")
    format_footer_and_number(slide6, 6)

    # Left content: Presets Table
    c_left = slide6.placeholders[1]
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.text = "Pedagogical Weight Presets"
    p0 = tf_l.paragraphs[0]
    p0.font.bold = True
    p0.font.size = Pt(17)
    p0.font.color.rgb = COLOR_IEEE_BLUE

    presets_info = [
        ("Balanced (Default): ", "20% each across all 5 dimensions. Suitable for general coursework."),
        ("Content-Focused: ", "40% Semantic, 30% Coverage, 15% Logic, 10% Grammar, 5% Formality (Science/History)."),
        ("Logic-Heavy: ", "35% Logic, 25% Semantic, 15% Coverage, 15% Grammar, 10% Formality (Math/CS proofs)."),
        ("Academic Writing: ", "25% Formality, 25% Grammar, 20% Semantic, 15% Coverage, 15% Logic (Language Arts)."),
        ("Instant Recalibration: ", "Teachers can recalculate entire class cohorts in <10ms via REST API (`/grade/recalculate`) without re-running models!")
    ]
    for title, desc in presets_info:
        p = tf_l.add_paragraph()
        r1 = p.add_run()
        r1.text = title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_DARK_TEXT
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    # Right content: ML Optimizer & Thresholding
    c_right = slide6.placeholders[2]
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.text = "ML Weight Optimization & Calibration"
    p0_r = tf_r.paragraphs[0]
    p0_r.font.bold = True
    p0_r.font.size = Pt(17)
    p0_r.font.color.rgb = COLOR_IEEE_BLUE

    ml_info = [
        ("Learned Empirical Weights: ", "Linear regression on annotated samples discovers optimal feature contributions (Logic 40%, Coverage 30%, Semantic 20%)."),
        ("Zero-Weight Formality: ", "In STEM/CS exams, optimizer sets Formality to 0%—confirming empirical consensus that casual tone should not penalize code comprehension."),
        ("Adaptive Threshold Calibration: ", "Optimizing class thresholds (Correct: 0.65-0.70 / Partial: 0.38-0.40) dramatically boosts classification metrics:"),
        ("  → CS Domain Dataset: ", "Macro F1 increases by +10.3% (0.783 → 0.886); QWK increases by +8.0% (0.830 → 0.909)."),
        ("  → STEM Science Dataset: ", "Macro F1 increases by +7.6% (0.760 → 0.836); QWK increases by +6.8% (0.773 → 0.842).")
    ]
    for item in ml_info:
        p = tf_r.add_paragraph()
        if item[0].startswith("  →"):
            r1 = p.add_run()
            r1.text = item[0]
            r1.font.bold = True
            r1.font.size = Pt(12)
            r1.font.color.rgb = COLOR_GREEN
            r2 = p.add_run()
            r2.text = item[1]
            r2.font.size = Pt(12)
            r2.font.color.rgb = COLOR_DARK_TEXT
        else:
            r1 = p.add_run()
            r1.text = item[0]
            r1.font.bold = True
            r1.font.size = Pt(13)
            r1.font.color.rgb = COLOR_DARK_TEXT
            r2 = p.add_run()
            r2.text = item[1]
            r2.font.size = Pt(12)
            r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(6)

    set_speaker_notes(slide6, (
        "[Timing: 5:45 - 6:45]\n"
        "Key Point: Another major advantage of our hybrid system is flexibility.\n"
        "Instructors can choose human-designed presets (like Logic-Heavy for computer science), "
        "OR use our ML Weight Optimizer to learn weights directly from past student papers.\n"
        "Notice that our optimizer found Formality should have 0% weight in STEM subjects—students shouldn't lose marks for slang if the physics is right!"
    ))

    # -------------------------------------------------------------
    # SLIDE 7: Experimental Setup & Datasets (Layout 3: Two Content)
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(prs.slide_layouts[3])
    format_title(slide7, "Rigorous Benchmark Evaluation Across Multi-Domain Datasets and Agreement Metrics")
    format_footer_and_number(slide7, 7)

    # Left content: Datasets
    c_left = slide7.placeholders[1]
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.text = "Multi-Domain Evaluation Corpus"
    p0 = tf_l.paragraphs[0]
    p0.font.bold = True
    p0.font.size = Pt(17)
    p0.font.color.rgb = COLOR_IEEE_BLUE

    corpus_pts = [
        ("SemEval-2013 Task 7: ", "Golden international benchmark for ASAG. Covers Beetle (electrical fundamentals, 4,000+ answers) and Scientsbank (general science, 10,000+ answers)."),
        ("Computer Science Benchmark: ", "150 verified student submissions on Data Structures, Algorithms, and Object-Oriented Programming (Mohler et al. protocol)."),
        ("STEM Science Dataset: ", "200 curated student responses in Physics, Chemistry, and Environmental Science."),
        ("Label Distribution: ", "Balanced 3-way evaluation: Correct (100%), Partially Correct (50%), and Contradictory/Incorrect (0%).")
    ]
    for title, desc in corpus_pts:
        p = tf_l.add_paragraph()
        r1 = p.add_run()
        r1.text = title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_DARK_TEXT
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    # Right content: Metrics
    c_right = slide7.placeholders[2]
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.text = "Statistical Evaluation Metrics"
    p0_r = tf_r.paragraphs[0]
    p0_r.font.bold = True
    p0_r.font.size = Pt(17)
    p0_r.font.color.rgb = COLOR_IEEE_BLUE

    metrics_pts = [
        ("Quadratic Weighted Kappa (QWK): ", "Primary conference metric. Penalizes severe misclassifications (Correct vs Contradictory) quadratically heavier than minor shifts (Correct vs Partial)."),
        ("Pearson Correlation (r): ", "Measures linear agreement between predicted continuous scores and human instructor scores."),
        ("Macro & Weighted F1-Score: ", "Ensures reliable performance across minority class labels without majority-class skew."),
        ("Validation Protocol: ", "5-fold cross-validation with paired non-parametric Wilcoxon signed-rank tests and Bonferroni corrections.")
    ]
    for title, desc in metrics_pts:
        p = tf_r.add_paragraph()
        r1 = p.add_run()
        r1.text = title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_DARK_TEXT
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    set_speaker_notes(slide7, (
        "[Timing: 6:45 - 7:45]\n"
        "Key Point: Moving to evaluation: we tested across three distinct domains including the gold-standard SemEval-2013 Task 7.\n"
        "In educational assessment, raw accuracy is misleading. Quadratic Weighted Kappa (QWK) is the industry standard "
        "because grading a completely wrong answer as 'correct' is far worse than grading 'correct' as 'partially correct'.\n"
        "We also report Pearson r and F1."
    ))

    # -------------------------------------------------------------
    # SLIDE 8: Main Quantitative Results (Layout 5: Title Only + Table & Callout)
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide8, "HybridASAG Outperforms Baselines with Superior Inter-Rater Agreement and Accuracy")
    format_footer_and_number(slide8, 8)
    add_placeholder_badge(slide8)

    # Results Table
    t_left = Inches(0.92)
    t_top = Inches(2.0)
    t_w = Inches(11.5)
    t_h = Inches(3.6)

    tb_shape = slide8.shapes.add_table(7, 6, t_left, t_top, t_w, t_h)
    tb = tb_shape.table
    tb.columns[0].width = Inches(2.5) # Model
    tb.columns[1].width = Inches(1.8) # Accuracy
    tb.columns[2].width = Inches(1.8) # Macro F1
    tb.columns[3].width = Inches(1.8) # Weighted F1
    tb.columns[4].width = Inches(1.8) # QWK
    tb.columns[5].width = Inches(1.8) # Pearson r

    h_titles = ["Model Architecture", "Accuracy", "Macro F1", "Weighted F1", "QWK (Kappa)", "Pearson (r)"]
    for ci, h in enumerate(h_titles):
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

    results_data = [
        ("TF-IDF Baseline", "0.523 ± 0.03", "0.489 ± 0.03", "0.512 ± 0.03", "0.412 ± 0.04", "0.534 ± 0.04"),
        ("Keywords (KeyBERT only)", "0.467 ± 0.04", "0.423 ± 0.04", "0.451 ± 0.04", "0.356 ± 0.05", "0.478 ± 0.04"),
        ("SBERT (Sentence-BERT)", "0.612 ± 0.03", "0.587 ± 0.03", "0.601 ± 0.03", "0.534 ± 0.03", "0.623 ± 0.03"),
        ("LLM-ZeroShot (Prompted)", "0.645 ± 0.03", "0.621 ± 0.04", "0.638 ± 0.03", "0.567 ± 0.04", "0.654 ± 0.04"),
        ("Hybrid-RuleOnly (No LLM)", "0.723 ± 0.02", "0.698 ± 0.03", "0.715 ± 0.02", "0.645 ± 0.03", "0.734 ± 0.03"),
        ("Hybrid-ASAG (Proposed Full)", "0.756 ± 0.02", "0.734 ± 0.02", "0.748 ± 0.02", "0.689 ± 0.03", "0.768 ± 0.02")
    ]

    for ri, r in enumerate(results_data, start=1):
        is_hybrid = "Hybrid-ASAG" in r[0]
        bg = RGBColor(230, 245, 235) if is_hybrid else (RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE)
        for ci, val in enumerate(r):
            cell = tb.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(12)
                if is_hybrid:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_GREEN if ci > 0 else COLOR_NAVY
                else:
                    p.font.color.rgb = COLOR_DARK_TEXT
                if ci > 0:
                    p.alignment = PP_ALIGN.CENTER

    # Bottom summary callout
    b_card = add_card(slide8, Inches(0.92), Inches(5.8), Inches(11.5), Inches(0.9), bg_color=RGBColor(240, 248, 255), border_color=COLOR_IEEE_BLUE)
    btf = b_card.text_frame
    btf.word_wrap = True
    p = btf.paragraphs[0]
    p.text = "Key Findings & Statistical Tests:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_IEEE_BLUE
    p.font.name = FONT_FAMILY

    bullets = [
        "Hybrid-ASAG outperforms Prompted Zero-Shot LLM by +11.1% Accuracy and +12.2% QWK (p = 0.002, Very Significant **).",
        "Even without LLM generation ('Hybrid-RuleOnly'), the 5 lightweight extractors beat SBERT and Zero-Shot LLM (QWK 0.645 vs 0.567)."
    ]
    for b in bullets:
        p = btf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.font.name = FONT_FAMILY

    set_speaker_notes(slide8, (
        "[Timing: 7:45 - 9:00]\n"
        "Key Point: Here are the main results.\n"
        "Notice the comparison: Prompted Zero-Shot LLMs achieve 0.645 accuracy and 0.567 QWK.\n"
        "Our HybridASAG system achieves 0.756 accuracy and 0.689 QWK—a massive 12.2% gain in inter-rater agreement.\n"
        "Crucially, look at 'Hybrid-RuleOnly': even if you disable the LLM completely, our 5 models alone score 0.645 QWK, "
        "proving that the structured features—not LLM hallucinations—drive the superior performance."
    ))

    # -------------------------------------------------------------
    # SLIDE 9: Ablation Study (Layout 5: Title Only + Image & Bullets)
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide9, "Ablation Confirms Semantic Similarity and NLI Logic as the Primary Drivers of Accuracy")
    format_footer_and_number(slide9, 9)
    add_placeholder_badge(slide9)

    # Insert Fig 11 if exists
    if os.path.exists(FIG11_PATH):
        slide9.shapes.add_picture(FIG11_PATH, Inches(0.92), Inches(1.9), width=Inches(7.2))
    elif os.path.exists(FIG1_PATH):
        slide9.shapes.add_picture(FIG1_PATH, Inches(0.92), Inches(1.9), width=Inches(7.2))

    # Right side: Analytical Takeaways
    rc = add_card(slide9, Inches(8.3), Inches(1.9), Inches(4.1), Inches(4.7))
    rtf = rc.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "Component Impact Analysis"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY
    p.font.name = FONT_FAMILY

    abl_pts = [
        ("NLI Logic is Crucial for QWK: ", "Omitting DeBERTa causes the most catastrophic agreement drop (Δ QWK = -0.103, p < 0.001 ***). Logic prevents misclassifying contradiction answers as correct."),
        ("Semantic Embeddings Anchor F1: ", "Removing SimCSE produces the largest Macro F1 drop (Δ F1 = -0.053, p < 0.01 **), confirming paraphrastic semantic capture is vital."),
        ("Surface Form Does Not Penalize Knowledge: ", "Ablating Formality (Δ F1 = -0.008) and Grammar (Δ F1 = -0.015) shows negligible impact. This prevents bias against non-native ESL writers.")
    ]
    for title, desc in abl_pts:
        p = rtf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_AMBER if "NLI" in title else COLOR_IEEE_BLUE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    set_speaker_notes(slide9, (
        "[Timing: 9:00 - 10:15]\n"
        "Key Point: Which component matters most? Our ablation study provides definitive empirical proof.\n"
        "Looking at the right chart, removing NLI Logic results in a 0.103 crash in QWK with p < 0.001.\n"
        "Why? Because when a student gives a logically inverted answer, SimCSE thinks the words match! Only NLI catches the fact that they contradicted the premise.\n"
        "Meanwhile, Grammar and Formality have very low impact—which is exactly what we want in fair grading."
    ))

    # -------------------------------------------------------------
    # SLIDE 10: LLM Comparative Benchmark (Layout 5: Title Only + Table & Takeaways)
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide10, "Qwen2.5-3B Delivers High JSON Schema Compliance with Minimal Compute Overhead")
    format_footer_and_number(slide10, 10)
    add_placeholder_badge(slide10)

    # Benchmark Table
    tb_l = Inches(0.92)
    tb_t = Inches(2.0)
    tb_w = Inches(11.5)
    tb_h = Inches(2.6)

    tbl_s = slide10.shapes.add_table(4, 6, tb_l, tb_t, tb_w, tb_h)
    tbl = tbl_s.table
    tbl.columns[0].width = Inches(2.8) # Model
    tbl.columns[1].width = Inches(1.7) # Model Size
    tbl.columns[2].width = Inches(1.7) # JSON Compl.
    tbl.columns[3].width = Inches(1.7) # Parse Error
    tbl.columns[4].width = Inches(1.8) # Hallucination Rate
    tbl.columns[5].width = Inches(1.8) # Peak RAM

    h_llm = ["Candidate Model", "Parameters / Disk", "JSON Compliance", "Parse Error Rate", "Hallucination Rate", "Peak RAM (CPU)"]
    for ci, h in enumerate(h_llm):
        cell = tbl.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT_FAMILY
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    llm_rows = [
        ("Qwen2.5-3B-Instruct (Selected)", "3.1B (~6.5 GB)", "90.5%", "8.1%", "10.7%", "6.7 GB"),
        ("Phi-3-Mini-4K-Instruct", "3.8B (~7.2 GB)", "78.0%", "21.4%", "18.1%", "7.2 GB"),
        ("Qwen2.5-1.5B-Instruct", "1.5B (~3.1 GB)", "68.9%", "29.5%", "23.5%", "3.1 GB")
    ]
    for ri, r in enumerate(llm_rows, start=1):
        is_sel = ri == 1
        bg = RGBColor(230, 245, 235) if is_sel else (RGBColor(245, 248, 253) if ri % 2 == 1 else COLOR_WHITE)
        for ci, val in enumerate(r):
            cell = tbl.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT_FAMILY
                p.font.size = Pt(12)
                if is_sel:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_GREEN if ci in (2, 3) else COLOR_NAVY
                else:
                    p.font.color.rgb = COLOR_DARK_TEXT
                if ci > 0:
                    p.alignment = PP_ALIGN.CENTER

    # Engineering Highlights Card
    h_card = add_card(slide10, Inches(0.92), Inches(4.9), Inches(11.5), Inches(1.8))
    htf = h_card.text_frame
    htf.word_wrap = True
    p = htf.paragraphs[0]
    p.text = "Deployment & Engineering Takeaways:"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_IEEE_BLUE

    llm_eng_pts = [
        "Superior Schema Adherence: Qwen2.5-3B achieved 90.5% JSON compliance, compared to only 78.0% for Phi-3-Mini, significantly reducing pipeline crashes.",
        "4-Bit Quantization Ready: When deployed with 4-bit NF4 quantization, Qwen2.5-3B fits within 2.5 GB of VRAM, running on commodity edge workstations.",
        "Deterministic Fallback Safety Net: In high-concurrency situations, `skip_llm=True` bypasses generation and produces templated feedback in 5ms with zero VRAM."
    ]
    for b in llm_eng_pts:
        p = htf.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(4)

    set_speaker_notes(slide10, (
        "[Timing: 10:15 - 11:15]\n"
        "Key Point: To generate feedback, we evaluated multiple candidate SLMs (Small Language Models).\n"
        "In automated production pipelines, formatting failures are unacceptable. Phi-3-Mini failed JSON parsing 21% of the time, "
        "whereas Qwen2.5-3B achieved over 90% compliance.\n"
        "Furthermore, with 4-bit quantization, it runs in just 2.5 GB of VRAM. If the system is under heavy load, "
        "our rule-based engine can kick in instantly without dropping a single student request."
    ))

    # -------------------------------------------------------------
    # SLIDE 11: System Demonstration & Explainability (Layout 5: Title Only + UI)
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide11, "Interactive Dashboard Translates Numerical Sub-Scores into Actionable Diagnostics")
    format_footer_and_number(slide11, 11)

    # Insert UI Screenshot
    if os.path.exists(UI_SCREENSHOT_PATH):
        slide11.shapes.add_picture(UI_SCREENSHOT_PATH, Inches(0.92), Inches(1.9), width=Inches(7.3))

    # Right side: Pedagogical Explainability breakdown
    ui_card = add_card(slide11, Inches(8.4), Inches(1.9), Inches(4.0), Inches(4.7))
    uitf = ui_card.text_frame
    uitf.word_wrap = True
    p = uitf.paragraphs[0]
    p.text = "Explainability in Practice"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_NAVY

    ui_pts = [
        ("Transparent Composite Score: ", "Overall 67/100 clearly displayed alongside classification badge ('Partially Correct')."),
        ("Fine-Grained Criterion Breakdown: ", "Visual bars directly show where points were lost: Formality (100%) and Grammar (95%) were high, but Keyword Coverage was only 20%."),
        ("Missing Vocabulary Extraction: ", "Explicitly lists key terms missing from reference answer: 'photosynthesis', 'converts', 'carbon dioxide'."),
        ("Actionable Pedagogical Advice: ", "Qwen2.5-3B provides a targeted tip: 'Try to explicitly connect how solar energy converts carbon dioxide into glucose'.")
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
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(8)

    set_speaker_notes(slide11, (
        "[Timing: 11:15 - 12:15]\n"
        "Key Point: This slide shows our live production system running in our laboratory.\n"
        "Look at how an educator or student experiences this result.\n"
        "The student scored 67. Why? Not because the model 'felt like it'.\n"
        "They see 95% grammar, 100% formality, but only 20% keyword coverage.\n"
        "The system explicitly highlights that the words 'photosynthesis' and 'carbon dioxide' were omitted.\n"
        "This is true formative assessment: transparent, reproducible, and instructive."
    ))

    # -------------------------------------------------------------
    # SLIDE 12: Conclusion & Contributions (Layout 1: Title and Content / 3 Cards)
    # -------------------------------------------------------------
    slide12 = prs.slides.add_slide(prs.slide_layouts[5])
    format_title(slide12, "Summary: A Trustworthy, Interpretable, and Resource-Efficient Paradigm for ASAG")
    format_footer_and_number(slide12, 12)

    # 3 Summary Cards
    c_w = Inches(3.6)
    c_h = Inches(4.5)
    c_y = Inches(2.0)

    # Contrib 1
    c1 = add_card(slide12, Inches(0.92), c_y, c_w, c_h)
    c1_tf = c1.text_frame
    c1_tf.word_wrap = True
    p = c1_tf.paragraphs[0]
    p.text = "1. Decoupled Architecture"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE

    c1_bullets = [
        "Proposed a novel 5+1 hybrid paradigm that decouples numerical grading from textual feedback.",
        "Eliminates LLM hallucination in grade calculation while retaining teacher-like feedback generation.",
        "Supports domain-specific rubric weighting without model fine-tuning."
    ]
    for b in c1_bullets:
        p = c1_tf.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(10)

    # Contrib 2
    c2 = add_card(slide12, Inches(0.92) + c_w + Inches(0.35), c_y, c_w, c_h)
    c2_tf = c2.text_frame
    c2_tf.word_wrap = True
    p = c2_tf.paragraphs[0]
    p.text = "2. Empirical Superiority"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE

    c2_bullets = [
        "Achieves 0.689 QWK and 0.768 Pearson r, outperforming Prompted Zero-Shot LLMs and SBERT.",
        "Statistically verified through Wilcoxon signed-rank and Bonferroni tests.",
        "Ablation identifies NLI Logic as the single most critical factor in preventing erroneous grading."
    ]
    for b in c2_bullets:
        p = c2_tf.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(10)

    # Contrib 3
    c3 = add_card(slide12, Inches(0.92) + (c_w + Inches(0.35)) * 2, c_y, c_w, c_h)
    c3_tf = c3.text_frame
    c3_tf.word_wrap = True
    p = c3_tf.paragraphs[0]
    p.text = "3. Practical Deployment"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_IEEE_BLUE

    c3_bullets = [
        "Inference requires < 3 GB RAM for scoring, running on standard CPU hardware.",
        "4-bit quantized Qwen2.5-3B enables low-latency feedback with 90.5% JSON reliability.",
        "Fully implemented open-source FastAPI and React dashboard ready for university adoption."
    ]
    for b in c3_bullets:
        p = c3_tf.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(10)

    set_speaker_notes(slide12, (
        "[Timing: 12:15 - 13:15]\n"
        "Key Point: To summarize our three core contributions:\n"
        "First: A new decoupled paradigm that solves the black-box problem.\n"
        "Second: Comprehensive empirical validation proving that NLI Logic + Semantic Similarity beat monolithic LLMs.\n"
        "Third: A lightweight, open-source engineering system ready for real-world school and university deployment."
    ))

    # -------------------------------------------------------------
    # SLIDE 13: Future Work & Q&A Anchor (Layout 3: Two Content)
    # -------------------------------------------------------------
    slide13 = prs.slides.add_slide(prs.slide_layouts[3])
    format_title(slide13, "Advancing Toward Cross-Lingual Evaluation and Adaptive Rubric Calibration")
    format_footer_and_number(slide13, 13)

    # Left content: Future Work
    c_left = slide13.placeholders[1]
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.text = "Future Research Directions"
    p0 = tf_l.paragraphs[0]
    p0.font.bold = True
    p0.font.size = Pt(18)
    p0.font.color.rgb = COLOR_IEEE_BLUE

    fw_items = [
        ("Cross-Lingual Generalization: ", "Extending the 5-model pipeline to low-resource languages (e.g., Vietnamese, Malay) via multilingual encoders (XLM-RoBERTa, mDeBERTa)."),
        ("Dynamic Rubric Synthesis: ", "Automated few-shot generation of question-specific rubric weights based on Bloom's taxonomy level."),
        ("Multimodal STEM Extension: ", "Integrating OCR and vision-language encoders to evaluate scientific diagrams and mathematical LaTeX derivations.")
    ]
    for title, desc in fw_items:
        p = tf_l.add_paragraph()
        r1 = p.add_run()
        r1.text = title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_DARK_TEXT
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_DARK_TEXT
        p.space_after = Pt(10)

    # Right content: Contact & Q&A
    c_right = slide13.placeholders[2]
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.text = "Open Science & Discussion"
    p0_r = tf_r.paragraphs[0]
    p0_r.font.bold = True
    p0_r.font.size = Pt(18)
    p0_r.font.color.rgb = COLOR_AMBER

    contacts = [
        ("Code & Data Repository: ", "github.com/dothanhhxx/AutoGradeSystem"),
        ("Pretrained Weights: ", "Hugging Face Model Hub (Open Weights)"),
        ("Corresponding Author: ", "Do Thanh Hai"),
        ("Conference Session: ", "IEEE ICOCO 2026 — AI in Education")
    ]
    for title, desc in contacts:
        p = tf_r.add_paragraph()
        r1 = p.add_run()
        r1.text = title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_DARK_TEXT
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_IEEE_BLUE if "github" in desc else COLOR_DARK_TEXT
        p.space_after = Pt(8)

    # Clean bottom box for Thank You and Q&A (no stray placeholder bullet)
    qa_card = add_card(slide13, Inches(6.8), Inches(5.5), Inches(5.6), Inches(1.2), bg_color=RGBColor(240, 245, 252), border_color=COLOR_IEEE_BLUE)
    qtf = qa_card.text_frame
    qtf.word_wrap = True
    p_qa = qtf.paragraphs[0]
    p_qa.text = "Thank you for your attention!\nWe welcome questions from the committee & audience."
    p_qa.font.bold = True
    p_qa.font.size = Pt(14)
    p_qa.font.color.rgb = COLOR_NAVY
    p_qa.alignment = PP_ALIGN.CENTER

    set_speaker_notes(slide13, (
        "[Timing: 13:15 - 15:00+ (Q&A Anchor)]\n"
        "Key Point: Leave this slide on screen during the 2-3 minute Q&A session.\n"
        "Anticipated Q&A Questions:\n"
        "1. Q: Why not fine-tune a single LLaMA model on SemEval?\n"
        "   A: Fine-tuning a single model still reproduces a black box. You cannot guarantee that length bias disappears, "
        "and you cannot adjust weights per course without retraining.\n"
        "2. Q: Why is Formality given 0% weight in ML optimization?\n"
        "   A: Because in STEM/CS grading, penalizing informal language often introduces cultural and ESL bias rather than measuring true mastery.\n"
        "3. Q: How does latency compare?\n"
        "   A: Layer 1 runs in parallel in ~120ms on GPU or ~450ms on CPU. Pure LLM generation takes 5-15 seconds."
    ))

    prs.save(OUTPUT_PATH)
    print(f"Successfully generated {OUTPUT_PATH} with {len(prs.slides)} slides.")

if __name__ == "__main__":
    main()
