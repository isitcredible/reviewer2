"""Stage functions for the Reviewer 2 pipeline.

Each stage is a thin wrapper that loads a prompt, substitutes context, calls
Gemini, and writes the result to ``output_dir``. Orchestration — which stages
run in which order, what gets chained into what — lives in ``pipeline.py``.
"""

from __future__ import annotations

import json
import os

from reviewer2.core import call_gemini, load_prompt, save_output
from reviewer2.helpers import (
    extract_info_fields,
    get_citation_block,
    inject_page_numbers,
    load_instruction,
)
from reviewer2.mathpix import extract_equations_mathpix

APPENDIX_INJECTION = """

═══════════════════════════════════════════════════════════════
**CRITICAL APPENDIX NOTE:**

This material has been provided as verified supplementary context for the review.
Treat it as authoritative.
It will be added at the end of the report, and should be referred to specifically as "the Appendix to this review".
═══════════════════════════════════════════════════════════════

{{APPENDIX_CONTENT}}
"""

# ============================================================================
# STAGE 0: EXTRACTION
# ============================================================================

def stage_00a_metadata(pdf_path, output_dir):
    print("  → 00a Metadata (Search Enabled)...")
    prompt = load_prompt("prompts/00a_metadata.txt")

    result = call_gemini(
        prompt,
        pdf_path,
        model_type="flash_2_5",
        temperature=0.0,
        thinking_budget=0,
        system_instruction=load_instruction("bureaucrat.txt"),
        step="00a_metadata",
        use_search=True,
    )
    save_output(result, "00a_metadata.txt", output_dir)
    return result

def stage_00b_metadata_clean(raw, pdf_path, output_dir):
    print("  → 00b Metadata Clean...")
    prompt = load_prompt("prompts/00b_metadata_clean.txt").replace("{{RAW_INPUT}}", raw)
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=200, system_instruction=load_instruction("bureaucrat.txt"), step="00b_metadata_clean")
    save_output(result, "00b_metadata_clean.txt", output_dir)
    return extract_info_fields(result)

def stage_00b_2_metadata_math(metadata, output_dir):
    print("  → 00b_2 Metadata Math Fixer...")
    target_keys = ["title", "abstract_summary", "key_methodology", "research_question", "central_argument"]
    subset = {k: metadata.get(k) for k in target_keys if metadata.get(k)}
    if not subset:
        return metadata

    json_input = json.dumps(subset, indent=2)
    prompt = (
        "You are a LaTeX expert. Review the following JSON metadata. "
        "Identify mathematical notation within the text values and convert them into proper inline LaTeX mathematical formatting ($...$).\n"
        "DO NOT CHANGE ANY OTHER WORDING; DO NOT ABBREVIATE ANYTHING; E.G. YOU MUST NOT CHANGE GROSS DOMESTIC PRODUCT TO GDP.\n"
        "The text values should be identical to the Input JSON, except with proper inline LaTeX mathematical formatting ($...$).\n"
        "YOUR ONLY FUNCTION IS TO CONVERT MATHEMATICAL NOTATION INTO PROPER INLINE LATEX MATHEMATICAL FORMATTING.\n"
        "MAKE NO OTHER CHANGES.\n"
        "Return ONLY the raw valid JSON object.\n\n"
        f"Input JSON:\n{json_input}"
    )

    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=100, step="00b_2_metadata_math")

    try:
        clean_result = result.replace("```json", "").replace("```", "").strip()
        fixed_data = json.loads(clean_result)
        for k, v in fixed_data.items():
            if k in metadata and v:
                metadata[k] = v
        save_output(json.dumps(fixed_data, indent=2), "00b_2_metadata_math.json", output_dir)
    except json.JSONDecodeError:
        print("  ⚠ Warning: Failed to parse JSON from Metadata Math Fixer.")
        save_output(result, "00b_2_metadata_math_FAILED.txt", output_dir)
    return metadata

def stage_00c_contributions(pdf_path, metadata, output_dir):
    print("  → 00c Contributions...")
    prompt = load_prompt("prompts/00c_contributions.txt").replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=1000, system_instruction=load_instruction("bureaucrat.txt"), step="00c_contributions")
    save_output(result, "00c_contributions.txt", output_dir)
    return result

# ============================================================================
# STAGE 1: RESEARCH (RED TEAM)
# ============================================================================

def stage_01a_breaker(pdf_path, metadata, output_dir, contribs, appendix_text=None):
    print("  → 01a Breaker...")
    prompt = load_prompt("prompts/01a_breaker.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.2, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01a_breaker")
    save_output(result, "01a_breaker.txt", output_dir)
    return result

def stage_01a_2_breaker_revisit(pdf_path, round_1_output, metadata, output_dir):
    print("  → 01a_2 Breaker Round 2 (Non-Empirical Deep Dive)...")

    base_prompt = load_prompt("prompts/01a_breaker.txt")

    revisit_instruction = f"""

    ═══════════════════════════════════════════════════════════════
    **ROUND 2 INSTRUCTION: DEEP REASONING**
    ═══════════════════════════════════════════════════════════════
    You have already analyzed this text once. These were the results of Round 1:

    {round_1_output}

    **YOUR TASK:**
    NOW TRY TO FIND ANOTHER 10 ISSUES.
    THEY MUST BE **DIFFERENT** TO THE ISSUES FROM ROUND 1.

    USE YOUR SOTA REASONING AND MULTIMODAL UNDERSTANDING TO DO BETTER.
    DIG DEEPER INTO LOGICAL INCONSISTENCIES, THEORETICAL GAPS, AND ARGUMENTATIVE FLAWS.
    """

    full_prompt = base_prompt + revisit_instruction
    full_prompt = full_prompt.replace("{{CONTRIBUTIONS}}", "(See text)").replace("{{CITATION}}", get_citation_block(metadata))
    full_prompt = inject_page_numbers(full_prompt, metadata)

    result = call_gemini(
        full_prompt,
        pdf_path,
        model_type="pro_3_1",
        temperature=0.6,
        thinking_level="high",
        system_instruction=load_instruction("researcher.txt"),
        step="01a_2_breaker_revisit",
    )

    save_output(result, "01a_2_breaker_revisit.txt", output_dir)
    return result

def stage_01b_butcher(pdf_path, metadata, output_dir, contribs, appendix_text=None):
    print("  → 01b Butcher...")
    prompt = load_prompt("prompts/01b_butcher.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.2, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01b_butcher")
    save_output(result, "01b_butcher.txt", output_dir)
    return result

def stage_01c_shredder(pdf_path, metadata, output_dir, contribs, appendix_text=None):
    print("  → 01c Shredder...")
    prompt = load_prompt("prompts/01c_shredder.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.1, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01c_shredder")
    save_output(result, "01c_shredder.txt", output_dir)
    return result

def stage_01d_collector(pdf_path, b1, s1, metadata, output_dir, appendix_text=None):
    print("  → 01d Collector...")
    context = f"BUTCHER:\n{b1}\nSHREDDER:\n{s1}"
    prompt = load_prompt("prompts/01d_collector.txt").replace("{{INPUT_CONTEXT}}", context).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01d_collector")
    save_output(result, "01d_collector.txt", output_dir)
    return result

def stage_01e_math_extract(pdf_path, contribs, metadata, output_dir, appendix_text=None):
    print("  → 01e Math...")
    prompt = load_prompt("prompts/01e_math.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="01e_math")
    save_output(result, "01e_math.txt", output_dir)
    return result

EQUATION_REFERENCE = """

═══════════════════════════════════════════════════════════════
EQUATION REFERENCE (from independent OCR extraction)
═══════════════════════════════════════════════════════════════

The following equations were extracted from the PDF by a dedicated OCR system (Mathpix), NOT by visual AI parsing. These are your PRIMARY reference for what the equations actually say. When checking equations, read them from this list rather than trying to parse them visually from the PDF. If your visual reading of the PDF disagrees with the Mathpix extraction below, TRUST THE MATHPIX VERSION unless you have strong reason to believe the OCR failed (e.g. garbled output, missing equation).

Use the PDF for context (prose, definitions, variable meanings) but use THIS LIST for the actual mathematical content of displayed equations.

PAY SPECIAL ATTENTION to the scope of negation signs. In LaTeX, -\\frac{A+B}{C} means the negative applies to the ENTIRE fraction, giving numerator -(A+B) = -A-B. The + between A and B is correct.

{{EQUATIONS}}
"""


def _inject_equations(prompt, equations):
    """Append equation reference to prompt if available."""
    if equations:
        return prompt + EQUATION_REFERENCE.replace("{{EQUATIONS}}", equations)
    return prompt


def stage_01e2_equation_extraction(pdf_path, metadata, output_dir):
    """Extract equations via Mathpix (all pages) + Flash (cleanup)."""
    print("  → 01e2 Equation Extraction (Mathpix)...")

    try:
        from pypdf import PdfReader
        num_pages = len(PdfReader(pdf_path).pages)
    except Exception:
        num_pages = 100
    page_ranges = f"1-{num_pages}"
    print(f"    ↳ Sending all {num_pages} pages to Mathpix")

    mathpix_md = extract_equations_mathpix(pdf_path, page_ranges)
    if not mathpix_md:
        save_output(None, "01e2_equations.txt", output_dir)
        return None

    save_output(mathpix_md, "01e2_mathpix_raw.txt", output_dir)

    extract_prompt = (
        "Below is text extracted from an academic paper by an OCR system. "
        "Extract EVERY displayed equation (equations on their own line, delimited by $$...$$) "
        "into a clean numbered list.\n\n"
        "For each equation, output:\n"
        "- The equation number (from the \\tag{} if present, otherwise assign a sequential label like [unnumbered-1])\n"
        "- The LaTeX of the equation exactly as extracted\n\n"
        "Do NOT include inline math. Only displayed equations.\n"
        "Do NOT modify the LaTeX. Copy it exactly.\n"
        "Output format:\n\n"
        "Eq [number]: [LaTeX]\n\n"
        "---\n\n" + mathpix_md
    )
    equations = call_gemini(
        extract_prompt, None,
        model_type="flash_2_5", temperature=0.0, thinking_budget=-1,
        system_instruction="You are a precise equation extractor.",
        step="01e2_extract",
    )

    save_output(equations, "01e2_equations.txt", output_dir)
    return equations


def stage_01fa_math_check(pdf_path, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fa The Re-Deriver...")
    prompt = load_prompt("prompts/01fa_math_check.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fa_math_check")
    save_output(result, "01fa_math_check.txt", output_dir)
    return result

def stage_01fb_math_proofread(pdf_path, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fb The Proofreader...")
    prompt = load_prompt("prompts/01fb_math_proofread.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fb_math_proofread")
    save_output(result, "01fb_math_proofread.txt", output_dir)
    return result

def stage_01fc_math_audit(pdf_path, proofreader_output, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fc The Auditor...")
    prompt = load_prompt("prompts/01fc_math_audit.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{PROOFREADER_OUTPUT}}", proofreader_output or "=NULL=")
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fc_math_audit")
    save_output(result, "01fc_math_audit.txt", output_dir)
    return result

MATHPIX_PAPER_INJECTION = """

═══════════════════════════════════════════════════════════════
FULL PAPER TEXT (from Mathpix OCR)
═══════════════════════════════════════════════════════════════

The following is the complete text of the paper, extracted by Mathpix OCR. All equations (both displayed and inline) are rendered in LaTeX. This is your PRIMARY source for reading equations — it is more reliable than visual PDF parsing. Use this text for all derivation checks, scope verification, and notation verification.

{{MATHPIX_RAW}}
"""

def stage_01fd_math_sober(pdf_path, rederiver_output, proofreader_output, auditor_output, contribs, metadata, output_dir, appendix_text=None, equations=None, mathpix_raw=None):
    print("  → 01fd Sober Math Checker...")
    prompt = load_prompt("prompts/01fd_math_sober.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{REDERIVER_OUTPUT}}", rederiver_output).replace("{{PROOFREADER_OUTPUT}}", proofreader_output or "=NULL=").replace("{{AUDITOR_OUTPUT}}", auditor_output)
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    if mathpix_raw:
        prompt += MATHPIX_PAPER_INJECTION.replace("{{MATHPIX_RAW}}", mathpix_raw)
    result = call_gemini(prompt, None if mathpix_raw else pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fd_math_sober")
    save_output(result, "01fd_math_sober.txt", output_dir)
    return result

def stage_01g_the_void(pdf_path, contribs, metadata, output_dir, appendix_text=None):
    print("  → 01g The Void...")
    prompt = load_prompt("prompts/01g_the_void.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text:
        prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.3, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01g_the_void")
    save_output(result, "01g_the_void.txt", output_dir)
    return result

# ============================================================================
# CODE ANALYSIS STAGES
# ============================================================================

_BINARY_EXTENSIONS = {".dta", ".rdata", ".rds", ".xlsx", ".xls", ".zip",
                      ".gz", ".tar", ".7z", ".pkl", ".feather", ".parquet",
                      ".sav", ".por"}

def _extract_file_text(file_path):
    """Extract readable text from a code or document file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return f"[Could not extract PDF text: {e}]"

    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[Could not extract Word document text: {e}]"

    if ext in _BINARY_EXTENSIONS:
        return f"[Binary file — content not extractable: {os.path.basename(file_path)}]"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[Could not read file: {e}]"


def compile_code_to_pdf(code_dir, ordered_files, output_path):
    """Compile ordered code files into a dense single PDF."""
    print("  → 01_code_b Compiling code to PDF...")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas as rl_canvas

    FONT_SIZE = 7
    LEADING = 8.5
    MARGIN = 20
    HEADER_SIZE = 8
    page_w, page_h = A4
    text_w = page_w - 2 * MARGIN

    c = rl_canvas.Canvas(output_path, pagesize=A4)
    y = page_h - MARGIN

    def new_page():
        nonlocal y
        c.showPage()
        y = page_h - MARGIN

    def draw_line(text, font="Courier", size=FONT_SIZE):
        nonlocal y
        if y < MARGIN + LEADING:
            new_page()
        c.setFont(font, size)
        c.drawString(MARGIN, y, text)
        y -= LEADING

    map_path = os.path.join(code_dir, "_file_map.txt")
    if os.path.exists(map_path):
        map_text = _extract_file_text(map_path)
        draw_line("=== REPLICATION PACKAGE FILE MAP ===", "Helvetica-Bold", HEADER_SIZE)
        y -= 2
        for line in map_text.splitlines():
            if line.startswith("Extracted files to be reviewed:"):
                continue
            if "(Click to clear" in line:
                continue
            draw_line(line, "Courier", FONT_SIZE)
        y -= LEADING * 2

    for fname in ordered_files:
        fpath = os.path.join(code_dir, fname)
        if not os.path.exists(fpath):
            continue

        if y < MARGIN + 35:
            new_page()

        if os.path.exists(map_path):
            draw_line("--- FILE MAP (YOU ARE HERE) ---", "Helvetica-Bold", HEADER_SIZE)
            for line in map_text.splitlines():
                if line.startswith("Extracted files to be reviewed:"):
                    continue
                if "(Click to clear" in line:
                    continue
                basename = os.path.basename(fname)
                if line.endswith(basename):
                    draw_line(line + "  <-- [YOU ARE HERE]", "Courier-Bold", FONT_SIZE)
                else:
                    draw_line(line, "Courier", FONT_SIZE)
            y -= LEADING

        sep = "=" * 60
        draw_line(sep, "Helvetica-Bold", HEADER_SIZE)
        draw_line(f"FILE: {fname}", "Helvetica-Bold", HEADER_SIZE)
        draw_line(sep, "Helvetica-Bold", HEADER_SIZE)
        y -= 2

        text = _extract_file_text(fpath)
        c.setFont("Courier", FONT_SIZE)
        for line in text.splitlines():
            line = line.replace("\t", "    ")
            if not line.strip():
                y -= LEADING * 0.4
                if y < MARGIN:
                    new_page()
                continue
            wrapped = simpleSplit(line, "Courier", FONT_SIZE, text_w)
            for wline in wrapped:
                draw_line(wline)

        y -= LEADING

    c.save()
    print(f"    ✓ Code PDF compiled: {output_path}")
    return output_path


def stage_01_code_gonzo(combined_pdf, metadata, output_dir):
    """Divergence Hunter: hunts for gaps between paper claims and code implementation."""
    print("  → 01_code_c Divergence Hunter...")
    prompt = load_prompt("prompts/01i_code_gonzo.txt").replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, combined_pdf, model_type="pro_3_1", temperature=0.5,
                         thinking_level="high", system_instruction=load_instruction("researcher.txt"),
                         step="01i_code_gonzo")
    save_output(result, "01i_code_gonzo.txt", output_dir)
    return result


def stage_01_code_gonzo_b(combined_pdf, metadata, output_dir):
    """Bug Hunter: internal technical errors in the code independent of paper claims."""
    print("  → 01_code_c2 Bug Hunter...")
    prompt = load_prompt("prompts/01j_code_gonzo_b.txt").replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, combined_pdf, model_type="pro_3_1", temperature=0.5,
                         thinking_level="high", system_instruction=load_instruction("researcher.txt"),
                         step="01j_code_gonzo_b")
    save_output(result, "01j_code_gonzo_b.txt", output_dir)
    return result


def stage_01_code_gonzo_c(combined_pdf, metadata, output_dir):
    """Data Archaeologist: errors in data construction and cleaning pipelines."""
    print("  → 01_code_c3 Data Archaeologist...")
    prompt = load_prompt("prompts/01k_code_gonzo_c.txt").replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, combined_pdf, model_type="pro_3_1", temperature=0.5,
                         thinking_level="high", system_instruction=load_instruction("researcher.txt"),
                         step="01k_code_gonzo_c")
    save_output(result, "01k_code_gonzo_c.txt", output_dir)
    return result


def stage_01_code_compiler(g1, g2, g3, metadata, output_dir):
    """01l: Consolidate raw findings from three gonzo reviewers into a single deduplicated list."""
    print("  → 01l Code Compiler (Consolidate)...")

    combined_inputs = (str(g1 or "") + str(g2 or "") + str(g3 or "")).strip()
    if not combined_inputs or combined_inputs.lower() in ["none", "n/a", "no issues found", "=null="]:
        print("    ℹ No code findings to consolidate. Skipping API call.")
        save_output("=NULL=", "01l_code_compiler.txt", output_dir)
        return "=NULL="

    prompt = load_prompt("prompts/01l_code_compiler.txt")
    if "[Error:" in prompt:
        raise FileNotFoundError("Prompt file missing: prompts/01l_code_compiler.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{GONZO_1}}", g1 or "None")
    prompt = prompt.replace("{{GONZO_2}}", g2 or "None")
    prompt = prompt.replace("{{GONZO_3}}", g3 or "None")
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)

    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1,
                         system_instruction=load_instruction("bureaucrat.txt"), step="01l_code_compiler")
    save_output(result, "01l_code_compiler.txt", output_dir)
    return result


def stage_01_code_checker(consolidated, metadata, output_dir, code_pdf):
    """01m: Verify each consolidated issue against the actual code."""
    print("  → 01m Code Checker (Verify)...")

    if not consolidated or consolidated.strip() in ["", "=NULL="]:
        print("    ℹ No consolidated issues to check. Skipping API call.")
        save_output("=NULL=", "01m_code_checker.txt", output_dir)
        return "=NULL="

    prompt = load_prompt("prompts/01m_code_checker.txt")
    if "[Error:" in prompt:
        raise FileNotFoundError("Prompt file missing: prompts/01m_code_checker.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{CONSOLIDATED_ISSUES}}", consolidated)
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, code_pdf, model_type="pro_2_5", temperature=0.1, thinking_budget=1000,
                         system_instruction=load_instruction("bureaucrat.txt"), step="01m_code_checker")
    save_output(result, "01m_code_checker.txt", output_dir)
    return result


def stage_01_code_list(consolidated, checker_output, metadata, output_dir):
    """01n: Compile final verified list from consolidated issues + checker verdicts."""
    print("  → 01n Code List (Final)...")

    if not consolidated or consolidated.strip() in ["", "=NULL="]:
        print("    ℹ No consolidated issues — nothing to compile.")
        save_output("=NULL=", "01n_code_list.txt", output_dir)
        return "=NULL="

    prompt = load_prompt("prompts/01n_code_list.txt")
    if "[Error:" in prompt:
        raise FileNotFoundError("Prompt file missing: prompts/01n_code_list.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{CONSOLIDATED_ISSUES}}", consolidated)
    prompt = prompt.replace("{{CHECKER_OUTPUT}}", checker_output or "No checker output available.")
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.5, thinking_level="low",
                         system_instruction=load_instruction("thinker.txt"), step="01n_code_list")
    save_output(result, "01n_code_list.txt", output_dir)
    return result

def stage_04b_data_editor(findings, metadata, output_dir, paper_review_context=None):
    print("  → 04b Data Editor...")
    prompt = load_prompt("prompts/04b_data_editor.txt").replace("{{CITATION}}", get_citation_block(metadata)).replace("{{CODE_CHECK_FINDINGS}}", findings)
    if paper_review_context:
        prompt = prompt.replace("{{PAPER_REVIEW_CONTEXT}}", paper_review_context)
    else:
        prompt = prompt.replace("{{PAPER_REVIEW_CONTEXT}}", "None provided.")
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="04b_data_editor")
    save_output(result, "04b_data_editor.txt", output_dir)
    return result


# ============================================================================
# END CODE ANALYSIS STAGES
# ============================================================================

def stage_01o_summarizer(br1, bu1, sh1, col, alg, math, void, contribs, metadata, output_dir, math_analysis=None):
    print("  → 01o Summarizer...")
    context = f"BREAKER:\n{br1}\nBUTCHER:\n{bu1}\nSHREDDER:\n{sh1}\nCOLLECTOR:\n{col}\nALG:\n{alg}\nMATH:\n{math}\nTHE_VOID:\n{void}"
    if math_analysis:
        context += f"\n\nVERIFIED_MATH_ANALYSIS:\n{math_analysis}"
    prompt = load_prompt("prompts/01o_summarizer.txt").replace("{{INPUT_EVIDENCE}}", context).replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="medium", system_instruction=load_instruction("bureaucrat.txt"), step="01o_summarizer")
    save_output(result, "01o_summarizer.txt", output_dir)
    return result

# ============================================================================
# STAGE 2: THE LIST OF POTENTIAL ISSUES
# ============================================================================

def stage_02a_numbers(pdf_path, issues, metadata, output_dir, equations=None):
    print("  → 02a Numbers...")
    prompt = load_prompt("prompts/02a_numbers.txt").replace("{{POTENTIAL_ISSUES}}", issues).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.3, thinking_level="medium", system_instruction=load_instruction("bureaucrat.txt"), step="02a_numbers")
    save_output(result, "02a_numbers.txt", output_dir)
    return result

def stage_02b_compiler_1(issues, numbers, output_dir):
    print("  → 02b Compiler 1 (Sanitize)...")
    prompt = load_prompt("prompts/02b_compiler_1.txt").replace("{{POTENTIAL_ISSUES}}", issues).replace("{{NUMBER_CHECK}}", numbers)
    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02b_compiler_1")
    save_output(result, "02b_compiler_1.txt", output_dir)
    return result

def stage_02c_blue_team(pdf_path, compiled_issues_v1, contribs, metadata, output_dir):
    print("  → 02c Blue Team...")
    prompt = load_prompt("prompts/02c_blue_team.txt").replace("{{COMPILED_ISSUES_V1}}", compiled_issues_v1).replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="02c_blue_team")
    save_output(result, "02c_blue_team.txt", output_dir)
    return result

def stage_02d_compiler_2(compiled_issues_v1, blue_team, output_dir):
    print("  → 02d Compiler 2 (Add Defence)...")
    prompt = load_prompt("prompts/02d_compiler_2.txt").replace("{{COMPILED_ISSUES_V1}}", compiled_issues_v1).replace("{{BLUE_TEAM}}", blue_team)
    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02d_compiler_2")
    save_output(result, "02d_compiler_2.txt", output_dir)
    return result

def stage_02e_assessment(pdf_path, compiled_issues_v2, metadata, output_dir):
    print("  → 02e Assessment...")
    prompt = load_prompt("prompts/02e_assessment.txt").replace("{{COMPILED_ISSUES_V2}}", compiled_issues_v2).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.4, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02e_assessment")
    save_output(result, "02e_assessment.txt", output_dir)
    return result

def stage_02f_compiler_3(compiled_issues_v2, assessment, output_dir):
    print("  → 02f Compiler 3 (Add Assessment)...")
    prompt = load_prompt("prompts/02f_compiler_3.txt").replace("{{COMPILED_ISSUES_V2}}", compiled_issues_v2).replace("{{ASSESSMENT}}", assessment)
    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02f_compiler_3")
    save_output(result, "02f_compiler_3.txt", output_dir)
    return result

def stage_02g_list_v1(pdf_path, compiled_issues_v3, metadata, output_dir):
    print("  → 02g List V1 (Finalize)...")
    prompt = load_prompt("prompts/02g_list_v1.txt").replace("{{COMPILED_ISSUES_V3}}", compiled_issues_v3).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02g_list_v1")
    save_output(result, "02g_list_v1.txt", output_dir)
    return result

# ============================================================================
# STAGE 3: THE CHECKS & REVISION
# ============================================================================

def stage_03a_checker_1(pdf_path, list_v1, metadata, output_dir, equations=None):
    print("  → 03a Checker 1 (Verification)...")
    prompt = load_prompt("prompts/03a_checker_1.txt").replace("{{POTENTIAL_ISSUES_V1}}", list_v1).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="03a_checker_1")
    save_output(result, "03a_checker_1.txt", output_dir)
    return result

def stage_03b_external(pdf_path, list_v1, metadata, output_dir):
    print("  → 03b External Check...")
    prompt = load_prompt("prompts/03b_external.txt").replace("{{POTENTIAL_ISSUES_V1}}", list_v1).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="03b_external")
    save_output(result, "03b_external.txt", output_dir)
    return result

def stage_03c_list_v2(list_v1, fact_check, ext_check, metadata, output_dir):
    print("  → 03c List V2 (Finalize Dossier)...")
    prompt = load_prompt("prompts/03c_list_v2.txt")
    prompt = prompt.replace("{{POTENTIAL_ISSUES_V1}}", list_v1)
    prompt = prompt.replace("{{FACT_CHECK}}", fact_check)
    prompt = prompt.replace("{{EXTERNAL_CHECK}}", ext_check)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="03c_list_v2")
    save_output(result, "03c_list_v2.txt", output_dir)
    return result

# ============================================================================
# STAGE 4: REVIEWER
# ============================================================================

def stage_04a_reviewer(pdf_path, contribs, list_v2, metadata, output_dir, appendix_text=None):
    print("  → 04a Reviewer...")
    prompt = load_prompt("prompts/04a_reviewer.txt")
    prompt = prompt.replace("{{CONTRIBUTIONS}}", contribs)
    prompt = prompt.replace("{{POTENTIAL_ISSUES_V2}}", list_v2)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))

    if appendix_text:
        note = """

**CRITICAL APPENDIX NOTE:**
An Appendix has also been compiled that will be added to your review.
You will see it referred to as "the Appendix to this review"; this is what that means.
You can also reference this Appendix, **but you must not confuse it with any appendices in the original text.**
        """
        prompt += note

    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.4, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="04a_reviewer")
    save_output(result, "04a_reviewer.txt", output_dir)
    return result

# ============================================================================
# STAGE 5: REVISION (Checkers + Corrector)
# ============================================================================

def stage_05a_checker_2(pdf_path, review, metadata, output_dir, equations=None):
    print("  → 05a Checker 2 (Claims/Logic)...")
    prompt = load_prompt("prompts/05a_checker_2.txt").replace("{{REVIEW}}", review)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="05a_checker_2")
    save_output(result, "05a_checker_2.txt", output_dir)
    return result

def stage_05b_checker_3(pdf_path, review, metadata, output_dir):
    print("  → 05b Checker 3 (Pages/Quotes)...")
    prompt = load_prompt("prompts/05b_checker_3.txt").replace("{{REVIEW}}", review).replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="05b_checker_3")
    save_output(result, "05b_checker_3.txt", output_dir)
    return result

def stage_05c_reviser(review, chk2, chk3, metadata, output_dir):
    print("  → 05c Reviser...")
    prompt = load_prompt("prompts/05c_reviser.txt")
    prompt = prompt.replace("{{REVIEW}}", review)
    prompt = prompt.replace("{{SUMMARY}}", metadata.get("abstract_summary", ""))
    prompt = prompt.replace("{{CHECKER_2}}", chk2).replace("{{CHECKER_3}}", chk3)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="05c_reviser")
    save_output(result, "05c_reviser.txt", output_dir)
    return result

# ============================================================================
# STAGE 6: LEGAL
# ============================================================================

def stage_06_legal(review, metadata, output_dir):
    print("  → 06 Legal...")
    prompt = load_prompt("prompts/06_legal.txt").replace("{{REVIEW_V2}}", review).replace("{{CITATION}}", get_citation_block(metadata))
    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="06_legal")
    save_output(result, "06_legal.txt", output_dir)
    return result

# ============================================================================
# STAGE 7: FORMATTER
# ============================================================================

def stage_07_formatter(review, legal, metadata, output_dir, appendices=None):
    print("  → 07 Formatter...")
    prompt = load_prompt("prompts/07_formatter.txt").replace("{{REVIEW_V2}}", review).replace("{{LEGAL}}", legal).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{PAPER_CONTEXT}}", metadata.get("doc_type", "paper"))

    if appendices:
        prompt += (
            "\n\n═══════════════════════════════════════════════════════════════\n"
            "APPENDICES INSTRUCTION:\n"
            "The following text contains Appendices. You MUST append them to the very end of your output,\n"
            "starting on a new page (\\newpage). Do not edit or summarize them.\n"
            "Preserve their headers (# Appendix X).\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            f"{appendices}"
        )

    prompt = inject_page_numbers(prompt, metadata, is_code_stage=bool(metadata.get("code_dir")))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="07_formatter")
    save_output(result, "07_formatter.txt", output_dir)
    return result

# ============================================================================
# WRITER MODE (Stages 8-9)
# ============================================================================

def stage_08a_alchemist(pdf_path, formatted, metadata, output_dir, writer_mode=True):
    print("  → 08a Alchemist...")
    prompt = load_prompt("prompts/08a_alchemist.txt")
    prompt = prompt.replace("{{FORMATTED_REVIEW}}", formatted)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="08a_alchemist")

    separator = "===COPYEDITOR_INSTRUCTIONS==="
    if separator in result:
        parts = result.split(separator)
        public = parts[0].strip()
        private = parts[1].strip()
    else:
        public = result
        private = "No specific instructions."

    save_output(public, "08a_alchemist.txt", output_dir)
    save_output(private, "08a_alchemist_instructions.txt", output_dir)
    return public, private

def stage_08b_polisher(alchemist_text, metadata, output_dir):
    print("  → 08b Polisher...")
    prompt = load_prompt("prompts/08b_polisher.txt").replace("{{ALCHEMIST}}", alchemist_text).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{PAPER_CONTEXT}}", metadata.get("doc_type", "paper"))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="08b_polisher")
    save_output(result, "08b_polisher.txt", output_dir)
    return result

def stage_09a_proofreader(pdf_path, metadata, output_dir, writer_mode=True):
    print("  → 09a Proofreader...")
    prompt = load_prompt("prompts/09a_proofreader.txt")
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="09a_proofreader")
    save_output(result, "09a_proofreader.txt", output_dir)
    return result

def stage_09b_proofread_clean(raw, metadata, output_dir, writer_mode=True):
    print("  → 09b Proofread Clean...")
    prompt = load_prompt("prompts/09b_proofread_clean.txt").replace("{{RAW_PROOFREAD}}", raw)
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="09b_proofread_clean")
    save_output(result, "09b_proofread_clean.txt", output_dir)
    return result

def stage_09c_copyedit(pdf_path, review, polished_note, inst, metadata, output_dir, writer_mode=True):
    print("  → 09c Copyedit...")
    prompt = load_prompt("prompts/09c_copyedit.txt")
    prompt = prompt.replace("{{ORIGINAL_REVIEW}}", review)
    prompt = prompt.replace("{{EDITOR_NOTE}}", polished_note)
    prompt = prompt.replace("{{SECRET_INSTRUCTIONS}}", inst)
    prompt = inject_page_numbers(prompt, metadata)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="09c_copyedit")
    save_output(result, "09c_copyedit.txt", output_dir)
    return result
