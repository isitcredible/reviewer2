import os
import re
import json
from datetime import datetime
from reviewer2.core import call_gemini, call_model, save_output, load_prompt
from reviewer2.helpers import load_instruction, extract_info_fields, get_citation_block, inject_page_numbers, load_fact_check_resource, split_code_dir, ensure_file_map
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

ESTIMAND_INJECTION = """

═══════════════════════════════════════════════════════════════
**THE TARGET PARAMETER** (this is an empirical paper)

A quantitative claim names a parameter the study means to recover. Before asking whether the estimate is precise, ask whether the design recovers *that* parameter. Other reviewers dissect the estimator's mechanics and what it averages; your job here is the prior question of identification: does this coefficient equal the quantity the authors say it does?

- **Identification:** E.g.: What must hold for the reported coefficient to equal the quantity the authors interpret it as (an average effect, an elasticity, a structural parameter)? Is that argued, or merely assumed by the choice of method?
- **Local vs. claimed:** E.g.: If identification rests on only part of the variation (compliers, a discontinuity, marginal units), is the recovered parameter that *local* effect, or the broader population or policy parameter the conclusions invoke?
- **Right number, wrong parameter:** E.g.: A coefficient can be precise and cleanly identified yet still not be what the question needs: conditioning on something realised after treatment or on a collider (so the controls change what is identified); a partial-equilibrium estimate read as a general-equilibrium claim; a short-horizon effect read as a long-run parameter; an estimand on a transformed scale (logs with many zeros, a ratio) reported as a level or percentage effect.
- **Placebo logic:** E.g.: Pointed at an outcome the theory says should be unmoved, would the same machinery still deliver the headline? If so, it is not identifying what it claims.

Treat a failure here as foundational: if the design cannot recover the target parameter, the precision of the estimate is beside the point.
═══════════════════════════════════════════════════════════════
"""

# ============================================================================
# STAGE 0: EXTRACTION
# ============================================================================

def stage_00a_metadata(pdf_path, output_dir):
    print("  → 00a Metadata (Search Enabled)...")
    prompt = load_prompt("prompts/00a_metadata.txt")
    
    # Enable search to verify citation details
    result = call_gemini(
        prompt, 
        pdf_path, 
        model_type="flash_2_5", 
        temperature=0.0, 
        thinking_budget=0, 
        system_instruction=load_instruction("bureaucrat.txt"), 
        step="00a_metadata",
        use_search=True # <--- ACTIVATE SEARCH
    )
    save_output(result, "00a_metadata.txt", output_dir)
    return result

def stage_00b_metadata_clean(raw, pdf_path, output_dir):
    print("  → 00b Metadata Clean...")
    prompt = load_prompt("prompts/00b_metadata_clean.txt").replace("{{RAW_INPUT}}", raw)
    result = call_model(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="00b_metadata_clean")
    save_output(result, "00b_metadata_clean.txt", output_dir)
    return extract_info_fields(result)

def stage_00b_2_metadata_math(metadata, output_dir):
    print("  → 00b_2 Metadata Math Fixer...")
    target_keys = ["title", "abstract_summary", "key_methodology", "research_question", "central_argument"]
    subset = {k: metadata.get(k) for k in target_keys if metadata.get(k)}
    if not subset: return metadata

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

    result = call_model(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, step="00b_2_metadata_math")

    try:
        clean_result = result.replace("```json", "").replace("```", "").strip()
        fixed_data = json.loads(clean_result)
        for k, v in fixed_data.items():
            if k in metadata and v: metadata[k] = v
        save_output(json.dumps(fixed_data, indent=2), "00b_2_metadata_math.json", output_dir)
    except json.JSONDecodeError:
        print("  ⚠ Warning: Failed to parse JSON from Metadata Math Fixer.")
        save_output(result, "00b_2_metadata_math_FAILED.txt", output_dir)
    return metadata

def stage_00c_contributions(pdf_path, metadata, output_dir):
    print("  → 00c Contributions...")
    prompt = load_prompt("prompts/00c_contributions.txt").replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
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
    # Estimand / identification lens: empirical papers only (mirrors the is_empirical
    # branch in pipeline.py). Non-empirical papers and the Breaker revisits skip it.
    if str(metadata.get("is_empirical", "YES")).upper() == "YES":
        prompt += ESTIMAND_INJECTION
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01a_breaker")
    save_output(result, "01a_breaker.txt", output_dir)
    return result

def stage_01a_2_breaker_revisit(pdf_path, prior_output, metadata, output_dir, round_num=2):
    print(f"  → 01a_{round_num} Breaker Round {round_num} (Non-Empirical Deep Dive)...")

    # 1. Load the original Breaker prompt to get the base instructions
    base_prompt = load_prompt("prompts/01a_breaker.txt")

    # 2. Add the Revisit Instruction
    revisit_instruction = f"""

    ═══════════════════════════════════════════════════════════════
    **ROUND {round_num} INSTRUCTION: DEEP REASONING**
    ═══════════════════════════════════════════════════════════════
    You have already analyzed this text. These were the issues found in the previous round(s):

    {prior_output}

    **YOUR TASK:**
    NOW TRY TO FIND ANOTHER 10 ISSUES.
    THEY MUST BE **DIFFERENT** TO THE ISSUES ALREADY FOUND.

    USE YOUR SOTA REASONING TO DO BETTER.
    DIG DEEPER INTO LOGICAL INCONSISTENCIES, THEORETICAL GAPS, AND ARGUMENTATIVE FLAWS.
    """

    full_prompt = base_prompt + revisit_instruction

    # 3. Inject standard variables (non-empirical path: no estimand block)
    full_prompt = full_prompt.replace("{{CONTRIBUTIONS}}", "(See text)").replace("{{CITATION}}", get_citation_block(metadata))
    full_prompt = inject_page_numbers(full_prompt, metadata)

    # 4. Gemini 2.5 Pro, high temp for divergence (the deep 3.1 pass now runs at Round 1)
    result = call_gemini(
        full_prompt,
        pdf_path,
        model_type="pro_2_5",
        temperature=0.6, # Higher temp for creativity/divergence
        thinking_budget=-1, # 2.5-style dynamic thinking
        system_instruction=load_instruction("researcher.txt"),
        step=f"01a_{round_num}_breaker_revisit"
    )

    save_output(result, f"01a_{round_num}_breaker_revisit.txt", output_dir)
    return result

def stage_01b_butcher(pdf_path, metadata, output_dir, contribs, appendix_text=None):
    print("  → 01b Butcher...")
    prompt = load_prompt("prompts/01b_butcher.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.2, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01b_butcher")
    save_output(result, "01b_butcher.txt", output_dir)
    return result

def stage_01c_shredder(pdf_path, metadata, output_dir, contribs, appendix_text=None):
    print("  → 01c Shredder...")
    prompt = load_prompt("prompts/01c_shredder.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.1, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01c_shredder")
    save_output(result, "01c_shredder.txt", output_dir)
    return result

def stage_01d_collector(pdf_path, b1, s1, metadata, output_dir, appendix_text=None):
    print("  → 01d Collector...")
    context = f"BUTCHER:\n{b1}\nSHREDDER:\n{s1}"
    prompt = load_prompt("prompts/01d_collector.txt").replace("{{INPUT_CONTEXT}}", context).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01d_collector")
    save_output(result, "01d_collector.txt", output_dir)
    return result

def stage_01e_math_extract(pdf_path, contribs, metadata, output_dir, appendix_text=None):
    print("  → 01e Math...")
    prompt = load_prompt("prompts/01e_math.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)

    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="01e_math")
    save_output(result, "01e_math.txt", output_dir)
    return result

EQUATION_REFERENCE = """

═══════════════════════════════════════════════════════════════
EQUATION REFERENCE (from independent OCR extraction)
═══════════════════════════════════════════════════════════════

The following equations were extracted from the PDF by a dedicated OCR system (MathPix), NOT by visual AI parsing. These are your PRIMARY reference for what the equations actually say. When checking equations, read them from this list rather than trying to parse them visually from the PDF. If your visual reading of the PDF disagrees with the MathPix extraction below, TRUST THE MATHPIX VERSION unless you have strong reason to believe the OCR failed (e.g. garbled output, missing equation).

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
    """Extract equations via MathPix (all pages) + Flash (cleanup)."""
    print("  → 01e2 Equation Extraction (MathPix)...")

    # Part A: Count pages and send ALL to MathPix
    try:
        from pypdf import PdfReader
        num_pages = len(PdfReader(pdf_path).pages)
    except Exception:
        num_pages = 100  # safe fallback
    page_ranges = f"1-{num_pages}"
    print(f"    ↳ Sending all {num_pages} pages to MathPix")

    # Part B: MathPix extracts all pages
    mathpix_md = extract_equations_mathpix(pdf_path, page_ranges)
    if mathpix_md:
        # MathPix charges per page, outside the token ledger. Log a synthetic
        # usage entry (1 "input token" = 1 page, priced by the mathpix_pdf_page
        # row in pricing.csv) so the cost report includes the OCR charge.
        from datetime import datetime as _dt
        from reviewer2.core import USAGE_LOG as _usage_log, CURRENT_JOB_UUID as _job_uuid
        _usage_log.append({
            "model_name": "mathpix_pdf_page",
            "input_tokens": num_pages,
            "output_tokens": 0,
            "thoughts_tokens": 0,
            "timestamp": _dt.now().isoformat(),
            "step": "01e2_mathpix_ocr",
            "job_uuid": _job_uuid,
        })
    if not mathpix_md:
        save_output(None, "01e2_equations.txt", output_dir)
        return None

    # Save raw MathPix output before Flash processing
    save_output(mathpix_md, "01e2_mathpix_raw.txt", output_dir)

    # Part C: Flash extracts just the equations
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
        step="01e2_extract"
    )

    save_output(equations, "01e2_equations.txt", output_dir)
    return equations


def stage_01fa_math_check(pdf_path, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fa The Re-Deriver...")
    prompt = load_prompt("prompts/01fa_math_check.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fa_math_check")
    save_output(result, "01fa_math_check.txt", output_dir)
    return result

def stage_01fb_math_proofread(pdf_path, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fb The Proofreader...")
    prompt = load_prompt("prompts/01fb_math_proofread.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fb_math_proofread")
    save_output(result, "01fb_math_proofread.txt", output_dir)
    return result

def stage_01fc_math_audit(pdf_path, proofreader_output, contribs, metadata, output_dir, appendix_text=None, equations=None):
    print("  → 01fc The Auditor...")
    prompt = load_prompt("prompts/01fc_math_audit.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{PROOFREADER_OUTPUT}}", proofreader_output or "=NULL=")
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.1, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fc_math_audit")
    save_output(result, "01fc_math_audit.txt", output_dir)
    return result

MATHPIX_PAPER_INJECTION = """

═══════════════════════════════════════════════════════════════
FULL PAPER TEXT (from MathPix OCR)
═══════════════════════════════════════════════════════════════

The following is the complete text of the paper, extracted by MathPix OCR. All equations (both displayed and inline) are rendered in LaTeX. This is your PRIMARY source for reading equations — it is more reliable than visual PDF parsing. Use this text for all derivation checks, scope verification, and notation verification.

{{MATHPIX_RAW}}
"""

def stage_01fd_math_consolidate(rederiver_output, proofreader_output, auditor_output, metadata, output_dir):
    print("  → 01fd Math Consolidator...")
    prompt = load_prompt("prompts/01fd_math_consolidate.txt").replace("{{CITATION}}", get_citation_block(metadata)).replace("{{REDERIVER_OUTPUT}}", rederiver_output or "=NULL=").replace("{{PROOFREADER_OUTPUT}}", proofreader_output or "=NULL=").replace("{{AUDITOR_OUTPUT}}", auditor_output or "=NULL=")
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.1, thinking_level="medium", system_instruction=load_instruction("bureaucrat.txt"), step="01fd_math_consolidate")
    save_output(result, "01fd_math_consolidate.txt", output_dir)
    return result

def stage_01fe_math_blue(pdf_path, math_findings, contribs, metadata, output_dir, equations=None, mathpix_raw=None):
    print("  → 01fe Math Blue Team (defender)...")
    prompt = load_prompt("prompts/01fe_math_blue.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata)).replace("{{MATH_FINDINGS}}", math_findings)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    if mathpix_raw:
        prompt += MATHPIX_PAPER_INJECTION.replace("{{MATHPIX_RAW}}", mathpix_raw)
    result = call_gemini(prompt, None if mathpix_raw else pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="high", system_instruction=load_instruction("researcher.txt"), step="01fe_math_blue", use_code_exec=True)
    save_output(result, "01fe_math_blue.txt", output_dir)
    return result

def stage_01ff_math_assess(math_findings, blue_defence, metadata, output_dir, equations=None, mathpix_raw=None):
    print("  → 01ff Math Assessment (verdict)...")
    prompt = load_prompt("prompts/01ff_math_assess.txt").replace("{{CITATION}}", get_citation_block(metadata)).replace("{{MATH_FINDINGS}}", math_findings).replace("{{BLUE_DEFENCE}}", blue_defence)
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)
    if mathpix_raw:
        prompt += MATHPIX_PAPER_INJECTION.replace("{{MATHPIX_RAW}}", mathpix_raw)
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="01ff_math_assess")
    save_output(result, "01ff_math_assess.txt", output_dir)
    return result

def stage_01fg_math_list(assessment, output_dir):
    print("  → 01fg Math List (write up confirmed)...")
    prompt = load_prompt("prompts/01fg_math_list.txt").replace("{{ASSESSMENT}}", assessment)
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="01fg_math_list")
    save_output(result, "01fg_math_list.txt", output_dir)
    return result

def stage_01g_the_void(pdf_path, contribs, metadata, output_dir, appendix_text=None):
    print("  → 01g The Void...")
    prompt = load_prompt("prompts/01g_the_void.txt").replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    if appendix_text: prompt += APPENDIX_INJECTION.replace("{{APPENDIX_CONTENT}}", appendix_text)
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_2_5", temperature=0.3, thinking_budget=-1, system_instruction=load_instruction("researcher.txt"), step="01g_the_void")
    save_output(result, "01g_the_void.txt", output_dir)
    return result

# ============================================================================
# CODE ANALYSIS STAGES (optional)
# ============================================================================

_BINARY_EXTENSIONS = {'.dta', '.rdata', '.rds', '.xlsx', '.xls', '.zip',
                      '.gz', '.tar', '.7z', '.pkl', '.feather', '.parquet',
                      '.sav', '.por'}

def _extract_file_text(file_path):
    """Extract readable text from a code or document file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return f"[Could not extract PDF text: {e}]"

    if ext == '.docx':
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[Could not extract Word document text: {e}]"

    if ext in _BINARY_EXTENSIONS:
        return f"[Binary file — content not extractable: {os.path.basename(file_path)}]"

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"[Could not read file: {e}]"


def compile_code_to_pdf(code_dir, ordered_files, output_path):
    """Compile ordered code files into a dense single PDF."""
    print("  → 01_code_b Compiling code to PDF...")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import simpleSplit

    FONT_SIZE = 7
    LEADING = 8.5
    MARGIN = 20  # points (~7mm)
    HEADER_SIZE = 8
    page_w, page_h = A4
    text_w = page_w - 2 * MARGIN

    # Guarantee a package map exists. No-op if one is already present.
    ensure_file_map(code_dir)

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

    # Render File Map at the start if exists
    map_path = os.path.join(code_dir, "_file_map.txt")
    if os.path.exists(map_path):
        map_text = _extract_file_text(map_path)
        draw_line("=== REPLICATION PACKAGE FILE MAP ===", "Helvetica-Bold", HEADER_SIZE)
        y -= 2
        for line in map_text.splitlines():
            if line.startswith("Extracted files to be reviewed:"): continue
            if "(Click to clear" in line: continue
            draw_line(line, "Courier", FONT_SIZE)
        y -= LEADING * 2

    for fname in ordered_files:
        # PDFs are appended to the combined review PDF as native pages, not flattened
        # into this text dump; the file-list builder excludes them, but guard anyway.
        if fname.lower().endswith('.pdf'):
            continue
        fpath = os.path.join(code_dir, fname)
        if not os.path.exists(fpath):
            continue

        if y < MARGIN + 35:
            new_page()
            
        # Draw "You Are Here" Map
        if os.path.exists(map_path):
            draw_line("--- FILE MAP (YOU ARE HERE) ---", "Helvetica-Bold", HEADER_SIZE)
            for line in map_text.splitlines():
                if line.startswith("Extracted files to be reviewed:"): continue
                if "(Click to clear" in line: continue
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
            line = line.replace('\t', '    ')
            if not line.strip():
                y -= LEADING * 0.4
                if y < MARGIN:
                    new_page()
                continue
            # Split long lines to fit page width
            wrapped = simpleSplit(line, "Courier", FONT_SIZE, text_w)
            for wline in wrapped:
                draw_line(wline)

        y -= LEADING  # gap between files

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
    """Bug Hunter: hunts for internal technical errors in the code independent of paper claims."""
    print("  → 01_code_c2 Bug Hunter...")
    prompt = load_prompt("prompts/01j_code_gonzo_b.txt").replace("{{CITATION}}", get_citation_block(metadata))
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, combined_pdf, model_type="pro_3_1", temperature=0.5,
                         thinking_level="high", system_instruction=load_instruction("researcher.txt"),
                         step="01j_code_gonzo_b")
    save_output(result, "01j_code_gonzo_b.txt", output_dir)
    return result


def stage_01_code_gonzo_c(combined_pdf, metadata, output_dir):
    """Data Archaeologist: hunts for errors in data construction and cleaning pipelines."""
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

    # Safety Check: If all inputs are empty or common "no issues" strings, return =NULL=
    combined_inputs = (str(g1 or "") + str(g2 or "") + str(g3 or "")).strip()
    if not combined_inputs or combined_inputs.lower() in ["none", "n/a", "no issues found", "=null="]:
        print("    ℹ No code findings to consolidate. Skipping API call.")
        save_output("=NULL=", "01l_code_compiler.txt", output_dir)
        return "=NULL="

    prompt = load_prompt("prompts/01l_code_compiler.txt")
    if "[Error:" in prompt:
        raise FileNotFoundError(f"Prompt file missing: prompts/01l_code_compiler.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{GONZO_1}}", g1 or "None")
    prompt = prompt.replace("{{GONZO_2}}", g2 or "None")
    prompt = prompt.replace("{{GONZO_3}}", g3 or "None")
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    
    result = call_gemini(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1,
                         system_instruction=load_instruction("bureaucrat.txt"), step="01l_code_compiler")
    save_output(result, "01l_code_compiler.txt", output_dir)
    return result


def _norm_code(s):
    """Whitespace-insensitive normalization for matching quoted code to source."""
    return re.sub(r"\s+", " ", s.strip())


def _quote_signature_lines(quote):
    """Distinctive lines from a quoted code block — long enough and containing
    alphanumerics to be a real fingerprint (drops braces, ellipses, blanks)."""
    sigs = []
    for ln in quote.splitlines():
        s = ln.strip()
        if not s or s.startswith("```"):
            continue
        if set(s) <= set(". "):  # an ellipsis / dotted separator line
            continue
        n = _norm_code(ln)
        if len(n) >= 8 and re.search(r"[A-Za-z0-9]", n):
            sigs.append(n)
    return sigs


def _file_match_score(sigs, file_norm_text):
    """(ratio of signature lines present, has a long anchor line matched)."""
    if not sigs:
        return (0.0, False)
    matched = [s for s in sigs if s in file_norm_text]
    return (len(matched) / len(sigs), any(len(s) >= 15 for s in matched))


def _is_verified(ratio, anchor):
    return (ratio >= 0.5 and anchor) or ratio >= 0.8


def verify_and_relocate_code_findings(checker_output, code_dir, target_file):
    """Deterministic gate between 01m (Grounder) and 01n (writer).

    For each finding that makes a POSITIVE code citation (a cited file + a
    verbatim code quote), confirm the quoted code actually appears in that file.
    - Present in the cited file  -> keep (verified).
    - Present in a DIFFERENT file -> correct the attribution (relocate) and keep.
    - Present in NO replication file -> drop as unverifiable (kills fabrications).
    Findings with no code quote (e.g. 'missing data-construction code') are left
    untouched — there is nothing to verify. The LLM does not police itself here;
    string matching against the actual source does. Returns (output, stats)."""
    empty = {"verified": 0, "relocated": 0, "dropped": 0, "skipped": 0}
    if not checker_output or checker_output.strip() in ("", "=NULL="):
        return checker_output, empty
    if not code_dir or not os.path.exists(code_dir):
        return checker_output, empty

    # Map basename -> (relpath, normalized text) for every code file.
    try:
        ordered, _pkg = split_code_dir(code_dir, target_file)
    except Exception:
        ordered = [os.path.relpath(os.path.join(r, fn), code_dir)
                   for r, _d, fs in os.walk(code_dir) for fn in fs]
    files_norm = {}
    for rel in ordered:
        if rel.lower().endswith(".pdf"):
            continue
        try:
            files_norm[os.path.basename(rel)] = (rel, _norm_code(_extract_file_text(os.path.join(code_dir, rel))))
        except Exception:
            continue
    if not files_norm:
        return checker_output, empty

    stats = dict(empty)
    blocks = re.split(r"(?m)(?=^FINDING:)", checker_output)
    out = []
    for block in blocks:
        if not block.lstrip().startswith("FINDING:"):
            out.append(block)
            continue
        disp_m = re.search(r"(?m)^DISPOSITION:\s*(\w+)", block)
        if not disp_m or disp_m.group(1).upper() not in ("CHECK", "RELOCATE"):
            out.append(block)
            continue
        q_m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
        sigs = _quote_signature_lines(q_m.group(1)) if q_m else []
        if not sigs:                       # absence findings etc. — nothing to verify
            stats["skipped"] += 1
            out.append(block)
            continue
        cited = re.findall(r"`([^`]*\.(?:do|ado|m|R|r|py|jl|sas|f90))`", block)
        cited_base = os.path.basename(cited[0]) if cited else None

        if cited_base and cited_base in files_norm and _is_verified(*_file_match_score(sigs, files_norm[cited_base][1])):
            stats["verified"] += 1
            out.append(block)
            continue

        # Hunt for the quote across every file; relocate to the best real match.
        best, best_score = None, 0.0
        for base, (rel, ft) in files_norm.items():
            ratio, anchor = _file_match_score(sigs, ft)
            if _is_verified(ratio, anchor) and (ratio + (0.5 if anchor else 0)) > best_score:
                best, best_score = (base, rel), ratio + (0.5 if anchor else 0)
        if best:
            nb = re.sub(re.escape(cited_base), best[0], block) if cited_base else block
            nb = re.sub(r"(?m)^NOTE:", f"NOTE: [location corrected by verification to {best[1]}] ", nb, count=1)
            stats["relocated"] += 1
            out.append(nb)
        else:
            nb = re.sub(r"(?m)^DISPOSITION:\s*\w+", "DISPOSITION: DROP", block, count=1)
            nb = re.sub(r"(?m)^NOTE:", "NOTE: [dropped by verification: quoted code not found in any replication file] ", nb, count=1)
            stats["dropped"] += 1
            out.append(nb)
    return "".join(out), stats


def stage_01_code_checker(consolidated, metadata, output_dir, code_pdf):
    """01m: Verify each consolidated issue against the actual code."""
    print("  → 01m Code Checker (Verify)...")

    if not consolidated or consolidated.strip() in ["", "=NULL="]:
        print("    ℹ No consolidated issues to check. Skipping API call.")
        save_output("=NULL=", "01m_code_checker.txt", output_dir)
        return "=NULL="

    prompt = load_prompt("prompts/01m_code_checker.txt")
    if "[Error:" in prompt:
        raise FileNotFoundError(f"Prompt file missing: prompts/01m_code_checker.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{CONSOLIDATED_ISSUES}}", consolidated)
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    result = call_gemini(prompt, code_pdf, model_type="pro_3_1", temperature=0.1, thinking_level="high",
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
        raise FileNotFoundError(f"Prompt file missing: prompts/01n_code_list.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{CONSOLIDATED_ISSUES}}", consolidated)
    prompt = prompt.replace("{{CHECKER_OUTPUT}}", checker_output or "No checker output available.")
    
    # INJECT PAGE NUMBERS
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
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=True)
    
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="04b_data_editor")
    save_output(result, "04b_data_editor.txt", output_dir)
    return result


# ============================================================================
# END CODE ANALYSIS STAGES
# ============================================================================

def stage_01o_summarizer(br1, bu1, sh1, col, alg, math, void, contribs, metadata, output_dir, math_analysis=None):
    print("  → 01o Summarizer...")
    # No PDF input, no injection needed
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

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.3, thinking_level="medium", system_instruction=load_instruction("bureaucrat.txt"), step="02a_numbers")
    save_output(result, "02a_numbers.txt", output_dir)
    return result

def stage_02b_compiler_1(issues, numbers, output_dir):
    print("  → 02b Compiler 1 (Sanitize)...")
    prompt = load_prompt("prompts/02b_compiler_1.txt").replace("{{POTENTIAL_ISSUES}}", issues).replace("{{NUMBER_CHECK}}", numbers)
    result = call_model(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02b_compiler_1")
    save_output(result, "02b_compiler_1.txt", output_dir)
    return result

def stage_02c_blue_team(pdf_path, compiled_issues_v1, contribs, metadata, output_dir):
    print("  → 02c Blue Team...")
    prompt = load_prompt("prompts/02c_blue_team.txt").replace("{{COMPILED_ISSUES_V1}}", compiled_issues_v1).replace("{{CONTRIBUTIONS}}", contribs).replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02c_blue_team")
    save_output(result, "02c_blue_team.txt", output_dir)
    return result

def stage_02d_compiler_2(compiled_issues_v1, blue_team, output_dir):
    print("  → 02d Compiler 2 (Add Defence)...")
    prompt = load_prompt("prompts/02d_compiler_2.txt").replace("{{COMPILED_ISSUES_V1}}", compiled_issues_v1).replace("{{BLUE_TEAM}}", blue_team)
    result = call_model(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02d_compiler_2")
    save_output(result, "02d_compiler_2.txt", output_dir)
    return result

def stage_02e_assessment(pdf_path, compiled_issues_v2, metadata, output_dir):
    print("  → 02e Assessment...")
    prompt = load_prompt("prompts/02e_assessment.txt").replace("{{COMPILED_ISSUES_V2}}", compiled_issues_v2).replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.4, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02e_assessment")
    save_output(result, "02e_assessment.txt", output_dir)
    return result

def stage_02f_compiler_3(compiled_issues_v2, assessment, output_dir):
    print("  → 02f Compiler 3 (Add Assessment)...")
    prompt = load_prompt("prompts/02f_compiler_3.txt").replace("{{COMPILED_ISSUES_V2}}", compiled_issues_v2).replace("{{ASSESSMENT}}", assessment)
    result = call_model(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="02f_compiler_3")
    save_output(result, "02f_compiler_3.txt", output_dir)
    return result

def stage_02g_list_v1(pdf_path, compiled_issues_v3, metadata, output_dir):
    print("  → 02g List V1 (Finalize)...")
    prompt = load_prompt("prompts/02g_list_v1.txt").replace("{{COMPILED_ISSUES_V3}}", compiled_issues_v3).replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02g_list_v1")
    save_output(result, "02g_list_v1.txt", output_dir)
    return result

# ============================================================================
# STAGE 2H/2I: PROPORTIONALITY (DEPTH) + LIST V2
# ============================================================================

def stage_02h_proportionality(pdf_path, list_v1, contribs, metadata, output_dir):
    """Proportionality / depth pass. Reweights and deepens the v1 list: demotes
    deflectable cheap shots, pulls up and deepens the load-bearing issues,
    re-orders. Open posture — no prescribed critique shape. Firewall:
    reweight/reorder/deepen/merge/delete only; no new attack lines, no fabricated
    facts; new depth framed as scrutiny, not verdict. Output is the rewritten
    list; 02i finalizes it into list_v2."""
    print("  → 02h Proportionality (Depth)...")
    prompt = load_prompt("prompts/02h_proportionality_1.txt")
    prompt = prompt.replace("{{POTENTIAL_ISSUES_V1}}", list_v1)
    prompt = prompt.replace("{{CONTRIBUTIONS}}", contribs or "")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.4, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="02h_proportionality_1")
    save_output(result, "02h_proportionality_1.txt", output_dir)
    return result

def stage_02i_list_v2(pdf_path, proportioned_list, metadata, output_dir):
    """Finalize the proportioned list into list_v2. Light guardrail pass:
    enforces the firewall (strips anything fabricated/ungrounded against the PDF),
    enforces hedging, renders the canonical list format. Preserves 02h's substance
    and ordering — does NOT re-litigate, re-order, or trim depth. The proportioned
    list fills the {{POTENTIAL_ISSUES_V1}} slot in the prompt."""
    print("  → 02i List V2 (Finalize)...")
    prompt = load_prompt("prompts/02i_list_v2.txt")
    prompt = prompt.replace("{{POTENTIAL_ISSUES_V1}}", proportioned_list)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="02i_list_v2")
    save_output(result, "02i_list_v2.txt", output_dir)
    return result

# ============================================================================
# STAGE 3: THE CHECKS & REVISION
# ============================================================================

def stage_03a_checker_1(pdf_path, list_v1, metadata, output_dir, equations=None):
    print("  → 03a Checker 1 (Verification)...")
    prompt = load_prompt("prompts/03a_checker_1.txt").replace("{{POTENTIAL_ISSUES_V1}}", list_v1).replace("{{CITATION}}", get_citation_block(metadata))

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="03a_checker_1")
    save_output(result, "03a_checker_1.txt", output_dir)
    return result

def stage_03b_external(pdf_path, list_v1, metadata, output_dir):
    print("  → 03b External Check...")
    prompt = load_prompt("prompts/03b_external.txt").replace("{{POTENTIAL_ISSUES_V1}}", list_v1).replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="03b_external")
    save_output(result, "03b_external.txt", output_dir) 
    return result

def stage_03c_list_v3(list_v2, fact_check, ext_check, metadata, output_dir):
    # list_v2 is produced upstream at 02i, so this stage's fact-checked output
    # is list_v3. The prompt's {{POTENTIAL_ISSUES_V1}} slot takes list_v2.
    print("  → 03c List V3 (Finalize Dossier)...")
    prompt = load_prompt("prompts/03c_list_v3.txt")
    prompt = prompt.replace("{{POTENTIAL_ISSUES_V1}}", list_v2)
    prompt = prompt.replace("{{FACT_CHECK}}", fact_check)
    prompt = prompt.replace("{{EXTERNAL_CHECK}}", ext_check)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="03c_list_v3")
    save_output(result, "03c_list_v3.txt", output_dir)
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
    
    # INJECT PAGE NUMBERS
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

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    prompt = _inject_equations(prompt, equations)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="low", system_instruction=load_instruction("thinker.txt"), step="05a_checker_2")
    save_output(result, "05a_checker_2.txt", output_dir)
    return result

def stage_05b_checker_3(pdf_path, review, metadata, output_dir):
    print("  → 05b Checker 3 (Pages/Quotes)...")
    prompt = load_prompt("prompts/05b_checker_3.txt").replace("{{REVIEW}}", review).replace("{{CITATION}}", get_citation_block(metadata))
    
    # INJECT PAGE NUMBERS
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
    result = call_model(prompt, None, model_type="flash_2_5", temperature=0.0, thinking_budget=-1, system_instruction=load_instruction("bureaucrat.txt"), step="06_legal")
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
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata, is_code_stage=bool(metadata.get('code_dir')))
    
    result = call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3, thinking_level="low", system_instruction=load_instruction("bureaucrat.txt"), step="07_formatter")
    save_output(result, "07_formatter.txt", output_dir)
    return result

# ============================================================================
# WRITER MODE (Stages 8-9)
# ============================================================================

# Header + guidance wrapped around the 08_math_advisor output before it is injected
# into the alchemist. Lives here (not in 08a's prompt file) so non-math writer jobs
# inject an empty string and 08a's behaviour is completely unchanged for them.
MATH_ADVICE_INJECTION = """

═══════════════════════════════════════════════════════════════
MATH REMEDIATION ADVICE (specialist input)
═══════════════════════════════════════════════════════════════

A specialist mathematical advisor has prepared concrete remediation routes for the
verified mathematical errors in this paper. Treat this as expert input to fold into
your revision plan: where it identifies a correctable slip, steer the author to the
fix; where it marks an error as load-bearing and unrepairable, let that calibrate how
ambitious the principal plan can be. Do not contradict it on the mathematics, and do
not reproduce it as a list — integrate its substance into your prose advice.

{{MATH_ADVICE_BODY}}
"""

def stage_08_math_advisor(pdf_path, math_findings, contribs, metadata, output_dir):
    """Author-facing stages + math only. Reads the VERIFIED math findings (01ff) and,
    for each confirmed error, advises the author how to remediate it (correct /
    weaken assumption / relegate / concede), calibrated to how load-bearing the
    result is. Feeds 08a as an optional math_advice param. Does NOT hunt for new
    errors or re-litigate the verified ones. Always runs on Gemini Pro."""
    findings = (math_findings or "").strip()
    if not findings or "=NULL=" in findings or "VERDICT: CONFIRMED" not in findings:
        print("  → 08 Math Advisor (no confirmed math errors — skipping)...")
        save_output("=NULL=", "08_math_advisor.txt", output_dir)
        return "=NULL="
    print("  → 08 Math Advisor...")
    prompt = load_prompt("prompts/08_math_advisor.txt")
    prompt = prompt.replace("{{MATH_FINDINGS}}", findings)
    prompt = prompt.replace("{{CONTRIBUTIONS}}", contribs or "")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)

    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.3, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="08_math_advisor")
    save_output(result, "08_math_advisor.txt", output_dir)
    return result

def stage_08a_alchemist(pdf_path, formatted, metadata, output_dir, writer_mode=True, math_advice=None):
    print("  → 08a Alchemist...")
    prompt = load_prompt("prompts/08a_alchemist.txt")
    prompt = prompt.replace("{{FORMATTED_REVIEW}}", formatted)
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))

    # Specialist math remediation advice from 08_math_advisor (writer + do_math only).
    # Empty string for every other job, so the {{MATH_ADVICE}} slot vanishes and 08a's
    # behaviour is unchanged. Only inject when there is real advice (not =NULL=).
    advice = (math_advice or "").strip()
    if advice and "=NULL=" not in advice:
        block = MATH_ADVICE_INJECTION.replace("{{MATH_ADVICE_BODY}}", advice)
    else:
        block = ""
    prompt = prompt.replace("{{MATH_ADVICE}}", block)

    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.5, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="08a_alchemist")
    
    separator = "===COPYEDITOR_INSTRUCTIONS==="
    if separator in result:
        parts = result.split(separator)
        public = parts[0].strip(); private = parts[1].strip()
    else: public = result; private = "No specific instructions."
    
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
    
    # INJECT PAGE NUMBERS
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
    
    # INJECT PAGE NUMBERS
    prompt = inject_page_numbers(prompt, metadata)
    
    result = call_gemini(prompt, pdf_path, model_type="pro_3_1", temperature=0.2, thinking_level="high", system_instruction=load_instruction("thinker.txt"), step="09c_copyedit")
    save_output(result, "09c_copyedit.txt", output_dir)
    return result