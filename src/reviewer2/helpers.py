"""Shared pipeline utilities: code-zip ingestion, metadata parsing, cost reporting."""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile

from reviewer2.core import call_gemini, load_prompt
from reviewer2.paths import prompts_dir, pricing_csv

CODE_ALLOWED_EXTENSIONS = {
    ".py", ".r", ".do", ".ado", ".rmd", ".ipynb", ".jl", ".m",
    ".sql", ".sh", ".bash", ".txt", ".md", ".rst", ".sas", ".stata",
    ".f90", ".f", ".c", ".cpp", ".h", ".java", ".cs", ".rb", ".pl",
    ".swift", ".ts", ".js", ".lua", ".scala", ".go", ".log",
    ".pdf", ".docx",
}


def save_code_files(file_info_list, target_dir):
    """Save (filename, bytes) tuples into target_dir, recursively unpacking zips."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    def process_entry(filename, content):
        clean_name = filename.replace("\\", "/").strip("/")
        parts = [p for p in clean_name.split("/") if p and p != ".."]
        if not parts:
            return
        safe_path = "/".join(parts)

        if safe_path.lower().endswith(".zip"):
            try:
                prefix = safe_path[: safe_path.rfind("/") + 1] if "/" in safe_path else ""
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    for zinfo in z.infolist():
                        if zinfo.is_dir() or zinfo.filename.startswith("__MACOSX") or zinfo.filename.startswith("."):
                            continue
                        try:
                            z_content = z.read(zinfo.filename)
                            process_entry(prefix + zinfo.filename, z_content)
                        except Exception as e:
                            print(f"  ⚠ Error reading {zinfo.filename} from {safe_path}: {e}")
            except Exception as e:
                print(f"  ⚠ Error opening zip {safe_path}: {e}")
        else:
            ext = os.path.splitext(safe_path.lower())[1]
            if ext in CODE_ALLOWED_EXTENSIONS:
                target_path = os.path.join(target_dir, safe_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(content)

    for fname, content in file_info_list:
        process_entry(fname, content)


def validate_pdf_structure(pdf_path):
    """Quick Flash Lite check that a rendered PDF has all required sections."""
    print("  → Validating PDF Structure (Writer Mode Guard)...")

    prompt = (
        "Analyze this document structure.\n"
        "PART 1: To be valid, it **must** contain this sections:\n"
        "1. Editor's Note\n"
        "2. Summary\n"
        "3. Specific Issues\n"
        "4. Future Research\n"
        "5. Copyediting\n"
        "6. Proofreading\n\n"
        "PART 2: Check for FORBIDDEN CONTENT.\n"
        "Does the document contain the text '===COPYEDITOR_INSTRUCTIONS===' or a section explicitly labeled 'Secret Instructions' or 'Instructions for the Copyeditor'?\n\n"
        "REPLY STRICTLY IN THIS FORMAT:\n"
        "VALID: [YES/NO]\n"
        "MISSING SECTIONS: [None / List them]\n"
        "FORBIDDEN CONTENT FOUND: [YES/NO]"
    )

    try:
        response = call_gemini(
            prompt, pdf_path, model_type="flash_lite", temperature=0.0,
            system_instruction="You are a Quality Control bot. Reply strictly as requested.",
        )
        valid_structure = "VALID: YES" in response.upper()
        forbidden_leak = "FORBIDDEN CONTENT FOUND: YES" in response.upper()

        if forbidden_leak:
            print("  ❌ Validation FAILED: Secret Instructions leaked into PDF.")
            return False, "Secret Instructions leaked into PDF."
        if not valid_structure:
            print(f"  ❌ Validation FAILED. Flash Lite output: {response}")
            return False, response
        print("  ✓ PDF Structure Validated.")
        return True, "OK"
    except Exception as e:
        print(f"  ⚠ Validation Error: {e}")
        return False, str(e)


def load_instruction(filename):
    """Load a system-instruction persona file from prompts/."""
    path = prompts_dir() / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"  ⚠ Warning: System instruction file '{filename}' not found. Using default.")
        return "You are a helpful AI researcher."


def extract_info_fields(text):
    """Parse the 00a_metadata stage output into a structured dict."""
    info = {
        "year": None, "filename": "Report", "citation": None, "url": None,
        "title": None, "authors": None, "abstract_summary": None,
        "key_methodology": None, "research_question": None, "doc_type": None,
        "discipline": None, "title_authors": None, "is_empirical": None,
        "central_argument": None, "page_structure": None,
        "supplement_start_page": None,
        "corresponding_author": None,
        "author_email": None,
    }

    def get_val(pattern):
        match = re.search(r"(?i)\**" + pattern + r"\**:\s*(.+)", text, re.MULTILINE)
        return match.group(1).strip().strip('"') if match else None

    info["year"] = get_val(r"YEAR_OF_PUBLICATION")
    info["title"] = get_val(r"TITLE")
    info["authors"] = get_val(r"AUTHORS")
    info["discipline"] = get_val(r"DISCIPLINE")
    info["citation"] = get_val(r"CITATION")
    info["is_empirical"] = get_val(r"IS_EMPIRICAL")
    info["title_authors"] = get_val(r"TITLE_AUTHORS")
    info["url"] = get_val(r"URL")
    info["abstract_summary"] = get_val(r"ABSTRACT_SUMMARY")
    info["key_methodology"] = get_val(r"KEY_METHODOLOGY")
    info["research_question"] = get_val(r"RESEARCH_QUESTION")
    info["central_argument"] = get_val(r"CENTRAL_ARGUMENT")
    info["doc_type"] = get_val(r"DOCUMENT_TYPE")
    info["contains_algebra"] = get_val(r"CONTAINS_ALGEBRA")
    info["page_structure"] = get_val(r"PAGE_STRUCTURE")
    info["supplement_start_page"] = get_val(r"SUPPLEMENT_START_PAGE")
    info["corresponding_author"] = get_val(r"CORRESPONDING_AUTHOR")
    info["author_email"] = get_val(r"AUTHOR_EMAIL")
    return info


def get_citation_block(metadata):
    return metadata.get("citation") or "Citation not found."


def inject_page_numbers(prompt, metadata, is_code_stage=False):
    """Append the page-numbering rules block to a prompt."""
    p_struct = metadata.get("page_structure")
    if not p_struct or p_struct == "NULL":
        p_struct = "Standard pagination (1, 2, 3...) matching the PDF viewer."

    snippet = f"""

    ## PAGE NUMBER REFERENCE

    **PDF Structure:**
    {p_struct}

    **CRITICAL RULES FOR PAGE CITATIONS:**
    1. ALWAYS cite the PRINTED page numbers visible in the document
    - Main text: Use numeric pages (e.g., "p. 247", "pp. 250-252")
    - Supplements: Use the printed format (e.g., "p. S12", "p. A-5")

    2. NEVER cite PDF viewer page numbers
    - Wrong: "PDF page 35" or "on page 35 of the PDF"
    - Right: "p. 247" or "p. S12"
    """

    if is_code_stage:
        snippet += """
    3. REPLICATION CODE: The replication code section (which follows the paper/appendix) does NOT have printed page numbers.
    - NEVER cite page numbers for code issues.
    - ALWAYS cite the specific **FILE NAME** (e.g., `analysis.do`) and provide a **CODE QUOTE** instead.
    """

    return prompt + snippet


def calculate_cost(usage_log, pricing_path=None):
    """Print a per-stage cost breakdown, return the total in dollars."""
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║            FINAL TOKEN & COST REPORT              ║")
    print("╚═══════════════════════════════════════════════════╝")

    if pricing_path is None:
        pricing_path = pricing_csv()
    pricing_path = str(pricing_path)

    if not os.path.exists(pricing_path):
        print(f"  ⚠ pricing CSV not found at {pricing_path}.")
        return 0.0

    if not usage_log:
        return 0.0

    prices = {}
    with open(pricing_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prices[row["model_name"]] = {
                "input_price_low": float(row["input_price_low"]),
                "output_price_low": float(row["output_price_low"]),
                "input_price_high": float(row["input_price_high"]),
                "output_price_high": float(row["output_price_high"]),
                "cutoff": int(row.get("cutoff") or 128000),
            }

    total_cost = 0.0
    step_costs: dict[str, float] = {}

    print(f"\n{'Step':<30} | {'Model':<25} | {'In Tok':<8} | {'Out Tok':<8} | {'Cost':<10}")
    print("-" * 95)

    for entry in usage_log:
        step = entry.get("step", "unknown")
        model = entry["model_name"]
        in_tok = entry["input_tokens"]
        out_tok = entry["output_tokens"]

        price = prices.get(model)
        if not price:
            print(f"{step:<30} | {model:<25} | {in_tok:<8.0f} | {out_tok:<8.0f} | MISSING")
            continue

        is_high = in_tok > price["cutoff"]
        p_in = price["input_price_high"] if is_high else price["input_price_low"]
        p_out = price["output_price_high"] if is_high else price["output_price_low"]
        c_in = (in_tok / 1e6) * p_in
        c_out = (out_tok / 1e6) * p_out
        entry_cost = c_in + c_out

        total_cost += entry_cost
        step_costs[step] = step_costs.get(step, 0) + entry_cost
        print(f"{step:<30} | {model:<25} | {in_tok:<8.0f} | {out_tok:<8.0f} | ${entry_cost:.4f}")

    print("\n" + "-" * 95)
    print(f"\n{'COST BY STEP SUMMARY':^95}")
    print("-" * 50)

    for step, cost in sorted(step_costs.items(), key=lambda x: x[1], reverse=True):
        pct = (cost / total_cost * 100) if total_cost > 0 else 0
        print(f"  {step:<35} ${cost:.4f}  ({pct:5.1f}%)")

    print("-" * 50)
    print(f"  {'TOTAL ESTIMATED COST':<35} ${total_cost:.4f}\n")
    return total_cost


def sanitize_math_for_latex(text):
    if not text:
        return text
    return re.sub(r"`(\$[^`]+?\$)`", r"\1", text)


def is_output_truncated(text):
    """Ask Flash Lite whether the text cuts off mid-sentence."""
    if not text or len(text) < 100:
        return True

    tail_sample = text[-1000:]
    prompt = f"""
    You are a data integrity checker. Look at the end of this file content.
    It is a list of issues in an academic text.
    Does it appear to end abruptly, in the middle of a sentence, without a full stop?

    Content tail:
    {tail_sample}

    Reply with exactly one word: YES or NO.
    """

    result = call_gemini(prompt, model_type="flash_lite", thinking_budget=0, temperature=0.0)
    return "YES" in result.strip().upper()
