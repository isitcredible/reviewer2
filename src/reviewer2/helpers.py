"""Shared helpers: replication-package handling, citations, page numbers, cost report."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re

from pypdf import PdfReader

from reviewer2.core import load_prompt
from reviewer2.paths import prompt_path, pricing_csv


CODE_ALLOWED_EXTENSIONS = {
    '.py', '.r', '.do', '.ado', '.rmd', '.ipynb', '.jl', '.m',
    '.sql', '.sh', '.bash', '.txt', '.md', '.rst', '.sas', '.stata',
    '.f90', '.f', '.c', '.cpp', '.h', '.java', '.cs', '.rb', '.pl',
    '.swift', '.ts', '.js', '.lua', '.scala', '.go',
    '.pdf', '.docx'
}
# Data/output extensions are deliberately excluded (.csv, .log, .dta, .rds, .xlsx,
# .parquet, .shp, …): they are data or run-output, not code to review, and
# compiling them as code pages wastes tokens and can trip the combined-size limit. ".txt" stays
# allowed but is content-sniffed (see is_code_file / _looks_like_data) so code or
# readmes shipped as .txt are kept while tabular data dumps saved as .txt are skipped.


def _looks_like_data(text):
    """True if `text` looks like a delimited DATA dump (rows of numbers/fields)
    rather than code or prose. Keeps data files that happen to carry a .txt
    extension out of the code review. Errs toward
    keeping: only clearly tabular-numeric content with little code structure is
    classified as data."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 25:
        return False  # small files (readmes, notes, short scripts) are always kept
    sample = lines[:300]
    code_markers = ('=', '{', '}', '<-', '(', ')', ';', 'def ', 'function ',
                    'import ', 'class ', 'return', '//', '/*', '#')
    structured = sum(1 for ln in sample if any(m in ln for m in code_markers))
    if structured / len(sample) > 0.20:
        return False  # has code/config structure → keep
    delim = re.compile(r'[,\t;|]+|\s{2,}| +')
    numeric = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?%?$')
    # A field is data-like if it is numeric (quotes stripped first: "002" is a
    # value, not prose) or a fragment of a quoted CSV string — the delimiter
    # split breaks quoted fields like "Springfield, IL" into quote-edged
    # pieces, which the numeric test alone misses, letting text-heavy tables
    # (crosswalks, codebook rows) through as "code".
    def _datalike(f):
        return bool(numeric.match(f.strip('"\''))) or f.startswith('"') or f.endswith('"')
    field_counts, numeric_rows = [], 0
    for ln in sample:
        fields = [f for f in delim.split(ln.strip()) if f]
        if len(fields) < 2:
            continue
        field_counts.append(len(fields))
        if sum(1 for f in fields if _datalike(f)) / len(fields) >= 0.6:
            numeric_rows += 1
    if len(field_counts) < 0.5 * len(sample):
        return False  # most lines aren't tabular → prose/code
    frac_numeric = numeric_rows / len(field_counts)
    med = sorted(field_counts)[len(field_counts) // 2]
    consistent = sum(1 for c in field_counts if abs(c - med) <= 1) / len(field_counts)
    return frac_numeric >= 0.6 and med >= 2 and consistent >= 0.75


def _decode(content):
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None


def is_code_file(filename, content=None, path=None):
    """Whether a replication-package file is CODE/text we review, vs DATA/output
    we skip. Extension-gated against CODE_ALLOWED_EXTENSIONS; a .txt is
    additionally content-sniffed. Pass `content` (bytes) or `path` (on disk)."""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in CODE_ALLOWED_EXTENSIONS:
        return False
    if ext == '.txt':
        text = None
        if content is not None:
            text = _decode(content)
        elif path:
            try:
                with open(path, "rb") as fh:
                    text = _decode(fh.read())
            except Exception:
                text = None
        if text is not None and _looks_like_data(text):
            return False
    return True


def _pdf_page_count_path(path):
    try:
        return len(PdfReader(path).pages)
    except Exception:
        return 0


def _docx_text_path(path):
    try:
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except Exception:
        return ""


def _pdf_fingerprint_bytes(content):
    """(page_count, normalized first-page-text prefix) for duplicate detection."""
    try:
        r = PdfReader(io.BytesIO(content))
        first = (r.pages[0].extract_text() or "") if r.pages else ""
        return (len(r.pages), " ".join(first.split())[:500])
    except Exception:
        return (0, "")


def _main_pdf_matchers(main_pdf_bytes=None, main_pdf_path=None):
    """(md5, fingerprint) for the main-text PDF, or None if unavailable."""
    b = main_pdf_bytes
    if b is None and main_pdf_path and os.path.exists(main_pdf_path):
        try:
            with open(main_pdf_path, "rb") as f:
                b = f.read()
        except Exception:
            b = None
    if b is None:
        return None
    try:
        return (hashlib.md5(b).hexdigest(), _pdf_fingerprint_bytes(b))
    except Exception:
        return None


def _is_main_dup(content, matchers):
    """A package PDF duplicates the main text if byte-identical, OR same page count
    AND identical normalized first-page text (catches a re-exported copy)."""
    if not matchers:
        return False
    main_md5, main_fp = matchers
    try:
        if hashlib.md5(content).hexdigest() == main_md5:
            return True
    except Exception:
        pass
    cand_fp = _pdf_fingerprint_bytes(content)
    return bool(main_fp[0]) and cand_fp[0] == main_fp[0] and bool(main_fp[1]) and cand_fp[1] == main_fp[1]


def split_code_dir(code_dir, main_pdf_path=None):
    """Walk code_dir (applying is_code_file) and split into (code_files, pdf_files),
    both as sorted relative paths. code_files are non-PDF code/text/docx compiled to
    the code PDF as text; pdf_files are PDFs appended to the combined PDF as native
    pages. PDFs that duplicate the main-text PDF are dropped."""
    matchers = _main_pdf_matchers(main_pdf_path=main_pdf_path)
    code_files, pdf_files = [], []
    for root, _dirs, files in os.walk(code_dir):
        for f in files:
            if f.startswith('.') or f == "_file_map.txt":
                continue
            fpath = os.path.join(root, f)
            if not is_code_file(f, path=fpath):
                continue
            rel_dir = os.path.relpath(root, code_dir)
            rel = f if rel_dir == '.' else os.path.join(rel_dir, f)
            if f.lower().endswith('.pdf'):
                try:
                    with open(fpath, "rb") as fh:
                        cb = fh.read()
                    if _is_main_dup(cb, matchers):
                        print(f"  ↳ skipping {rel} (duplicate of the main text)")
                        continue
                except Exception:
                    pass
                pdf_files.append(rel)
            else:
                code_files.append(rel)
    code_files.sort(); pdf_files.sort()
    return code_files, pdf_files


def generate_tree_string(paths):
    """Render a list of '/'-relative paths as an indented file tree. Each file
    line ends with the bare basename, which the compile_code_to_pdf renderer
    relies on for its '[YOU ARE HERE]' per-file anchoring."""
    if not paths:
        return ""
    paths = sorted(paths)
    tree_lines = []
    current_path = []
    for p in paths:
        parts = p.split('/')
        name = parts.pop()
        i = 0
        while i < len(current_path) and i < len(parts) and current_path[i] == parts[i]:
            i += 1
        while len(current_path) > i:
            current_path.pop()
        while i < len(parts):
            tree_lines.append('  ' * i + '+-- ' + parts[i] + '/')
            current_path.append(parts[i])
            i += 1
        tree_lines.append('  ' * len(current_path) + '|-- ' + name)
    return "\n".join(tree_lines)


def ensure_file_map(code_dir):
    """Guarantee a `_file_map.txt` exists in code_dir before the package is
    compiled. Without one, the compiler's package map and per-file 'YOU ARE
    HERE' anchors silently vanish. Synthesise one by walking code_dir for
    everything included (data files too, so the model can see what exists even
    when it isn't dumped). Never overwrites an existing map."""
    if not code_dir or not os.path.isdir(code_dir):
        return None
    map_path = os.path.join(code_dir, "_file_map.txt")
    if os.path.exists(map_path):
        return map_path
    rels = []
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != "__pycache__"]
        for f in files:
            if f.startswith('.') or f == "_file_map.txt":
                continue
            rel = os.path.relpath(os.path.join(root, f), code_dir)
            rels.append(rel.replace(os.sep, '/'))
    if not rels:
        return None
    tree = generate_tree_string(rels)
    try:
        with open(map_path, "w", encoding="utf-8") as fh:
            fh.write(tree)
    except Exception as e:
        print(f"  ⚠ Could not write synthesised file map: {e}")
        return None
    print(f"  → Synthesised replication package file map ({len(rels)} files).")
    return map_path



# Generic publication-status descriptors the metadata stage sometimes drops into
# the journal/venue slot of a citation and italicises as if they were a journal
# title (an unpublished / working paper has no journal). They are NOT journal
# names, so the italics are wrong.
_NON_JOURNAL_DESCRIPTORS = {
    "unpublished paper", "unpublished manuscript", "unpublished ms", "unpublished",
    "mimeo", "mimeograph", "working paper", "discussion paper", "manuscript",
    "typescript", "photocopy", "preprint", "in preparation", "unpublished data",
}


def load_instruction(filename):
    """Loads system instruction from the prompts/ folder."""
    path = prompt_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"  ⚠ Warning: System instruction file '{filename}' not found. Using default.")
        return "You are a helpful AI researcher."


def _sanitize_citation(citation):
    """Unwrap markdown italics around a non-journal descriptor in a citation, so
    'Unpublished paper' etc. no longer prints as a fake italic journal title.
    Genuine journal / book titles stay italicised (only exact descriptor spans
    are unwrapped)."""
    if not citation:
        return citation

    def _unwrap(m):
        inner = m.group(1).strip()
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", inner.lower())).strip()
        return inner if key in _NON_JOURNAL_DESCRIPTORS else m.group(0)

    return re.sub(r"\*([^*]+)\*", _unwrap, citation)


def extract_info_fields(text):
    """Parses the raw metadata text output into a dictionary."""
    info = {
        "year": None, "filename": "Report", "citation": None, "url": None,
        "title": None, "authors": None, "abstract_summary": None,
        "key_methodology": None, "research_question": None, "doc_type": None,
        "discipline": None, "title_authors": None, "is_empirical": None,
        "central_argument": None,
        "supplement_start_page": None,
        "corresponding_author": None,
        "author_email": None
    }
    
    def get_val(pattern):
        match = re.search(r"(?i)\**" + pattern + r"\**:\s*(.+)", text, re.MULTILINE)
        return match.group(1).strip().strip('"') if match else None

    info["year"] = get_val(r"YEAR_OF_PUBLICATION")
    info["title"] = get_val(r"TITLE")
    info["authors"] = get_val(r"AUTHORS")
    info["discipline"] = get_val(r"DISCIPLINE")
    info["citation"] = _sanitize_citation(get_val(r"CITATION"))
    info["is_empirical"] = get_val(r"IS_EMPIRICAL")
    info["title_authors"] = get_val(r"TITLE_AUTHORS")
    info["url"] = get_val(r"URL")
    info["abstract_summary"] = get_val(r"ABSTRACT_SUMMARY")
    info["key_methodology"] = get_val(r"KEY_METHODOLOGY")
    info["research_question"] = get_val(r"RESEARCH_QUESTION")
    info["central_argument"] = get_val(r"CENTRAL_ARGUMENT")
    info["doc_type"] = get_val(r"DOCUMENT_TYPE")
    info["contains_algebra"] = get_val(r"CONTAINS_ALGEBRA")
    info["supplement_start_page"] = get_val(r"SUPPLEMENT_START_PAGE")
    
    info["corresponding_author"] = get_val(r"CORRESPONDING_AUTHOR")
    info["author_email"] = get_val(r"AUTHOR_EMAIL")
    
    return info


def get_citation_block(metadata):
    """Extract citation from metadata."""
    citation = metadata.get("citation") or "Citation not found."
    return _sanitize_citation(citation)


def inject_page_numbers(prompt, metadata, is_code_stage=False):
    """
    Append the explicit page structure instructions to prompts.
    Replaces the old table-based mapping with the user's specific template.
    """
    # 1. Prefer the deterministic physical->printed map (pagemap.build_page_map,
    # cached in metadata['page_map'] at ingestion). When it is high-confidence,
    # render it through the page_numbers.txt template so every stage gets
    # the real "PDF viewer page X -> printed page Y" map, not a guessed range.
    pmap = metadata.get("page_map")
    if pmap and pmap.get("confidence") == "high":
        from reviewer2.core import load_page_numbers_snippet
        snippet = load_page_numbers_snippet(pmap)
    else:
        # 2. Fallback: no reliable map. Tell the model the truth — cite the PDF page
        # index and say so. (The old LLM-guessed page_structure string lived here and
        # was the original off-by-one source; the field has been removed.)
        p_struct = ("No reliable printed-page map could be extracted. Cite the "
                    "PDF page index you see and state that it is the PDF page index.")
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


def load_fact_check_resource():
    """Loads the aggressive fact check protocol."""
    try:
        return load_prompt("prompts/resources/fact_check.txt")
    except:
        return "NOTE: Aggressive fact-checking protocol not found. Proceed with standard caution."


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
