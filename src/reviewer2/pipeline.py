"""Pipeline orchestrator for reviewer2.

Public entry point: ``run(pdf_path, work_dir, ...)``. The CLI in
``reviewer2.cli`` is a thin wrapper around it.

Every stage writes its output as a ``.txt`` file in ``work_dir``. To resume an
interrupted run, call ``run`` again with the same ``work_dir`` and a
``start_stage`` at or before the first stage that did not finish; earlier
stages are read back from disk. No database is used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import uuid as uuid_lib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from reviewer2 import core, pagemap, ref_checker, stages
from reviewer2.core import USAGE_LOG, cleanup_resources, merge_pdfs_python, save_output
from reviewer2.helpers import calculate_cost, split_code_dir
from reviewer2.render_text import render_text

MAX_COMBINED_PAGES = 1000


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce a report."""


class _Stop(Exception):
    """Raised internally when ``stop_stage`` is reached."""


class _Tee:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            self._original.write(data)
            self._log_file.write(data)

    def flush(self):
        self._original.flush()
        self._log_file.flush()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _generate_filename_slug(authors, title, job_uuid):
    raw_auth = authors or "Anon"
    raw_auth = re.sub(r"^(authors?|by):?\s*", "", raw_auth, flags=re.IGNORECASE)
    auth_slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_auth).strip("_")[:25]
    raw_title = title or "Paper"
    clean_title = re.sub(r"^(the|a|an)\s+", "", raw_title, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", clean_title) if w]
    title_slug = "_".join(words[:6]) or "Paper"
    return f"{auth_slug}_{title_slug}_{job_uuid}"


def _create_instruction_page(text, output_path):
    """Render the user's note as a one-page PDF that is prepended to the paper."""
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        margin = 50
        text_width = width - (margin * 2)
        y = height - margin
        c.setFont("Helvetica-Bold", 12)
        heading = ("**CRITICAL SYSTEM INSTRUCTION: YOU MUST TAKE THE FOLLOWING INTO "
                   "ACCOUNT WHEN ASSESSING THIS TEXT:**")
        for line in simpleSplit(heading, "Helvetica-Bold", 12, text_width):
            c.drawString(margin, y, line)
            y -= 15
        y -= 10
        c.setFont("Helvetica", 12)
        for line in simpleSplit(text, "Helvetica", 12, text_width):
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 12)
            c.drawString(margin, y, line)
            y -= 15
        c.save()
        return True
    except Exception as e:
        print(f"  ⚠ Failed to generate instruction page: {e}")
        return False


def _assemble_full_review(narrative_text, issues_text):
    if not narrative_text:
        return ""
    if not issues_text:
        return narrative_text
    split_pattern = r"(?i)^(#{1,2}\s*\*?Future Research\*?)"
    parts = re.split(split_pattern, narrative_text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) >= 3:
        return f"{parts[0].strip()}\n\n## Potential Issues\n\n{issues_text.strip()}\n\n{parts[1]}{parts[2]}"
    print("  ⚠ Could not find 'Future Research' section to split. Appending Issues at end.")
    return f"{narrative_text}\n\n## Potential Issues\n\n{issues_text}"


def _run_stage_with_retry(stage_func, step_name, max_retries=1):
    """Execute a stage function with exponential backoff."""
    retries = 0
    while retries <= max_retries:
        try:
            print(f"\n  ► Executing {step_name} (Attempt {retries + 1})...")
            return stage_func()
        except Exception as e:
            retries += 1
            print(f"    ✗ Error in {step_name}: {e}")
            if retries > max_retries:
                print(f"    ⛔ {step_name} FAILED after {max_retries + 1} attempts.")
                raise
            sleep_time = 30 * (2 ** (retries - 1))
            print(f"    ⏳ Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)


def _run_parallel(named_thunks):
    """Run independent stage thunks concurrently and return {name: result}.

    Stage calls are API-bound, so threads are enough. Every thunk runs to
    completion even if one fails; the first failure is then re-raised."""
    if not named_thunks:
        return {}
    if len(named_thunks) == 1:
        name, fn = next(iter(named_thunks.items()))
        return {name: fn()}
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=len(named_thunks)) as pool:
        futures = {name: pool.submit(fn) for name, fn in named_thunks.items()}
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except BaseException as e:
                print(f"    ✗ Parallel stage {name} failed: {e}")
                errors.append(e)
    if errors:
        raise errors[0]
    return results


def _load_output(dir_path, filename):
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _clean_code_issues(text):
    """Strip fenced code blocks and inline backticks; escape markdown underscores."""
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"(?<!\\)_", r"\_", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stop_if(stop_stage, threshold, label):
    if stop_stage < threshold:
        print(f"  🛑 Reached stop stage ({label}).")
        raise _Stop()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    pdf_path,
    work_dir,
    *,
    math=False,
    code=False,
    copyedit=True,
    editor_note=True,
    references=True,
    supp_pdfs=None,
    code_dir=None,
    note="",
    citation="",
    start_stage=0.0,
    stop_stage=100.0,
    max_retries=1,
    skip_size_check=False,
):
    """Run the pipeline on ``pdf_path`` and return the final plain-text report path.

    Args:
        pdf_path: Path to the source PDF.
        work_dir: Directory for all stage outputs and the final report. Created
            if missing.
        math: Enable the math-audit stages. Requires ``MATHPIX_APP_ID`` and
            ``MATHPIX_APP_KEY`` in the environment.
        code: Enable the code-audit stages. Requires ``code_dir``.
        copyedit: Enable the copyedit stages. Coupled with ``editor_note``: the
            author-facing stages (08–09) run only when both are on.
        editor_note: Enable the editor's note. See ``copyedit``.
        references: Check every entry in the paper's bibliography against
            OpenAlex, Crossref and OpenLibrary, with a grounded web search for
            what those cannot confirm, and add a References section to the report.
        supp_pdfs: Supplementary PDFs to merge after the main paper.
        code_dir: Directory holding the replication package. Ignored unless
            ``code`` is set.
        note: Optional instructions to the reviewer, prepended to the paper as
            an extra first page.
        citation: Citation string to use if metadata extraction produces none.
        start_stage: Resume from this stage number (``0.0`` for a fresh run).
        stop_stage: Last stage number to run (``100.0`` runs to the end).
        max_retries: Per-stage retry budget on exception.
        skip_size_check: Bypass the 1000-page combined-volume limit.

    Returns:
        Path to the plain-text report, or ``work_dir`` if ``stop_stage`` halted
        the run before rendering.

    Raises:
        PipelineError: The pipeline could not produce a report.
        FileNotFoundError: ``pdf_path`` does not exist.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    writer_mode = bool(copyedit and editor_note)
    job_uuid = uuid_lib.uuid4().hex[:8]
    core.CURRENT_JOB_UUID = job_uuid

    log_file = open(work_dir / "pipeline_execution.log", "a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    print(f"→ PIPELINE STARTED: run={job_uuid} | work_dir={work_dir}")
    print(f"  Add-ons: math={math} code={code} copyedit={copyedit} "
          f"editor_note={editor_note} references={references}")

    try:
        return _run_inner(
            pdf_path=str(pdf_path),
            work_dir=str(work_dir),
            job_uuid=job_uuid,
            do_math=math,
            do_code=bool(code and code_dir),
            writer_mode=writer_mode,
            do_references=references,
            supp_pdfs=[str(Path(p).expanduser().resolve()) for p in (supp_pdfs or [])],
            code_dir=str(Path(code_dir).expanduser().resolve()) if code_dir else "",
            note=note or "",
            citation=citation,
            start_stage=start_stage,
            stop_stage=stop_stage,
            max_retries=max_retries,
            skip_size_check=skip_size_check,
        )
    except _Stop:
        return work_dir
    finally:
        try:
            calculate_cost(USAGE_LOG)
        except Exception as e:
            print(f"  ⚠ Cost report failed (non-fatal): {e}")
        cleanup_resources()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_inner(*, pdf_path, work_dir, job_uuid, do_math, do_code, writer_mode,
               do_references, supp_pdfs, code_dir, note, citation, start_stage,
               stop_stage, max_retries, skip_size_check):
    output_dir = work_dir
    target_file = os.path.join(output_dir, "original_source.pdf")
    main_only_file = os.path.join(output_dir, "original_main_only.pdf")

    if start_stage == 0.0:
        shutil.copy2(pdf_path, target_file)
    if not os.path.exists(target_file):
        raise PipelineError(f"Source PDF missing at {target_file}; start from stage 0.")

    # ==================================================================
    # PRE-FLIGHT: combined page limit (paper + compiled replication package)
    # ==================================================================
    early_code_pdf = os.path.join(output_dir, "_early_code.pdf")
    if start_stage == 0.0 and do_code and os.path.isdir(code_dir):
        try:
            print("  → Pre-flight: compiling replication package for size check...")
            ordered, package_pdfs = split_code_dir(code_dir, target_file)
            stages.compile_code_to_pdf(code_dir, ordered, early_code_pdf)
            main_pages = len(PdfReader(target_file).pages)
            code_pages = len(PdfReader(early_code_pdf).pages)
            for pp in package_pdfs:
                try:
                    code_pages += len(PdfReader(os.path.join(code_dir, pp)).pages)
                except Exception:
                    pass
            total = main_pages + code_pages
            print(f"  → Pre-flight: {main_pages} paper pages + {code_pages} code pages = {total}")
            if total > MAX_COMBINED_PAGES and not skip_size_check:
                raise PipelineError(
                    f"Combined volume ({total} pages) exceeds the {MAX_COMBINED_PAGES}-page "
                    "limit. Trim the replication package or pass skip_size_check=True.")
            print(f"  ✓ Volume check {'bypassed' if total > MAX_COMBINED_PAGES else 'passed'} ({total} pages).")
        except PipelineError:
            raise
        except Exception as e:
            print(f"  ⚠ Pre-flight volume check failed (non-fatal): {e}")

    # ==================================================================
    # STAGE 0 PREP: user note and supplement merge
    # ==================================================================
    if start_stage == 0.0:
        if note.strip():
            print("  → Prepending the reviewer note as an instruction page...")
            instr_pdf = os.path.join(output_dir, "_instruction.pdf")
            if _create_instruction_page(note, instr_pdf):
                writer = PdfWriter()
                for page in PdfReader(instr_pdf).pages:
                    writer.add_page(page)
                for page in PdfReader(target_file).pages:
                    writer.add_page(page)
                tmp = target_file.replace(".pdf", "_with_note.pdf")
                with open(tmp, "wb") as f_out:
                    writer.write(f_out)
                shutil.move(tmp, target_file)
                os.remove(instr_pdf)

        if supp_pdfs:
            staging = os.path.join(output_dir, "_merge_staging")
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(staging)
            try:
                # merge_pdfs_python inserts the separator page itself; do not stage it.
                for i, supp in enumerate(supp_pdfs):
                    if os.path.exists(supp) and supp.lower().endswith(".pdf"):
                        shutil.copy(supp, os.path.join(staging, f"10_10_{i:03d}_{os.path.basename(supp)}"))
                shutil.copy(target_file, main_only_file)  # main text alone, for the math proofreader
                print("  → Merging supplements...")
                merged, _ = merge_pdfs_python(target_file, staging, output_dir)
                if os.path.abspath(merged) != os.path.abspath(target_file):
                    shutil.move(merged, target_file)
            except Exception as e:
                print(f"  ⚠ Supplement merge failed: {e}")
            finally:
                shutil.rmtree(staging, ignore_errors=True)

    # ==================================================================
    # METADATA
    # ==================================================================
    metadata_json = os.path.join(output_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_json):
        with open(metadata_json, "r") as f:
            metadata = json.load(f)

    ref_check_thread = None
    appendix_text = None

    code_combined_path = os.path.join(output_dir, "code_combined.pdf")
    code_pdf_path = os.path.join(output_dir, "code_compiled.pdf")
    code_combined_pdf = code_combined_path if os.path.exists(code_combined_path) else None
    code_pdf = code_pdf_path if os.path.exists(code_pdf_path) else None

    # ==================================================================
    # STAGE 0: EXTRACTION
    # ==================================================================
    if start_stage <= 0.05 and stop_stage >= 0.0:
        s00a = _run_stage_with_retry(
            lambda: stages.stage_00a_metadata(target_file, output_dir),
            "00a_metadata", max_retries)
        metadata = _run_stage_with_retry(
            lambda: stages.stage_00b_metadata_clean(s00a, target_file, output_dir),
            "00b_metadata_clean", max_retries)
        if not metadata.get("citation") and citation:
            metadata["citation"] = citation
        metadata["filename"] = _generate_filename_slug(
            metadata.get("authors") or metadata.get("title_authors"),
            metadata.get("title"), job_uuid)
        metadata = stages.stage_00b_2_metadata_math(metadata, output_dir)
        metadata["uuid"] = job_uuid
        metadata["mode"] = "Writer" if writer_mode else "Reader"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)

    if os.path.exists(metadata_json):
        with open(metadata_json, "r") as f:
            metadata = json.load(f)

    # ==================================================================
    # PAGE MAP (physical PDF page -> printed page label)
    # ==================================================================
    # Built from the PDF's page labels and a text scan, then reconciled by a
    # model that looks at the pages (unnumbered title pages, folios in images)
    # and validated deterministically. Cached so resumes do not repeat it.
    # Stages fall back to citing PDF page indices if no reliable map exists.
    try:
        if os.path.exists(target_file):
            metadata["page_map"] = pagemap.build_page_map(target_file)
            pm = metadata["page_map"]
            print(f"  → Page map: {pm['method']}/{pm['confidence']} "
                  f"({pm['cross_check']}); ceilings={pm['ceilings']}")
            pm_cache = os.path.join(output_dir, "page_map.json")
            if os.path.exists(pm_cache):
                with open(pm_cache) as f:
                    metadata["page_map"] = json.load(f)
                print("  → Page map: reconciled map loaded from cache")
            else:
                def _reconcile_call(prompt, pdf):
                    return core.call_gemini(prompt, pdf, model_type="flash_2_5",
                                            temperature=0.0, thinking_budget=-1,
                                            step="00d_pagemap_reconcile")
                metadata["page_map"] = pagemap.reconcile_with_flash(
                    metadata["page_map"], target_file, _reconcile_call)
                try:
                    with open(pm_cache, "w") as f:
                        json.dump(metadata["page_map"], f, indent=2)
                except Exception:
                    pass
                rm = metadata["page_map"]
                print(f"  → Page map reconciled: {rm.get('method')}/{rm.get('confidence')}")
    except Exception as e:
        print(f"  ⚠ Page-map build failed (stages fall back): {e}")
        metadata.setdefault("page_map", None)

    if start_stage <= 0.9 and stop_stage >= 0.9:
        _run_stage_with_retry(
            lambda: stages.stage_00c_contributions(target_file, metadata, output_dir),
            "00c_contributions", max_retries)
    s00c = _load_output(output_dir, "00c_contributions.txt")

    # ==================================================================
    # REFERENCE CHECK (background thread, joined before rendering)
    # ==================================================================
    if do_references and not _load_output(output_dir, "reference_check.txt"):
        ref_check_thread = threading.Thread(
            target=ref_checker.run_reference_check,
            args=(target_file, output_dir, dict(metadata)),
            daemon=True)
        ref_check_thread.start()
        print("  → Reference check started (background thread)")

    # ==================================================================
    # STAGE 1: RED TEAM, MATH AND CODE TRACKS (run concurrently)
    # ==================================================================
    _stop_if(stop_stage, 1.0, "before stage 1")

    is_empirical = str(metadata.get("is_empirical", "YES")).upper() == "YES"
    s01a = s01b = s01c = s01d = s01e = s01e2 = s01f_alg = s01g = None

    if start_stage <= 1.99:

        def _track_red_team():
            """01a, 01b, 01c and 01g in parallel, then 01d; or 01a and two
            further Breaker rounds for non-empirical papers."""
            t01a = t01b = t01c = t01d = t01g = None
            finders = {}
            if start_stage <= 1.0 and stop_stage >= 1.0:
                finders["01a_breaker"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01a_breaker(target_file, metadata, output_dir, s00c, appendix_text),
                    "01a_breaker", max_retries)
            if is_empirical and start_stage <= 1.1 and stop_stage >= 1.1:
                finders["01b_butcher"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01b_butcher(target_file, metadata, output_dir, s00c, appendix_text),
                    "01b_butcher", max_retries)
                finders["01c_shredder"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01c_shredder(target_file, metadata, output_dir, s00c, appendix_text),
                    "01c_shredder", max_retries)
            if start_stage <= 1.6 and stop_stage >= 1.6:
                finders["01g_the_void"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01g_the_void(target_file, s00c, metadata, output_dir, appendix_text),
                    "01g_the_void", max_retries)

            results = _run_parallel(finders)
            t01a = results.get("01a_breaker") or _load_output(output_dir, "01a_breaker.txt")
            t01g = results.get("01g_the_void") or _load_output(output_dir, "01g_the_void.txt")

            if is_empirical:
                t01b = results.get("01b_butcher") or _load_output(output_dir, "01b_butcher.txt")
                t01c = results.get("01c_shredder") or _load_output(output_dir, "01c_shredder.txt")
                if start_stage <= 1.2 and stop_stage >= 1.2:
                    t01d = _run_stage_with_retry(
                        lambda: stages.stage_01d_collector(target_file, t01b, t01c, metadata, output_dir, appendix_text),
                        "01d_collector", max_retries)
                else:
                    t01d = _load_output(output_dir, "01d_collector.txt")
            else:
                print("  → Non-empirical paper: skipping Butcher/Shredder, running Breaker rounds 2 and 3.")
                if start_stage <= 1.1 and stop_stage >= 1.1:
                    t01a_2 = _run_stage_with_retry(
                        lambda: stages.stage_01a_2_breaker_revisit(target_file, t01a, metadata, output_dir, round_num=2),
                        "01a_2_breaker_revisit", max_retries)
                    t01a_3 = _run_stage_with_retry(
                        lambda: stages.stage_01a_2_breaker_revisit(target_file, f"{t01a}\n\n{t01a_2}", metadata, output_dir, round_num=3),
                        "01a_3_breaker_revisit", max_retries)
                else:
                    t01a_2 = _load_output(output_dir, "01a_2_breaker_revisit.txt")
                    t01a_3 = _load_output(output_dir, "01a_3_breaker_revisit.txt")
                t01b = (f"BREAKER_ROUND_2_RESULTS (NON_EMPIRICAL_DEEP_DIVE):\n{t01a_2}\n\n"
                        f"BREAKER_ROUND_3_RESULTS:\n{t01a_3}")
                t01c = "N/A (Non-Empirical)"
                t01d = "N/A (Non-Empirical)"
            return t01a, t01b, t01c, t01d, t01g

        def _track_math():
            """01e, plus (with math) 01e2, then (01fa ∥ 01fb) -> 01fc ->
            01fd consolidate -> 01fe defence -> 01ff assessment -> 01fg write-up."""
            t01e = t01e2 = None
            t01f_alg = "=NULL="
            group = {}
            if start_stage <= 1.3 and stop_stage >= 1.3:
                group["01e_math"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01e_math_extract(target_file, s00c, metadata, output_dir, appendix_text),
                    "01e_math", max_retries)
            if do_math and start_stage <= 1.35 and stop_stage >= 1.35:
                group["01e2_equation_extraction"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01e2_equation_extraction(target_file, metadata, output_dir),
                    "01e2_equation_extraction", max_retries)
            results = _run_parallel(group)
            t01e = results.get("01e_math") or _load_output(output_dir, "01e_math.txt")
            if not do_math:
                return t01e, t01e2, t01f_alg

            t01e2 = results.get("01e2_equation_extraction") or _load_output(output_dir, "01e2_equations.txt")
            if start_stage <= 1.4 and stop_stage >= 1.4:
                proofread_pdf = main_only_file if os.path.exists(main_only_file) else target_file
                fr = _run_parallel({
                    "01fa_math_check": lambda: _run_stage_with_retry(
                        lambda: stages.stage_01fa_math_check(target_file, s00c, metadata, output_dir, appendix_text, equations=t01e2),
                        "01fa_math_check", max_retries),
                    "01fb_math_proofread": lambda: _run_stage_with_retry(
                        lambda: stages.stage_01fb_math_proofread(proofread_pdf, s00c, metadata, output_dir, appendix_text, equations=t01e2),
                        "01fb_math_proofread", max_retries),
                })
                t01fa, t01fb = fr["01fa_math_check"], fr["01fb_math_proofread"]
            else:
                t01fa = _load_output(output_dir, "01fa_math_check.txt")
                t01fb = _load_output(output_dir, "01fb_math_proofread.txt")

            if start_stage <= 1.5 and stop_stage >= 1.5:
                t01fb = t01fb or _load_output(output_dir, "01fb_math_proofread.txt") or ""
                t01fc = _run_stage_with_retry(
                    lambda: stages.stage_01fc_math_audit(target_file, t01fb, s00c, metadata, output_dir, appendix_text, equations=t01e2),
                    "01fc_math_audit", max_retries)
            else:
                t01fc = _load_output(output_dir, "01fc_math_audit.txt")

            if start_stage <= 1.55 and stop_stage >= 1.55:
                t01fa = t01fa or _load_output(output_dir, "01fa_math_check.txt") or ""
                t01fb = t01fb or _load_output(output_dir, "01fb_math_proofread.txt") or ""
                t01fc = t01fc or _load_output(output_dir, "01fc_math_audit.txt") or ""
                mathpix_raw = _load_output(output_dir, "01e2_mathpix_raw.txt")
                t01fd = _run_stage_with_retry(
                    lambda: stages.stage_01fd_math_consolidate(t01fa, t01fb, t01fc, metadata, output_dir),
                    "01fd_math_consolidate", max_retries)
                t01fe = _run_stage_with_retry(
                    lambda: stages.stage_01fe_math_blue(target_file, t01fd, s00c, metadata, output_dir, equations=t01e2, mathpix_raw=mathpix_raw),
                    "01fe_math_blue", max_retries)
                t01ff = _run_stage_with_retry(
                    lambda: stages.stage_01ff_math_assess(t01fd, t01fe, metadata, output_dir, equations=t01e2, mathpix_raw=mathpix_raw),
                    "01ff_math_assess", max_retries)
                t01f_alg = _run_stage_with_retry(
                    lambda: stages.stage_01fg_math_list(t01ff, output_dir),
                    "01fg_math_list", max_retries)
            else:
                t01f_alg = _load_output(output_dir, "01fg_math_list.txt") or "=NULL="
            return t01e, t01e2, t01f_alg

        def _track_code():
            """Compile the package, run 01i ∥ 01j ∥ 01k, then 01l -> 01m ->
            deterministic location gate -> 01n."""
            nonlocal code_pdf, code_combined_pdf
            if not do_code:
                return
            if not (os.path.isdir(code_dir) or code_combined_pdf or code_pdf):
                return
            print("\n  → CODE ANALYSIS ENABLED")

            if os.path.isdir(code_dir) and start_stage <= 1.7:
                # Code and text files are compiled to a text PDF; PDFs in the
                # package are appended as native pages (copies of the paper dropped).
                ordered_code, package_pdfs = split_code_dir(code_dir, target_file)
                if os.path.exists(early_code_pdf):
                    shutil.move(early_code_pdf, code_pdf_path)
                else:
                    stages.compile_code_to_pdf(code_dir, ordered_code, code_pdf_path)
                code_pdf = code_pdf_path

                cw = PdfWriter()
                main_reader = PdfReader(target_file)
                code_reader = PdfReader(code_pdf)
                pkg_readers = []
                for pp in package_pdfs:
                    try:
                        pkg_readers.append(PdfReader(os.path.join(code_dir, pp)))
                    except Exception as e:
                        print(f"  ⚠ Could not append package PDF {pp}: {e}")
                total = len(main_reader.pages) + len(code_reader.pages) + sum(len(r.pages) for r in pkg_readers)
                if total > MAX_COMBINED_PAGES and not skip_size_check and start_stage > 0:
                    raise PipelineError(f"Combined PDF exceeds the {MAX_COMBINED_PAGES}-page limit ({total} pages).")
                for reader in [main_reader, code_reader, *pkg_readers]:
                    for p in reader.pages:
                        cw.add_page(p)
                with open(code_combined_path, "wb") as f:
                    cw.write(f)
                code_combined_pdf = code_combined_path

            gonzo = {}
            if start_stage <= 1.8 and stop_stage >= 1.8:
                gonzo["01i_code_gonzo"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo(code_combined_pdf, metadata, output_dir),
                    "01i_code_gonzo", max_retries)
            if start_stage <= 1.81 and stop_stage >= 1.81:
                gonzo["01j_code_gonzo_b"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo_b(code_combined_pdf, metadata, output_dir),
                    "01j_code_gonzo_b", max_retries)
            if start_stage <= 1.82 and stop_stage >= 1.82:
                gonzo["01k_code_gonzo_c"] = lambda: _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo_c(code_combined_pdf, metadata, output_dir),
                    "01k_code_gonzo_c", max_retries)
            _run_parallel(gonzo)

            if start_stage <= 1.83 and stop_stage >= 1.83:
                g1 = _load_output(output_dir, "01i_code_gonzo.txt") or ""
                g2 = _load_output(output_dir, "01j_code_gonzo_b.txt") or ""
                g3 = _load_output(output_dir, "01k_code_gonzo_c.txt") or ""
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_compiler(g1, g2, g3, metadata, output_dir),
                    "01l_code_compiler", max_retries)

            if start_stage <= 1.85 and stop_stage >= 1.85:
                consolidated = _load_output(output_dir, "01l_code_compiler.txt") or ""
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_checker(consolidated, metadata, output_dir, code_combined_pdf),
                    "01m_code_checker", max_retries)

            if start_stage <= 1.87 and stop_stage >= 1.87:
                consolidated = _load_output(output_dir, "01l_code_compiler.txt") or ""
                checker = _load_output(output_dir, "01m_code_checker.txt") or ""
                # Deterministic location gate: every quoted line of code must exist
                # in the cited file; relocate it if it is in another file, drop it
                # if it is nowhere.
                checker, vstats = stages.verify_and_relocate_code_findings(checker, code_dir, target_file)
                if any(vstats.get(k) for k in ("relocated", "dropped")):
                    save_output(checker, "01m_code_checker_verified.txt", output_dir)
                print(f"  → Code location gate: {vstats}")
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_list(consolidated, checker, metadata, output_dir),
                    "01n_code_list", max_retries)

        tracks = _run_parallel({
            "red_team": _track_red_team,
            "math": _track_math,
            "code": _track_code,
        })
        s01a, s01b, s01c, s01d, s01g = tracks["red_team"]
        s01e, s01e2, s01f_alg = tracks["math"]

        if start_stage <= 1.99 and stop_stage >= 1.99:
            _run_stage_with_retry(
                lambda: stages.stage_01o_summarizer(s01a, s01b, s01c, s01d, s01f_alg, s01e, s01g, s00c, metadata, output_dir),
                "01o_summarizer", max_retries)

    potential_issues = _load_output(output_dir, "01o_summarizer.txt")
    if s01e2 is None:
        s01e2 = _load_output(output_dir, "01e2_equations.txt")

    # ==================================================================
    # STAGE 2: FILTER AND EVALUATE THE POTENTIAL ISSUES
    # ==================================================================
    _stop_if(stop_stage, 2.0, "before stage 2")

    def _stage(threshold, fname, step, fn):
        """Run ``fn`` if ``threshold`` lies inside [start_stage, stop_stage];
        otherwise read the cached output."""
        if start_stage <= threshold and stop_stage >= threshold:
            return _run_stage_with_retry(fn, step, max_retries)
        return _load_output(output_dir, fname)

    s02a = _stage(2.0, "02a_numbers.txt", "02a_numbers",
                  lambda: stages.stage_02a_numbers(target_file, potential_issues, metadata, output_dir, equations=s01e2))
    s02b = _stage(2.1, "02b_compiler_1.txt", "02b_compiler_1",
                  lambda: stages.stage_02b_compiler_1(potential_issues, s02a, output_dir))
    s02c = _stage(2.2, "02c_blue_team.txt", "02c_blue_team",
                  lambda: stages.stage_02c_blue_team(target_file, s02b, s00c, metadata, output_dir))
    s02d = _stage(2.3, "02d_compiler_2.txt", "02d_compiler_2",
                  lambda: stages.stage_02d_compiler_2(s02b, s02c, output_dir))
    s02e = _stage(2.4, "02e_assessment.txt", "02e_assessment",
                  lambda: stages.stage_02e_assessment(target_file, s02d, metadata, output_dir))
    s02f = _stage(2.5, "02f_compiler_3.txt", "02f_compiler_3",
                  lambda: stages.stage_02f_compiler_3(s02d, s02e, output_dir))
    s02g = _stage(2.6, "02g_list_v1.txt", "02g_list_v1",
                  lambda: stages.stage_02g_list_v1(target_file, s02f, metadata, output_dir))
    s02h = _stage(2.7, "02h_proportionality_1.txt", "02h_proportionality_1",
                  lambda: stages.stage_02h_proportionality(target_file, s02g, s00c, metadata, output_dir))
    s02i = _stage(2.8, "02i_list_v2.txt", "02i_list_v2",
                  lambda: stages.stage_02i_list_v2(target_file, s02h, metadata, output_dir))

    # ==================================================================
    # STAGE 3: CHECKS
    # ==================================================================
    _stop_if(stop_stage, 3.0, "before stage 3")

    # The proportioned list (02i) is what the checks run on; fall back to 02g.
    list_v2 = s02i
    if list_v2 is None or "=NULL=" in list_v2 or len(list_v2.strip()) < 50:
        if list_v2 is not None:
            print("  → 02i list missing or short. Using 02g for the checks.")
        list_v2 = s02g

    checks = {}
    if start_stage <= 3.0 and stop_stage >= 3.0:
        checks["03a_checker_1"] = lambda: _run_stage_with_retry(
            lambda: stages.stage_03a_checker_1(target_file, list_v2, metadata, output_dir, equations=s01e2),
            "03a_checker_1", max_retries)
    if start_stage <= 3.2 and stop_stage >= 3.2:
        checks["03b_external"] = lambda: _run_stage_with_retry(
            lambda: stages.stage_03b_external(target_file, list_v2, metadata, output_dir),
            "03b_external", max_retries)
    checks_r = _run_parallel(checks)
    s03a = checks_r.get("03a_checker_1") or _load_output(output_dir, "03a_checker_1.txt")
    s03b = checks_r.get("03b_external") or _load_output(output_dir, "03b_external.txt")
    s03c = _stage(3.4, "03c_list_v3.txt", "03c_list_v3",
                  lambda: stages.stage_03c_list_v3(list_v2, s03a, s03b, metadata, output_dir))

    if s03c is None or "=NULL=" in s03c or len(s03c.strip()) < 50:
        print("  → 03c missing or short. Using the 02i/02g list.")
        final_list = list_v2
    else:
        final_list = s03c

    # ==================================================================
    # STAGE 4: REVIEWER
    # ==================================================================
    _stop_if(stop_stage, 4.0, "before stage 4")
    s04 = _stage(4.0, "04a_reviewer.txt", "04a_reviewer",
                 lambda: stages.stage_04a_reviewer(target_file, s00c, final_list, metadata, output_dir, appendix_text))

    # ==================================================================
    # STAGE 5: REVIEW CHECKS AND REVISION
    # ==================================================================
    _stop_if(stop_stage, 5.0, "before stage 5")
    full_review_draft = _assemble_full_review(s04, final_list)

    revchecks = {}
    if start_stage <= 5.0 and stop_stage >= 5.0:
        revchecks["05a_checker_2"] = lambda: _run_stage_with_retry(
            lambda: stages.stage_05a_checker_2(target_file, full_review_draft, metadata, output_dir, equations=s01e2),
            "05a_checker_2", max_retries)
    if start_stage <= 5.2 and stop_stage >= 5.2:
        revchecks["05b_checker_3"] = lambda: _run_stage_with_retry(
            lambda: stages.stage_05b_checker_3(target_file, full_review_draft, metadata, output_dir),
            "05b_checker_3", max_retries)
    revchecks_r = _run_parallel(revchecks)
    s05a = revchecks_r.get("05a_checker_2") or _load_output(output_dir, "05a_checker_2.txt")
    s05b = revchecks_r.get("05b_checker_3") or _load_output(output_dir, "05b_checker_3.txt")

    if start_stage <= 5.8 and stop_stage >= 5.8:
        # Page guard: reject any citation correction from 05b that points
        # outside the document's real page range before 05c applies it.
        if s05b:
            try:
                s05b, rejected = pagemap.validate_chk3_corrections(s05b, metadata.get("page_map"))
                if rejected:
                    print(f"  → Page guard rejected {len(rejected)} out-of-range correction(s).")
            except Exception as e:
                print(f"  ⚠ Page guard skipped: {e}")
        s05c = _run_stage_with_retry(
            lambda: stages.stage_05c_reviser(full_review_draft, s05a, s05b, metadata, output_dir),
            "05c_reviser", max_retries)
    else:
        s05c = _load_output(output_dir, "05c_reviser.txt")

    if s05c is None or "=NULL=" in s05c or len(s05c.strip()) < 50:
        print("  → 05c missing or short. Using the assembled draft.")
        final_review_text = full_review_draft
    else:
        final_review_text = s05c

    # ==================================================================
    # STAGES 6-7: LEGAL AND FORMAT
    # ==================================================================
    _stop_if(stop_stage, 6.0, "before stage 6")
    s06 = _stage(6.0, "06_legal.txt", "06_legal",
                 lambda: stages.stage_06_legal(final_review_text, metadata, output_dir)) or ""

    _stop_if(stop_stage, 7.0, "before stage 7")
    s07 = _stage(7.0, "07_formatter.txt", "07_formatter",
                 lambda: stages.stage_07_formatter(final_review_text or "", s06, metadata, output_dir, None))

    # ==================================================================
    # DATA EDITOR (code-audit report)
    # ==================================================================
    if do_code and start_stage <= 7.5 and stop_stage >= 7.5:
        print("\n  → Writing the Data Editor report...")
        review_context = final_review_text if (final_review_text or "").strip() else None
        code_list = _load_output(output_dir, "01n_code_list.txt") or ""
        _run_stage_with_retry(
            lambda: stages.stage_04b_data_editor(code_list, metadata, output_dir, paper_review_context=review_context),
            "04b_data_editor", max_retries)

    final_body = _insert_data_editor(s07 or "", output_dir)

    if ref_check_thread is not None:
        print("  → Waiting for the reference check to finish...")
        ref_check_thread.join()

    with open(os.path.join(output_dir, "07_formatter.txt"), "w", encoding="utf-8") as f:
        f.write(final_body)

    # ==================================================================
    # STAGES 8-9: AUTHOR-FACING ADVICE AND COPYEDITING
    # ==================================================================
    if writer_mode:
        _stop_if(stop_stage, 8.0, "before stage 8")
        _run_writer_stages(target_file, output_dir, metadata, final_body, do_math,
                           start_stage, stop_stage, max_retries)

    # ==================================================================
    # RENDER
    # ==================================================================
    _prepare_body(output_dir, do_references)
    return _render_final_text(output_dir, metadata, job_uuid)


def _insert_data_editor(final_body, output_dir):
    """Insert the Data Editor section after Future Research (or Potential Issues).

    Any existing Data Editor section is removed first so that resumed runs do
    not duplicate it."""
    final_body = re.sub(r"\n*##\s*Data Editor\b.*?(?=\n##?\s|\Z)", "", final_body, flags=re.DOTALL)

    data_editor = _load_output(output_dir, "04b_data_editor.txt")
    code_list = _load_output(output_dir, "01n_code_list.txt")
    combined = ""
    if data_editor and data_editor.strip() != "=NULL=":
        combined += _clean_code_issues(data_editor.strip())
    if code_list and code_list.strip() != "=NULL=":
        if combined:
            combined += "\n\n"
        combined += _clean_code_issues(code_list.strip())
    if not combined:
        return final_body

    de_section = f"\n\n## Data Editor\n\n{combined}\n"

    def insert_after(header_re):
        parts = re.split(f"({header_re})", final_body, flags=re.IGNORECASE)
        if len(parts) < 3:
            return None
        nxt = re.search(r"\n(##|#)\s", parts[2])
        if nxt:
            point = nxt.start()
            return parts[0] + parts[1] + parts[2][:point] + de_section + parts[2][point:]
        return parts[0] + parts[1] + parts[2] + de_section

    for header in (r"##\s*Future Research", r"##\s*Potential Issues"):
        if re.search(header, final_body, flags=re.IGNORECASE):
            result = insert_after(header)
            if result:
                return result
    return final_body + de_section


def _run_writer_stages(target_file, output_dir, metadata, final_body, do_math,
                       start_stage, stop_stage, max_retries):
    """08 math advisor (with math), 08a Alchemist, 08b Polisher, 09a-09c copyediting."""
    print("\n═══════════════════════════════════════════════════════════════════")
    print("   AUTHOR-FACING STAGES")
    print("═══════════════════════════════════════════════════════════════════")

    # Code findings are for readers of the review, not for the revision advice.
    review_for_advice = re.sub(r"\n*## Data Editor\n.*", "", final_body, flags=re.DOTALL)

    # The math advisor reads the assessment stage (01ff), whose output carries
    # the VERDICT: CONFIRMED markers it filters on.
    math_advice = None
    if do_math:
        math_advice = _load_output(output_dir, "08_math_advisor.txt")
        if not math_advice:
            assessed = (_load_output(output_dir, "01ff_math_assess.txt")
                        or _load_output(output_dir, "01fd_math_consolidate.txt") or "")
            contribs = _load_output(output_dir, "00c_contributions.txt") or ""
            math_advice = _run_stage_with_retry(
                lambda: stages.stage_08_math_advisor(target_file, assessed, contribs, metadata, output_dir),
                "08_math_advisor", max_retries)

    def get_data():
        d = {}
        for key, fname in [("s08a", "08a_alchemist.txt"),
                           ("s08a_inst", "08a_alchemist_instructions.txt"),
                           ("s08b", "08b_polisher.txt"),
                           ("s09a", "09a_proofreader.txt")]:
            content = _load_output(output_dir, fname)
            if content is not None:
                d[key] = content
        return d

    def run_08a():
        return stages.stage_08a_alchemist(target_file, review_for_advice, metadata, output_dir, math_advice=math_advice)

    def run_08b():
        data = get_data()
        if "s08a" not in data:
            raise ValueError("Missing input: 08a_alchemist.txt")
        return stages.stage_08b_polisher(data["s08a"], metadata, output_dir)

    def run_09a():
        return stages.stage_09a_proofreader(target_file, metadata, output_dir)

    def run_09b():
        return stages.stage_09b_proofread_clean(get_data().get("s09a", "=NULL="), metadata, output_dir)

    def run_09c():
        data = get_data()
        return stages.stage_09c_copyedit(target_file, review_for_advice, data.get("s08b", "Note missing."),
                                         data.get("s08a_inst", "Instructions missing."), metadata, output_dir)

    steps = [
        {"id": "08a", "func": run_08a, "threshold": 8.0},
        {"id": "08b", "func": run_08b, "threshold": 8.5},
        {"id": "09a", "func": run_09a, "threshold": 9.0},
        {"id": "09b", "func": run_09b, "threshold": 9.1},
        {"id": "09c", "func": run_09c, "threshold": 9.2},
    ]
    global_retries, max_global_retries, idx = 0, 4, 0
    while idx < len(steps) and start_stage > steps[idx]["threshold"]:
        idx += 1

    while idx < len(steps):
        step = steps[idx]
        if step["threshold"] > stop_stage:
            print(f"  🛑 Reached stop stage ({stop_stage}) before {step['id']}.")
            raise _Stop()
        print(f"\n  ► Executing Step {step['id']}...")
        try:
            step["func"]()
            print(f"    ✓ Step {step['id']} complete.")
            idx += 1
            if step["id"] == "08b":
                data = get_data()
                if "s08b" in data:
                    metadata["editor_note"] = data["s08b"]
                    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"    ✗ Failure in step {step['id']}: {e}")
            try:
                print(f"    ↻ Retrying step {step['id']}...")
                step["func"]()
                print(f"    ✓ Step {step['id']} recovered.")
                idx += 1
            except Exception as e2:
                print(f"    ✗ Failure (second attempt) in step {step['id']}: {e2}")
                if global_retries < max_global_retries and idx > 0:
                    global_retries += 1
                    print(f"    ⚠ Stepping back to {steps[idx - 1]['id']} ({global_retries}/{max_global_retries})")
                    idx -= 1
                    time.sleep(10)
                else:
                    raise PipelineError(f"Author-facing stages failed at {step['id']}: {e2}")


def _prepare_body(output_dir, do_references):
    """Build 10_latex_body.txt: the formatted review with code blocks stripped and
    the reference-check section placed before the Data Editor (or at the end)."""
    source = os.path.join(output_dir, "07_formatter.txt")
    if not os.path.exists(source):
        raise PipelineError(f"Formatted review not found at {source}")
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)

    if do_references:
        text = re.sub(r"\n##\s*References\b.*?(?=\n#\s|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
        ref_section = _load_output(output_dir, "reference_check.txt")
        if ref_section and ref_section.strip() and "=NULL=" not in ref_section:
            block = f"\n\n{ref_section.strip()}\n"
            anchor = re.search(r"\n##\s*Data Editor\b", text) or re.search(r"\n#\s*Appendix", text)
            if anchor:
                text = text[:anchor.start()] + block + text[anchor.start():]
            else:
                text += block

    text = re.sub(r"\n{3,}", "\n\n", text)
    with open(os.path.join(output_dir, "10_latex_body.txt"), "w", encoding="utf-8") as f:
        f.write(text.strip())


def _render_final_text(output_dir, metadata, job_uuid):
    report_text = render_text(output_dir)
    slug = metadata.get("filename") or f"Report_{job_uuid}"
    output_path = Path(output_dir) / f"{slug}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\n  ✓ Final report: {output_path}")
    return output_path
