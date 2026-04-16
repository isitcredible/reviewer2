"""Pipeline orchestrator for reviewer2.

Public entry point: ``run(pdf_path, work_dir, ...)``. The CLI in
``reviewer2.cli`` is a thin wrapper around it.

The pipeline is resumable: stage outputs land as ``.txt`` files in ``work_dir``
and subsequent runs pick up where a previous run stopped. Deleting a stage
file forces that stage (and dependents) to re-run. No database is used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid as uuid_lib
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from reviewer2 import stages
from reviewer2.core import USAGE_LOG, cleanup_resources, merge_pdfs_python
from reviewer2.helpers import calculate_cost
from reviewer2.render_text import render_text

MAX_COMBINED_PAGES = 500


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce a report."""


class _Tee:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)

    def flush(self):
        self._original.flush()
        self._log_file.flush()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _generate_filename_slug(authors: str | None, title: str | None, job_uuid: str) -> str:
    raw_auth = authors or "Anon"
    raw_auth = re.sub(r"^(authors?|by):?\s*", "", raw_auth, flags=re.IGNORECASE)
    auth_slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_auth).strip("_")[:25]
    raw_title = title or "Paper"
    clean_title = re.sub(r"^(the|a|an)\s+", "", raw_title, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", clean_title) if w]
    title_slug = "_".join(words[:6]) or "Paper"
    return f"{auth_slug}_{title_slug}_{job_uuid}"


def _assemble_full_review(narrative_text: str, issues_text: str) -> str:
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


def _run_stage_with_retry(stage_func, step_name: str, max_retries: int = 1):
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


def _load_output(work_dir: Path, filename: str) -> str | None:
    path = work_dir / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _clean_code_issues_for_pdf(text: str) -> str:
    """Strip fenced code blocks and inline backticks; escape markdown underscores."""
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"(?<!\\)_", r"\_", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    pdf_path: str | Path,
    work_dir: str | Path,
    *,
    math: bool = False,
    code: bool = True,
    copyedit: bool = True,
    editor_note: bool = True,
    supp_pdfs: list[str | Path] | None = None,
    code_dir: str | Path | None = None,
    citation: str = "",
    start_stage: float = 0.0,
    stop_stage: float = 100.0,
    max_retries: int = 1,
    skip_size_check: bool = False,
) -> Path:
    """Run the pipeline on ``pdf_path`` and return the final plain-text report path.

    Args:
        pdf_path: Path to the source PDF.
        work_dir: Directory for all stage outputs and the final report. Created
            if missing. Re-running with the same ``work_dir`` resumes from the
            last successfully written stage.
        math: Enable math-audit stages. Requires ``MATHPIX_APP_ID`` and
            ``MATHPIX_APP_KEY`` in the environment.
        code: Enable code / replication-audit stages. Requires ``code_dir``.
        copyedit: Enable copyedit stages (Writer Mode). ``editor_note`` is
            also required for these stages to run — they share inputs.
        editor_note: Enable the editor's-note stage. See ``copyedit``.
        supp_pdfs: Paths to supplementary PDFs to merge after the main paper.
        code_dir: Directory of source-code files for the code-audit add-on.
            Ignored if ``code`` is False.
        citation: Optional citation string to force-fill if metadata extraction
            fails to produce one.
        start_stage: Resume from this stage number (use ``0.0`` for a fresh run).
        stop_stage: Halt before this stage number (use ``100.0`` to run to end).
        max_retries: Per-stage retry budget on exception.
        skip_size_check: Bypass the 500-page combined-volume circuit breaker.

    Returns:
        Path to the plain-text report (``work_dir / <slug>.txt``).

    Raises:
        PipelineError: The pipeline could not produce a report.
        FileNotFoundError: ``pdf_path`` does not exist.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve add-on gate. Writer Mode stages (08a–09c) require both editor_note
    # and copyedit: 09c reads outputs from 08a+08b, so they're coupled in this
    # release. If you disable either, all Writer Mode stages skip.
    writer_mode = bool(copyedit and editor_note)

    job_uuid = uuid_lib.uuid4().hex[:8]
    print(f"→ PIPELINE STARTED: job_uuid={job_uuid} | work_dir={work_dir}")
    print(f"  Addons: math={math} code={code} copyedit={copyedit} editor_note={editor_note}")

    # Route stdout/stderr through a tee so the full run is captured on disk.
    log_path = work_dir / "pipeline_execution.log"
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)

    try:
        return _run_inner(
            pdf_path=pdf_path,
            work_dir=work_dir,
            job_uuid=job_uuid,
            do_math=math,
            do_code=code,
            do_copyedit=copyedit,
            do_editor_note=editor_note,
            writer_mode=writer_mode,
            supp_pdfs=supp_pdfs or [],
            code_dir=Path(code_dir).expanduser().resolve() if code_dir else None,
            citation=citation,
            start_stage=start_stage,
            stop_stage=stop_stage,
            max_retries=max_retries,
            skip_size_check=skip_size_check,
        )
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
# Internal orchestration
# ---------------------------------------------------------------------------


def _run_inner(
    *,
    pdf_path: Path,
    work_dir: Path,
    job_uuid: str,
    do_math: bool,
    do_code: bool,
    do_copyedit: bool,
    do_editor_note: bool,
    writer_mode: bool,
    supp_pdfs: list,
    code_dir: Path | None,
    citation: str,
    start_stage: float,
    stop_stage: float,
    max_retries: int,
    skip_size_check: bool,
) -> Path:
    # -----------------------------------------------------------------------
    # SOURCE FILE SETUP
    # -----------------------------------------------------------------------
    target_file = work_dir / "original_source.pdf"
    main_only_file = work_dir / "original_main_only.pdf"

    if start_stage == 0.0 and not target_file.exists():
        shutil.copy2(pdf_path, target_file)

    if not target_file.exists():
        raise PipelineError(f"Source PDF missing at {target_file} (resume from start_stage=0.0)")

    # -----------------------------------------------------------------------
    # EARLY CIRCUIT BREAKER (page-limit sanity check)
    # -----------------------------------------------------------------------
    early_code_pdf = work_dir / "_early_code.pdf"
    if start_stage == 0.0 and do_code and code_dir and code_dir.exists():
        try:
            print("  → Pre-flight: compiling replication package for size check...")
            ordered = _walk_code_dir(code_dir)
            stages.compile_code_to_pdf(str(code_dir), ordered, str(early_code_pdf))
            main_pages = len(PdfReader(str(target_file)).pages)
            code_pages = len(PdfReader(str(early_code_pdf)).pages)
            total = main_pages + code_pages
            print(f"  → Pre-flight: {main_pages} paper + {code_pages} code = {total} pages")
            if total > MAX_COMBINED_PAGES and not skip_size_check:
                raise PipelineError(
                    f"Combined volume ({total} pages) exceeds "
                    f"{MAX_COMBINED_PAGES}-page limit. Trim the replication package "
                    f"or pass skip_size_check=True."
                )
            if total > MAX_COMBINED_PAGES:
                print(f"  ⚠ Size check bypassed ({total} pages).")
            else:
                print(f"  ✓ Volume check passed ({total} pages).")
        except PipelineError:
            raise
        except Exception as e:
            print(f"  ⚠ Pre-flight volume check failed (non-fatal): {e}")

    # -----------------------------------------------------------------------
    # PDF SUPPLEMENT MERGE (if any)
    # -----------------------------------------------------------------------
    if start_stage == 0.0 and supp_pdfs:
        staging = work_dir / "_merge_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            # Don't copy the separator PDF into staging — merge_pdfs_python
            # already inserts it once, ahead of the supplements. Copying it
            # here causes a double-insertion.
            for i, supp in enumerate(supp_pdfs):
                supp = Path(supp).expanduser().resolve()
                if supp.exists() and supp.suffix.lower() == ".pdf":
                    shutil.copy(supp, staging / f"10_10_{i:03d}_{supp.name}")
            # Save main-only (without supplements) for the math-proofread stage
            shutil.copy(target_file, main_only_file)
            print("  → Running supplement merge...")
            merged, _ = merge_pdfs_python(str(target_file), str(staging), str(work_dir))
            target_file = Path(merged)
        except Exception as e:
            print(f"  ⚠ Supplement merge failed: {e}")
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    # -----------------------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------------------
    metadata: dict = {}
    metadata_json = work_dir / "metadata.json"
    if metadata_json.exists():
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    code_combined_path = work_dir / "code_combined.pdf"
    code_pdf_path = work_dir / "code_compiled.pdf"
    code_combined_pdf = str(code_combined_path) if code_combined_path.exists() else None
    code_pdf = str(code_pdf_path) if code_pdf_path.exists() else None
    code_analysis = None

    # ====================================================================
    # STAGE 0: EXTRACTION
    # ====================================================================
    if start_stage <= 0.05 and stop_stage >= 0.0:
        s00a = _run_stage_with_retry(
            lambda: stages.stage_00a_metadata(str(target_file), str(work_dir)),
            "00a_metadata", max_retries,
        )
        metadata = _run_stage_with_retry(
            lambda: stages.stage_00b_metadata_clean(s00a, str(target_file), str(work_dir)),
            "00b_metadata_clean", max_retries,
        )
        if not metadata.get("citation") and citation:
            metadata["citation"] = citation
        metadata["filename"] = _generate_filename_slug(
            metadata.get("authors") or metadata.get("title_authors"),
            metadata.get("title"), job_uuid,
        )
        metadata = stages.stage_00b_2_metadata_math(metadata, str(work_dir))
        metadata["uuid"] = job_uuid
        metadata["mode"] = "Writer" if writer_mode else "Reader"
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if metadata_json.exists():
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    if start_stage <= 0.9 and stop_stage >= 0.9:
        _run_stage_with_retry(
            lambda: stages.stage_00c_contributions(str(target_file), metadata, str(work_dir)),
            "00c_contributions", max_retries,
        )

    s00c = _load_output(work_dir, "00c_contributions.txt")

    # ====================================================================
    # STAGE 1: RESEARCH (Red Team)
    # ====================================================================
    if stop_stage < 1.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 1).")
        return _finalize_no_render(work_dir)

    is_empirical = str(metadata.get("is_empirical", "YES")).upper() == "YES"
    s01a = s01b = s01c = s01d = s01e = s01e2 = s01fa = s01fb = s01fc = s01fd = s01f_alg = s01g = None
    s01a_2 = None

    if start_stage <= 1.99:
        # Report flow (core Red Team)
        if start_stage <= 1.0 and stop_stage >= 1.0:
            s01a = _run_stage_with_retry(
                lambda: stages.stage_01a_breaker(str(target_file), metadata, str(work_dir), s00c),
                "01a_breaker", max_retries,
            )
        else:
            s01a = _load_output(work_dir, "01a_breaker.txt")

        if is_empirical:
            if start_stage <= 1.1 and stop_stage >= 1.1:
                s01b = _run_stage_with_retry(
                    lambda: stages.stage_01b_butcher(str(target_file), metadata, str(work_dir), s00c),
                    "01b_butcher", max_retries,
                )
                s01c = _run_stage_with_retry(
                    lambda: stages.stage_01c_shredder(str(target_file), metadata, str(work_dir), s00c),
                    "01c_shredder", max_retries,
                )
            else:
                s01b = _load_output(work_dir, "01b_butcher.txt")
                s01c = _load_output(work_dir, "01c_shredder.txt")

            if start_stage <= 1.2 and stop_stage >= 1.2:
                s01d = _run_stage_with_retry(
                    lambda: stages.stage_01d_collector(str(target_file), s01b, s01c, metadata, str(work_dir)),
                    "01d_collector", max_retries,
                )
            else:
                s01d = _load_output(work_dir, "01d_collector.txt")
        else:
            print("  → Non-Empirical Paper detected: Skipping Butcher/Shredder, triggering Breaker Round 2.")
            if start_stage <= 1.1 and stop_stage >= 1.1:
                s01a = s01a or _load_output(work_dir, "01a_breaker.txt")
                s01a_2 = _run_stage_with_retry(
                    lambda: stages.stage_01a_2_breaker_revisit(str(target_file), s01a, metadata, str(work_dir)),
                    "01a_2_breaker_revisit", max_retries,
                )
            else:
                s01a_2 = _load_output(work_dir, "01a_2_breaker_revisit.txt")
            s01b = f"BREAKER_ROUND_2_RESULTS (NON_EMPIRICAL_DEEP_DIVE):\n{s01a_2}"
            s01c = "N/A (Non-Empirical)"
            s01d = "N/A (Non-Empirical)"

        # Math path (01e always; 01e2/01fa-01fd only if do_math)
        if start_stage <= 1.3 and stop_stage >= 1.3:
            s01e = _run_stage_with_retry(
                lambda: stages.stage_01e_math_extract(str(target_file), s00c, metadata, str(work_dir)),
                "01e_math", max_retries,
            )
        else:
            s01e = _load_output(work_dir, "01e_math.txt")

        if do_math:
            if start_stage <= 1.35 and stop_stage >= 1.35:
                s01e2 = _run_stage_with_retry(
                    lambda: stages.stage_01e2_equation_extraction(str(target_file), metadata, str(work_dir)),
                    "01e2_equation_extraction", max_retries,
                )
            else:
                s01e2 = _load_output(work_dir, "01e2_equations.txt")

            if start_stage <= 1.4 and stop_stage >= 1.4:
                s01fa = _run_stage_with_retry(
                    lambda: stages.stage_01fa_math_check(str(target_file), s00c, metadata, str(work_dir), equations=s01e2),
                    "01fa_math_check", max_retries,
                )
                proofread_pdf = str(main_only_file) if main_only_file.exists() else str(target_file)
                s01fb = _run_stage_with_retry(
                    lambda: stages.stage_01fb_math_proofread(proofread_pdf, s00c, metadata, str(work_dir), equations=s01e2),
                    "01fb_math_proofread", max_retries,
                )
            else:
                s01fa = _load_output(work_dir, "01fa_math_check.txt")
                s01fb = _load_output(work_dir, "01fb_math_proofread.txt")

            if start_stage <= 1.5 and stop_stage >= 1.5:
                s01fb = s01fb or _load_output(work_dir, "01fb_math_proofread.txt") or ""
                s01fc = _run_stage_with_retry(
                    lambda: stages.stage_01fc_math_audit(str(target_file), s01fb, s00c, metadata, str(work_dir), equations=s01e2),
                    "01fc_math_audit", max_retries,
                )
            else:
                s01fc = _load_output(work_dir, "01fc_math_audit.txt")

            if start_stage <= 1.55 and stop_stage >= 1.55:
                s01fa = s01fa or _load_output(work_dir, "01fa_math_check.txt") or ""
                s01fb = s01fb or _load_output(work_dir, "01fb_math_proofread.txt") or ""
                s01fc = s01fc or _load_output(work_dir, "01fc_math_audit.txt") or ""
                mathpix_raw = _load_output(work_dir, "01e2_mathpix_raw.txt")
                s01fd = _run_stage_with_retry(
                    lambda: stages.stage_01fd_math_sober(str(target_file), s01fa, s01fb, s01fc, s00c, metadata, str(work_dir), equations=s01e2, mathpix_raw=mathpix_raw),
                    "01fd_math_sober", max_retries,
                )
                s01f_alg = s01fd
            else:
                s01f_alg = _load_output(work_dir, "01fd_math_sober.txt") or "=NULL="
        else:
            s01f_alg = "=NULL="

        if start_stage <= 1.6 and stop_stage >= 1.6:
            s01g = _run_stage_with_retry(
                lambda: stages.stage_01g_the_void(str(target_file), s00c, metadata, str(work_dir)),
                "01g_the_void", max_retries,
            )
        else:
            s01g = _load_output(work_dir, "01g_the_void.txt")

        # Code analysis
        if do_code and code_dir and code_dir.exists():
            print("\n  → CODE ANALYSIS MODE ENABLED")
            if start_stage <= 1.7:
                if early_code_pdf.exists():
                    shutil.move(str(early_code_pdf), str(code_pdf_path))
                    print("  → Reusing pre-compiled code PDF.")
                else:
                    ordered = _walk_code_dir(code_dir)
                    stages.compile_code_to_pdf(str(code_dir), ordered, str(code_pdf_path))
                code_pdf = str(code_pdf_path)

                writer_pdf = PdfWriter()
                for p in PdfReader(str(target_file)).pages:
                    writer_pdf.add_page(p)
                for p in PdfReader(code_pdf).pages:
                    writer_pdf.add_page(p)
                total_pages = len(writer_pdf.pages)
                if total_pages > MAX_COMBINED_PAGES and not skip_size_check:
                    raise PipelineError(
                        f"Combined PDF ({total_pages} pages) exceeds {MAX_COMBINED_PAGES}-page limit."
                    )
                with open(code_combined_path, "wb") as f:
                    writer_pdf.write(f)
                code_combined_pdf = str(code_combined_path)

            if start_stage <= 1.8 and stop_stage >= 1.8:
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo(code_combined_pdf, metadata, str(work_dir)),
                    "01i_code_gonzo", max_retries,
                )
            if start_stage <= 1.81 and stop_stage >= 1.81:
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo_b(code_combined_pdf, metadata, str(work_dir)),
                    "01j_code_gonzo_b", max_retries,
                )
            if start_stage <= 1.82 and stop_stage >= 1.82:
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_gonzo_c(code_combined_pdf, metadata, str(work_dir)),
                    "01k_code_gonzo_c", max_retries,
                )
            if start_stage <= 1.83 and stop_stage >= 1.83:
                g1 = _load_output(work_dir, "01i_code_gonzo.txt") or ""
                g2 = _load_output(work_dir, "01j_code_gonzo_b.txt") or ""
                g3 = _load_output(work_dir, "01k_code_gonzo_c.txt") or ""
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_compiler(g1, g2, g3, metadata, str(work_dir)),
                    "01l_code_compiler", max_retries,
                )
            if start_stage <= 1.85 and stop_stage >= 1.85:
                consolidated = _load_output(work_dir, "01l_code_compiler.txt") or ""
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_checker(consolidated, metadata, str(work_dir), code_combined_pdf),
                    "01m_code_checker", max_retries,
                )
            if start_stage <= 1.87 and stop_stage >= 1.87:
                consolidated = _load_output(work_dir, "01l_code_compiler.txt") or ""
                checker = _load_output(work_dir, "01m_code_checker.txt") or ""
                _run_stage_with_retry(
                    lambda: stages.stage_01_code_list(consolidated, checker, metadata, str(work_dir)),
                    "01n_code_list", max_retries,
                )
            code_analysis = _load_output(work_dir, "01n_code_list.txt")
        elif do_code:
            code_analysis = _load_output(work_dir, "01n_code_list.txt")

        if start_stage <= 1.99 and stop_stage >= 1.99:
            _run_stage_with_retry(
                lambda: stages.stage_01o_summarizer(s01a, s01b, s01c, s01d, s01f_alg, s01e, s01g, s00c, metadata, str(work_dir)),
                "01o_summarizer", max_retries,
            )

    potential_issues = _load_output(work_dir, "01o_summarizer.txt")

    # ====================================================================
    # STAGE 2–3: LIST + CHECKS
    # ====================================================================
    if stop_stage < 2.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 2).")
        return _finalize_no_render(work_dir)

    s02g = _run_stage_2(str(target_file), potential_issues, s00c, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries)

    if stop_stage < 3.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 3).")
        return _finalize_no_render(work_dir)

    final_list_for_reviewer = _run_stage_3(str(target_file), s02g, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries)

    # ====================================================================
    # STAGE 4: REVIEWER
    # ====================================================================
    if stop_stage < 4.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 4).")
        return _finalize_no_render(work_dir)

    if start_stage <= 4.0 and stop_stage >= 4.0:
        s04 = _run_stage_with_retry(
            lambda: stages.stage_04a_reviewer(str(target_file), s00c, final_list_for_reviewer, metadata, str(work_dir)),
            "04a_reviewer", max_retries,
        )
    else:
        s04 = _load_output(work_dir, "04a_reviewer.txt")

    # ====================================================================
    # STAGE 5: REVISION
    # ====================================================================
    if stop_stage < 5.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 5).")
        return _finalize_no_render(work_dir)

    full_review_draft = _assemble_full_review(s04, final_list_for_reviewer)
    final_review_text = _run_stage_5(str(target_file), full_review_draft, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries)

    # ====================================================================
    # STAGE 6: LEGAL
    # ====================================================================
    if stop_stage < 6.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 6).")
        return _finalize_no_render(work_dir)

    if start_stage <= 6.0 and stop_stage >= 6.0:
        s06 = _run_stage_with_retry(
            lambda: stages.stage_06_legal(final_review_text, metadata, str(work_dir)),
            "06_legal", max_retries,
        )
    else:
        s06 = _load_output(work_dir, "06_legal.txt") or ""

    # ====================================================================
    # STAGE 7: FORMATTER
    # ====================================================================
    if stop_stage < 7.0:
        print("  🛑 Reached Stop Stage (Pre-Stage 7).")
        return _finalize_no_render(work_dir)

    if start_stage <= 7.0 and stop_stage >= 7.0:
        final_review_text = final_review_text or ""
        s07 = _run_stage_with_retry(
            lambda: stages.stage_07_formatter(final_review_text, s06, metadata, str(work_dir)),
            "07_formatter", max_retries,
        )
    else:
        s07 = _load_output(work_dir, "07_formatter.txt") or ""

    # ====================================================================
    # DATA EDITOR (deferred code analysis)
    # ====================================================================
    if do_code:
        if start_stage <= 7.5 and stop_stage >= 7.5:
            print("\n  → Finalizing Code Analysis (Data Editor)...")
            paper_review_context = final_review_text.strip() or None
            code_verified = _load_output(work_dir, "01n_code_list.txt") or ""
            _run_stage_with_retry(
                lambda: stages.stage_04b_data_editor(code_verified, metadata, str(work_dir), paper_review_context=paper_review_context),
                "04b_data_editor", max_retries,
            )

    # Assemble final body with Data Editor insertion
    final_body = _insert_data_editor(s07, work_dir)
    (work_dir / "07_formatter.txt").write_text(final_body, encoding="utf-8")

    # ====================================================================
    # WRITER MODE (Stages 8–9)
    # ====================================================================
    if writer_mode:
        if stop_stage < 8.0:
            print("  🛑 Reached Stop Stage (Pre-Writer Mode).")
            return _finalize_no_render(work_dir)
        _run_writer_mode(str(target_file), final_body, metadata, work_dir, start_stage, stop_stage, max_retries)

    # ====================================================================
    # PYTHON TEXT RENDER (replaces the three R renderers)
    # ====================================================================
    _prepare_latex_body(work_dir)
    return _render_final_text(work_dir, metadata, job_uuid)


# ---------------------------------------------------------------------------
# Stage-block helpers (keep the main orchestrator readable)
# ---------------------------------------------------------------------------


def _run_stage_2(target_file, potential_issues, s00c, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries):
    s02a = s02b = s02c = s02d = s02e = s02f = s02g = None

    if start_stage <= 2.9:
        if start_stage <= 2.0 and stop_stage >= 2.0:
            s02a = _run_stage_with_retry(
                lambda: stages.stage_02a_numbers(target_file, potential_issues, metadata, str(work_dir), equations=s01e2),
                "02a_numbers", max_retries,
            )
        else:
            s02a = _load_output(work_dir, "02a_numbers.txt")

        if start_stage <= 2.1 and stop_stage >= 2.1:
            s02a = s02a or _load_output(work_dir, "02a_numbers.txt")
            s02b = _run_stage_with_retry(
                lambda: stages.stage_02b_compiler_1(potential_issues, s02a, str(work_dir)),
                "02b_compiler_1", max_retries,
            )
        else:
            s02b = _load_output(work_dir, "02b_compiler_1.txt")

        if start_stage <= 2.2 and stop_stage >= 2.2:
            s02b = s02b or _load_output(work_dir, "02b_compiler_1.txt")
            s02c = _run_stage_with_retry(
                lambda: stages.stage_02c_blue_team(target_file, s02b, s00c, metadata, str(work_dir)),
                "02c_blue_team", max_retries,
            )
        else:
            s02c = _load_output(work_dir, "02c_blue_team.txt")

        if start_stage <= 2.3 and stop_stage >= 2.3:
            s02b = s02b or _load_output(work_dir, "02b_compiler_1.txt")
            s02c = s02c or _load_output(work_dir, "02c_blue_team.txt")
            s02d = _run_stage_with_retry(
                lambda: stages.stage_02d_compiler_2(s02b, s02c, str(work_dir)),
                "02d_compiler_2", max_retries,
            )
        else:
            s02d = _load_output(work_dir, "02d_compiler_2.txt")

        if start_stage <= 2.4 and stop_stage >= 2.4:
            s02d = s02d or _load_output(work_dir, "02d_compiler_2.txt")
            s02e = _run_stage_with_retry(
                lambda: stages.stage_02e_assessment(target_file, s02d, metadata, str(work_dir)),
                "02e_assessment", max_retries,
            )
        else:
            s02e = _load_output(work_dir, "02e_assessment.txt")

        if start_stage <= 2.5 and stop_stage >= 2.5:
            s02d = s02d or _load_output(work_dir, "02d_compiler_2.txt")
            s02e = s02e or _load_output(work_dir, "02e_assessment.txt")
            s02f = _run_stage_with_retry(
                lambda: stages.stage_02f_compiler_3(s02d, s02e, str(work_dir)),
                "02f_compiler_3", max_retries,
            )
        else:
            s02f = _load_output(work_dir, "02f_compiler_3.txt")

        if start_stage <= 2.6 and stop_stage >= 2.6:
            s02f = s02f or _load_output(work_dir, "02f_compiler_3.txt")
            s02g = _run_stage_with_retry(
                lambda: stages.stage_02g_list_v1(target_file, s02f, metadata, str(work_dir)),
                "02g_list_v1", max_retries,
            )
        else:
            s02g = _load_output(work_dir, "02g_list_v1.txt")
    else:
        s02g = _load_output(work_dir, "02g_list_v1.txt")
    return s02g


def _run_stage_3(target_file, s02g, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries):
    s03a = s03b = s03c = None

    if start_stage <= 3.9:
        if start_stage <= 3.0 and stop_stage >= 3.0:
            s02g = s02g or _load_output(work_dir, "02g_list_v1.txt")
            s03a = _run_stage_with_retry(
                lambda: stages.stage_03a_checker_1(target_file, s02g, metadata, str(work_dir), equations=s01e2),
                "03a_checker_1", max_retries,
            )
        else:
            s03a = _load_output(work_dir, "03a_checker_1.txt")

        if start_stage <= 3.2 and stop_stage >= 3.2:
            s02g = s02g or _load_output(work_dir, "02g_list_v1.txt")
            s03b = _run_stage_with_retry(
                lambda: stages.stage_03b_external(target_file, s02g, metadata, str(work_dir)),
                "03b_external", max_retries,
            )
        else:
            s03b = _load_output(work_dir, "03b_external.txt")

        if start_stage <= 3.4 and stop_stage >= 3.4:
            s02g = s02g or _load_output(work_dir, "02g_list_v1.txt")
            s03a = s03a or _load_output(work_dir, "03a_checker_1.txt")
            s03b = s03b or _load_output(work_dir, "03b_external.txt")
            s03c = _run_stage_with_retry(
                lambda: stages.stage_03c_list_v2(s02g, s03a, s03b, metadata, str(work_dir)),
                "03c_list_v2", max_retries,
            )
        else:
            s03c = _load_output(work_dir, "03c_list_v2.txt")
    else:
        s03c = _load_output(work_dir, "03c_list_v2.txt")

    if s03c is None:
        print("  ⚠ 03c file not found. Reverting to List V1 (02g).")
        return s02g
    if "=NULL=" in s03c or len(s03c.strip()) < 50:
        print("  → 03c returned =NULL= or is too short. Reverting to List V1 (02g).")
        return s02g
    return s03c


def _run_stage_5(target_file, full_review_draft, metadata, work_dir, s01e2, start_stage, stop_stage, max_retries):
    s05a = s05b = s05c = None

    if start_stage <= 5.9:
        if start_stage <= 5.0 and stop_stage >= 5.0:
            s05a = _run_stage_with_retry(
                lambda: stages.stage_05a_checker_2(target_file, full_review_draft, metadata, str(work_dir), equations=s01e2),
                "05a_checker_2", max_retries,
            )
        else:
            s05a = _load_output(work_dir, "05a_checker_2.txt")

        if start_stage <= 5.2 and stop_stage >= 5.2:
            s05b = _run_stage_with_retry(
                lambda: stages.stage_05b_checker_3(target_file, full_review_draft, metadata, str(work_dir)),
                "05b_checker_3", max_retries,
            )
        else:
            s05b = _load_output(work_dir, "05b_checker_3.txt")

        if start_stage <= 5.8 and stop_stage >= 5.8:
            s05a = s05a or _load_output(work_dir, "05a_checker_2.txt")
            s05b = s05b or _load_output(work_dir, "05b_checker_3.txt")
            s05c = _run_stage_with_retry(
                lambda: stages.stage_05c_reviser(full_review_draft, s05a, s05b, metadata, str(work_dir)),
                "05c_reviser", max_retries,
            )
        else:
            s05c = _load_output(work_dir, "05c_reviser.txt")

    if s05c is None:
        print("  ⚠ 05c file not found. Reverting to Full Review Draft.")
        return full_review_draft
    if "=NULL=" in s05c or len(s05c.strip()) < 50:
        print("  → 05c returned =NULL= or is too short. Reverting to Full Review Draft.")
        return full_review_draft
    return s05c


def _run_writer_mode(target_file, s07_review, metadata, work_dir, start_stage, stop_stage, max_retries):
    """Resilient Writer Mode pipeline with step-back recovery (stages 08a–09c)."""
    print("\n═══════════════════════════════════════════════════════════════════")
    print("   ENTERING WRITER MODE")
    print("═══════════════════════════════════════════════════════════════════")

    # Strip Data Editor block before passing to alchemist — code findings are
    # for the reader, not revision advice.
    alchemist_input = re.sub(r"\n*## Data Editor\n.*", "", s07_review, flags=re.DOTALL)

    def get_data():
        d = {}
        for key, fname in [
            ("s08a", "08a_alchemist.txt"),
            ("s08a_inst", "08a_alchemist_instructions.txt"),
            ("s08b", "08b_polisher.txt"),
            ("s09a", "09a_proofreader.txt"),
        ]:
            content = _load_output(work_dir, fname)
            if content is not None:
                d[key] = content
        return d

    def run_08a():
        return stages.stage_08a_alchemist(target_file, alchemist_input, metadata, str(work_dir))
    def run_08b():
        data = get_data()
        if "s08a" not in data:
            raise ValueError("Missing input: 08a_alchemist.txt")
        return stages.stage_08b_polisher(data["s08a"], metadata, str(work_dir))
    def run_09a():
        return stages.stage_09a_proofreader(target_file, metadata, str(work_dir))
    def run_09b():
        data = get_data()
        source = data.get("s09a", "=NULL=")
        return stages.stage_09b_proofread_clean(source, metadata, str(work_dir))
    def run_09c():
        data = get_data()
        polished = data.get("s08b", "Note missing.")
        inst = data.get("s08a_inst", "Instructions missing.")
        return stages.stage_09c_copyedit(target_file, alchemist_input, polished, inst, metadata, str(work_dir))

    steps = [
        {"id": "08a", "func": run_08a, "threshold": 8.0},
        {"id": "08b", "func": run_08b, "threshold": 8.5},
        {"id": "09a", "func": run_09a, "threshold": 9.0},
        {"id": "09b", "func": run_09b, "threshold": 9.1},
        {"id": "09c", "func": run_09c, "threshold": 9.2},
    ]

    global_retries = 0
    max_global_retries = 4
    idx = 0
    while idx < len(steps) and start_stage > steps[idx]["threshold"]:
        idx += 1

    while idx < len(steps):
        step = steps[idx]
        if step["threshold"] > stop_stage:
            print(f"  🛑 Reached Stop Stage ({stop_stage}). Stopping before {step['id']}.")
            return

        print(f"\n  ► Executing Step {step['id']}...")
        try:
            step["func"]()
            print(f"    ✓ Step {step['id']} Complete.")
            idx += 1
            if step["id"] == "08b":
                data = get_data()
                if "s08b" in data:
                    metadata["editor_note"] = data["s08b"]
                    (work_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"    ✗ FAILURE in Step {step['id']}: {e}")
            try:
                print(f"    ↻ Retrying Step {step['id']} (Attempt 2)...")
                step["func"]()
                print(f"    ✓ Step {step['id']} Recovered.")
                idx += 1
            except Exception as e2:
                print(f"    ✗ FAILURE (Attempt 2) in Step {step['id']}: {e2}")
                if global_retries < max_global_retries and idx > 0:
                    global_retries += 1
                    print(f"    ⚠️ FALLBACK ({global_retries}/{max_global_retries}): stepping back to {steps[idx-1]['id']}")
                    idx -= 1
                    time.sleep(10)
                else:
                    raise PipelineError(f"Writer Mode failed at {step['id']}: {e2}")


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def _walk_code_dir(code_dir: Path) -> list[str]:
    """Return sorted relative filenames under code_dir, excluding dotfiles and _file_map.txt."""
    ordered: list[str] = []
    for root, _, files in os.walk(code_dir):
        for f in files:
            if f.startswith(".") or f == "_file_map.txt":
                continue
            rel_dir = os.path.relpath(root, code_dir)
            ordered.append(f if rel_dir == "." else os.path.join(rel_dir, f))
    ordered.sort()
    return ordered


def _insert_data_editor(final_body: str, work_dir: Path) -> str:
    """Insert the Data Editor / code-issues section after Future Research (if present)."""
    final_body = final_body or ""
    data_editor = _load_output(work_dir, "04b_data_editor.txt")
    code_list = _load_output(work_dir, "01n_code_list.txt")

    combined = ""
    if data_editor and data_editor.strip() != "=NULL=":
        combined += _clean_code_issues_for_pdf(data_editor.strip())
    if code_list and code_list.strip() != "=NULL=":
        if combined:
            combined += "\n\n"
        combined += _clean_code_issues_for_pdf(code_list.strip())

    if not combined:
        return final_body

    de_section = f"\n\n## Data Editor\n\n{combined}\n"

    def insert_after(header_re: str) -> str | None:
        parts = re.split(f"({header_re})", final_body, flags=re.IGNORECASE)
        if len(parts) < 3:
            return None
        next_header = re.search(r"\n(##|#)\s", parts[2])
        if next_header:
            point = next_header.start()
            return parts[0] + parts[1] + parts[2][:point] + de_section + parts[2][point:]
        return parts[0] + parts[1] + parts[2] + de_section

    if "## Future Research" in final_body:
        result = insert_after(r"##\s*Future Research")
        if result:
            return result
    if "## Potential Issues" in final_body:
        result = insert_after(r"##\s*Potential Issues")
        if result:
            return result
    return final_body + de_section


def _prepare_latex_body(work_dir: Path) -> None:
    """Copy 07_formatter.txt → 10_latex_body.txt with code-block stripping."""
    source = work_dir / "07_formatter.txt"
    if not source.exists():
        print(f"  ⚠ CRITICAL: Source body text not found at {source}")
        return
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    (work_dir / "10_latex_body.txt").write_text(text.strip(), encoding="utf-8")


def _render_final_text(work_dir: Path, metadata: dict, job_uuid: str) -> Path:
    """Assemble the plain-text report via render_text and rename to the metadata slug."""
    report_text = render_text(work_dir)
    slug = metadata.get("filename") or f"Report_{job_uuid}"
    output_path = work_dir / f"{slug}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\n  ✓ Final report: {output_path}")
    return output_path


def _finalize_no_render(work_dir: Path) -> Path:
    """Return the work_dir as a pseudo-output when the pipeline halts before rendering."""
    return work_dir
