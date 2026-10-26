"""Offline end-to-end run: every model call is replaced by a canned response.

Exercises the whole stage graph (red team, math and code tracks, filters,
checks, reviewer, revision, legal, format, Data Editor, author-facing stages,
reference check and rendering) without network access or API keys.

    pytest tests/test_offline_run.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from reviewer2 import core, mathpix, pipeline, ref_checker, stages

METADATA = """TITLE: A Test Paper on Rainfall
AUTHORS: Jane Doe and John Roe
YEAR_OF_PUBLICATION: 2024
CITATION: Doe, J., and J. Roe. 2024. "A Test Paper on Rainfall." *Journal of Tests* 1 (1): 1-10.
DISCIPLINE: Economics
IS_EMPIRICAL: YES
DOCUMENT_TYPE: Journal article
ABSTRACT_SUMMARY: The paper estimates the effect of rainfall on crop yields.
KEY_METHODOLOGY: Difference-in-differences.
RESEARCH_QUESTION: Does rainfall raise crop yields?
CENTRAL_ARGUMENT: Rainfall raises crop yields.
CONTAINS_ALGEBRA: YES
"""

ISSUE = ("**1. The identification strategy may not recover the effect claimed.** "
         "The comparison on page 3 appears to conflate timing with treatment.")

REVIEW = """## Is It Credible?

The paper claims a large effect. The design appears unable to support it.

## The Bottom Line

The central estimate may be confounded.

## Future Research

Future work could use a longer panel.
"""


def fake_call(prompt=None, pdf_file_path=None, model_type="flash_lite", step=None, **kwargs):
    step = step or ""
    if step == "00b_metadata_clean":
        return METADATA
    if step == "00b_2_metadata_math":
        return "{}"
    if step == "00d_pagemap_reconcile":
        return "{}"
    if step.startswith("04a_reviewer"):
        return REVIEW
    if step.startswith("05c_reviser") or step.startswith("07_formatter"):
        return REVIEW.replace("## Future Research", f"## Potential Issues\n\n{ISSUE}\n\n## Future Research")
    if step.startswith("01ff_math_assess"):
        return "Finding 1: sign error in equation (2).\nVERDICT: CONFIRMED"
    return f"{ISSUE}\n"


def _make_pdf(path: Path, pages: int = 3) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(1, pages + 1):
        c.drawString(72, 720, f"A Test Paper on Rainfall, page {i}")
        c.drawString(72, 700, "We estimate y = a + b x with b = 0.5.")
        if i == pages:
            c.drawString(72, 680, "References")
            c.drawString(72, 660, "Smith, A. 2020. A Paper. Journal of Things 3: 1-2.")
        c.drawString(300, 40, str(i))
        c.showPage()
    c.save()


@pytest.fixture
def offline(monkeypatch):
    for mod in (core, stages, ref_checker):
        if hasattr(mod, "call_gemini"):
            monkeypatch.setattr(mod, "call_gemini", fake_call)
        if hasattr(mod, "call_model"):
            monkeypatch.setattr(mod, "call_model", fake_call)
    monkeypatch.setattr(stages, "extract_equations_mathpix",
                        lambda *a, **k: "$$y = a + b x$$", raising=False)
    monkeypatch.setattr(mathpix, "extract_equations_mathpix",
                        lambda *a, **k: "$$y = a + b x$$", raising=False)

    class _Resp:
        status_code = 404
        ok = False
        text = ""

        def json(self):
            return {}

    monkeypatch.setattr(ref_checker.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test")


def test_full_offline_run(tmp_path, offline):
    paper = tmp_path / "paper.pdf"
    _make_pdf(paper)
    supp = tmp_path / "appendix.pdf"
    _make_pdf(supp, pages=1)
    code_dir = tmp_path / "replication"
    code_dir.mkdir()
    (code_dir / "analysis.do").write_text("reg y x, cluster(state)\n")
    (code_dir / "README.txt").write_text("Run analysis.do.\n")

    work = tmp_path / "work"
    report = pipeline.run(
        paper, work, math=True, code=True, copyedit=True, editor_note=True,
        supp_pdfs=[supp], code_dir=code_dir, note="Focus on identification.",
    )
    text = Path(report).read_text(encoding="utf-8")
    assert "identification strategy" in text
    for stage_file in ["01o_summarizer.txt", "02i_list_v2.txt", "03c_list_v3.txt",
                       "01fg_math_list.txt", "01n_code_list.txt", "04b_data_editor.txt",
                       "08_math_advisor.txt", "09c_copyedit.txt", "10_latex_body.txt"]:
        assert (work / stage_file).exists(), stage_file


def test_page_map_builds(tmp_path):
    """The page map must build (it is wrapped in a try/except in the pipeline,
    so a failure there would otherwise pass silently)."""
    from reviewer2 import pagemap

    paper = tmp_path / "paper.pdf"
    _make_pdf(paper, pages=4)
    m = pagemap.build_page_map(str(paper))
    assert m["confidence"] in ("high", "medium", "low")
    assert m["method"]


def test_resume_and_stop(tmp_path, offline):
    paper = tmp_path / "paper.pdf"
    _make_pdf(paper)
    work = tmp_path / "work"
    out = pipeline.run(paper, work, copyedit=False, editor_note=False,
                       references=False, stop_stage=2.9)
    assert Path(out) == work.resolve()
    assert (work / "02i_list_v2.txt").exists()
    assert not (work / "03a_checker_1.txt").exists()

    report = pipeline.run(paper, work, copyedit=False, editor_note=False,
                          references=False, start_stage=3.0)
    assert Path(report).suffix == ".txt"
    assert (work / "03a_checker_1.txt").exists()
