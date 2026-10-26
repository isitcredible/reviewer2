"""Command-line interface for reviewer2."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewer2",
        description="Adversarial peer review for academic PDFs, powered by Google Gemini.",
    )
    parser.add_argument("pdf", help="Path to the PDF to review.")
    parser.add_argument(
        "-o", "--output", default="report.txt",
        help="Output text file path. Default: report.txt",
    )

    parser.add_argument(
        "--math", action="store_true",
        help="Enable the math-audit stages. Requires MATHPIX_APP_ID and MATHPIX_APP_KEY.",
    )
    parser.add_argument(
        "--code-dir", default=None, metavar="PATH",
        help="Path to a directory of replication source code. Passing this "
             "flag enables the code-audit add-on (off by default).",
    )
    parser.add_argument(
        "--no-copyedit", action="store_true",
        help="Skip the copyedit stages (proofreading and revision suggestions).",
    )
    parser.add_argument(
        "--no-editor-note", action="store_true",
        help="Skip the editor's-note stage (the Alchemist's revision advice). "
             "In this release, copyedit and editor's note are coupled: disabling "
             "either skips all Writer-Mode stages.",
    )
    parser.add_argument(
        "--no-references", action="store_true",
        help="Skip the reference check (bibliography lookups against OpenAlex, "
             "Crossref and OpenLibrary, plus a grounded web search).",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Base review only. Disables every add-on: math, code, copyedit, editor's note.",
    )
    parser.add_argument(
        "--note", default="", metavar="TEXT",
        help="Instructions to the reviewer, prepended to the paper as an extra first page.",
    )
    parser.add_argument(
        "--note-file", default=None, metavar="PATH",
        help="Read the reviewer note from a text file.",
    )
    parser.add_argument(
        "--supp", action="append", default=[], metavar="PDF",
        help="Path to a supplementary PDF to merge after the main paper. Repeatable.",
    )
    parser.add_argument(
        "--citation", default="",
        help="Manual citation string, used only if metadata extraction fails to produce one.",
    )

    parser.add_argument(
        "--work-dir", default=None,
        help="Directory for intermediate stage outputs. Default: a temp dir that is "
             "cleaned up on success. Pass an explicit path to keep the outputs or to "
             "resume a previous run.",
    )
    parser.add_argument(
        "--keep-work-dir", action="store_true",
        help="Keep the intermediate stage outputs after the run completes.",
    )
    parser.add_argument(
        "--start-stage", type=float, default=0.0, metavar="N",
        help="Resume from stage N in an existing --work-dir (e.g. 2.0). Earlier "
             "stages are read back from disk. Default: 0 (fresh run).",
    )
    parser.add_argument(
        "--stop-stage", type=float, default=100.0, metavar="N",
        help="Stop after the last stage numbered N or below (e.g. 2.9 stops after stage 2). Default: run to the end.",
    )
    parser.add_argument(
        "--skip-size-check", action="store_true",
        help="Bypass the 1000-page combined-volume limit.",
    )
    return parser


def resolve_addons(args: argparse.Namespace) -> dict[str, bool]:
    """Collapse CLI flags into a stage-enable map matching pipeline.run kwargs."""
    if args.base:
        return {"math": False, "code": False, "copyedit": False, "editor_note": False}
    return {
        "math": args.math,
        "code": bool(args.code_dir),
        "copyedit": not args.no_copyedit,
        "editor_note": not args.no_editor_note,
    }


def _require_env(var: str, because: str) -> None:
    if not os.environ.get(var):
        print(f"error: {var} not set. Required {because}.", file=sys.stderr)
        sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    _require_env("GEMINI_API_KEY", "for Gemini API access")
    note = args.note
    if args.note_file:
        note = Path(args.note_file).expanduser().read_text(encoding="utf-8")
    if args.start_stage > 0 and not args.work_dir:
        print("error: --start-stage needs the --work-dir of the run being resumed.", file=sys.stderr)
        return 2
    addons = resolve_addons(args)
    if addons["math"]:
        _require_env("MATHPIX_APP_ID", "for the --math add-on")
        _require_env("MATHPIX_APP_KEY", "for the --math add-on")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="reviewer2_"))
        cleanup_work_dir = not args.keep_work_dir

    # Defer import to keep --help fast and avoid pulling in genai unnecessarily.
    from reviewer2.pipeline import PipelineError, run

    try:
        final_txt = run(
            pdf_path=pdf_path,
            work_dir=work_dir,
            math=addons["math"],
            code=addons["code"],
            copyedit=addons["copyedit"],
            editor_note=addons["editor_note"],
            supp_pdfs=args.supp or None,
            code_dir=args.code_dir,
            references=not args.no_references,
            note=note,
            citation=args.citation,
            start_stage=args.start_stage,
            stop_stage=args.stop_stage,
            skip_size_check=args.skip_size_check,
        )
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted by user", file=sys.stderr)
        return 130

    if args.stop_stage < 100.0 and not (final_txt and Path(final_txt).is_file()):
        print(f"\n  stopped after stage {args.stop_stage}; outputs are in {work_dir}")
        return 0
    if final_txt and Path(final_txt).is_file():
        shutil.copy2(final_txt, output_path)
        print(f"\n✓ Report written to {output_path}")
    else:
        print(f"\nerror: pipeline did not produce a final report (stopped at {final_txt})", file=sys.stderr)
        return 1

    if cleanup_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif not args.work_dir:
        print(f"  (intermediate outputs kept at {work_dir})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
