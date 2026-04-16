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
        "--no-code", action="store_true",
        help="Skip the code / replication-audit stages.",
    )
    parser.add_argument(
        "--no-copyedit", action="store_true",
        help="Skip the copyedit / polish stages.",
    )
    parser.add_argument(
        "--no-editor-note", action="store_true",
        help="Skip the editor's note stage. (In this release, copyedit and editor's "
             "note are coupled — disabling either skips all Writer-Mode stages.)",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Base review only. Disables every add-on: math, code, copyedit, editor's note.",
    )
    parser.add_argument(
        "--code-dir", default=None,
        help="Directory of source-code files for the code-audit add-on.",
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
        "--skip-size-check", action="store_true",
        help="Bypass the 500-page combined-volume circuit breaker.",
    )
    return parser


def resolve_addons(args: argparse.Namespace) -> dict[str, bool]:
    """Collapse CLI flags into a stage-enable map matching pipeline.run kwargs."""
    if args.base:
        return {"math": False, "code": False, "copyedit": False, "editor_note": False}
    return {
        "math": args.math,
        "code": not args.no_code,
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
    addons = resolve_addons(args)
    if addons["math"]:
        _require_env("MATHPIX_APP_ID", "for the --math add-on")
        _require_env("MATHPIX_APP_KEY", "for the --math add-on")
    if addons["code"] and not args.code_dir:
        print("note: --code is enabled but no --code-dir given; code-audit stages will be skipped.",
              file=sys.stderr)

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
            citation=args.citation,
            skip_size_check=args.skip_size_check,
        )
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted by user", file=sys.stderr)
        return 130

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
