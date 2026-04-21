"""Mathpix PDF-to-LaTeX client for equation extraction."""

from __future__ import annotations

import json
import os
import time

import requests


def extract_equations_mathpix(pdf_path: str, page_ranges: str) -> str | None:
    """Send selected pages of a PDF to Mathpix; return markdown with LaTeX, or None on failure.

    Requires ``MATHPIX_APP_ID`` and ``MATHPIX_APP_KEY`` in the environment.
    """
    app_id = os.getenv("MATHPIX_APP_ID")
    app_key = os.getenv("MATHPIX_APP_KEY")

    if not app_id or not app_key:
        print("  ⚠  Mathpix credentials not found (MATHPIX_APP_ID / MATHPIX_APP_KEY). Skipping equation extraction.")
        return None

    headers = {"app_id": app_id, "app_key": app_key}

    try:
        with open(pdf_path, "rb") as f:
            r = requests.post(
                "https://api.mathpix.com/v3/pdf",
                headers=headers,
                files={"file": ("paper.pdf", f, "application/pdf")},
                data={
                    "options_json": json.dumps({
                        "conversion_formats": {"md": True},
                        "math_inline_delimiters": ["$", "$"],
                        "math_display_delimiters": ["$$", "$$"],
                        "rm_spaces": True,
                        "rm_fonts": True,
                        "include_equation_tags": True,
                        "page_ranges": page_ranges,
                        "metadata": {"improve_mathpix": False},
                    })
                },
                timeout=30,
            )

        result = r.json()
        pdf_id = result.get("pdf_id")
        if not pdf_id:
            print(f"  ⚠  Mathpix submission failed: {result}")
            return None

        print(f"    ↳ Mathpix PDF ID: {pdf_id} (pages: {page_ranges})")

        for _ in range(60):
            r = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}", headers=headers, timeout=15)
            status = r.json()
            if status.get("status") == "completed":
                break
            time.sleep(3)
        else:
            print("  ⚠  Mathpix timed out after 3 minutes.")
            _cleanup(headers, pdf_id)
            return None

        r = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}.md", headers=headers, timeout=30)
        md = r.text
        _cleanup(headers, pdf_id)
        return md

    except Exception as e:
        print(f"  ⚠  Mathpix error: {e}")
        return None


def _cleanup(headers: dict, pdf_id: str) -> None:
    try:
        requests.delete(f"https://api.mathpix.com/v3/pdf/{pdf_id}", headers=headers, timeout=10)
    except Exception:
        pass
