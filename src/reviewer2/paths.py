"""Path resolution for packaged data (prompts, pricing CSV).

Lets callers override the packaged defaults via environment variables, useful
for iterating on prompts outside an editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def prompts_dir() -> Path:
    """Directory containing stage prompts. Override: ``REVIEWER2_PROMPTS_DIR``."""
    override = os.environ.get("REVIEWER2_PROMPTS_DIR")
    if override:
        return Path(override)
    return _PACKAGE_DIR / "prompts"


def pricing_csv() -> Path:
    """CSV mapping Gemini model → $/M-tokens. Override: ``REVIEWER2_PRICING_CSV``."""
    override = os.environ.get("REVIEWER2_PRICING_CSV")
    if override:
        return Path(override)
    return _PACKAGE_DIR / "data" / "pricing.csv"
