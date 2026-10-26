# reviewer2 — project notes

## Running the pipeline from Claude Code

**Before invoking `reviewer2` on a real paper, confirm with the user.** A
default run takes 15–45 minutes and costs a few dollars in Gemini API
usage. That is not a routine command to fire off autonomously.

Entry point:

```bash
reviewer2 paper.pdf -o report.txt
```

Requires `GEMINI_API_KEY` in the environment. Install for development with
`pip install -e ".[pdf,dev]"` from the repo root; `pytest` runs the whole
pipeline offline with canned model responses.

Because runs are long, start them with `run_in_background: true` rather
than foreground. Outputs stream to the working directory (default: a temp
dir, override with `--work-dir`). To resume, rerun with the same
`--work-dir` and `--start-stage N` at or before the first unfinished stage.

Flags worth knowing: `--math` needs `MATHPIX_APP_ID` and
`MATHPIX_APP_KEY`; `--code-dir PATH` enables the replication-code audit;
`--no-references` skips the bibliography lookups; `--base` disables all
add-ons for the cheapest real run; the 1,000-page volume limit is
overridden with `--skip-size-check`.

## Versioning

Semver. When bumping the version, update **all four** in lockstep:

- `pyproject.toml` (`version = "..."`)
- `src/reviewer2/__init__.py` (`__version__ = "..."`)
- `README.md` (the `**Version:**` line)
- `CHANGELOG.md` (new section at top, dated, per Keep-a-Changelog style)

These drifted apart between the initial commit and v1.0.1 (pyproject was at
1.0.0 while `__init__.py` was still at 0.1.0). Grep for the old version
before committing to catch any that were missed.

Tag as `vX.Y.Z` after the release commit, then `git push && git push --tags`.

Bump size is a judgment call, not a mechanical rule. A change that technically
alters a default but has no real dependents yet is a patch, not a minor.
Breaking changes bump the major version.
