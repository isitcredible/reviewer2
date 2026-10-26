# Changelog

## 2.0.0 — 2026-10-26

The final version of the pipeline that ran the hosted Reviewer 2 service,
which has now been retired.

### Added
- **Reference check.** Every bibliography entry is extracted from the PDF,
  verified by a second pass, and looked up in OpenAlex, Crossref and
  OpenLibrary, with a gated, grounded web search for what those cannot
  confirm. The report gains a Reference Check section listing entries that
  could not be confirmed. `--no-references` turns it off.
- **Page map.** Printed page numbers are mapped to PDF pages from the page
  labels and a text scan, reconciled by a model that looks at the pages, and
  validated deterministically. Stages cite printed pages, and a guard
  rejects citation corrections that point outside the document.
- **Proportionality stages** (02h, 02i) between the first issues list and
  the checks: they demote objections an author could dismiss from their own
  text and deepen the ones the argument rests on.
- **Math advisor** (08), with `--math` and the author-facing stages: for
  each confirmed math error, remedies ranked from least to most costly.
- **Target-parameter check** in the Breaker for empirical papers: whether
  the design recovers the parameter the conclusions invoke.
- `--note` / `--note-file`: instructions to the reviewer, added as a first
  page.
- `--start-stage` / `--stop-stage` for resuming a run in `--work-dir`.
- Optional routing of six clerical stages to DeepSeek V4 Flash
  (`DEEPSEEK_API_KEY`, `deepseek` extra).
- Optional compression of PDFs over Gemini's 50 MB limit (`pdf` extra).
- An offline end-to-end test (`pytest`).

### Changed
- **Math audit re-architected.** The single sober-check stage (01fd) is
  replaced by four: consolidation without judgment (01fd), a defence that
  can run Python for numerical checks (01fe), an assessment with the whole
  paper in context (01ff), and a write-up (01fg). Only false findings are
  discarded; real but minor errors stay.
- **Code audit reworked.** The checker (01m) now grounds each finding in the
  package rather than ruling on it; a deterministic gate then confirms that
  every quoted line of code exists in the cited file, relocating or dropping
  it otherwise; the final list (01n) frames findings as things to check.
  Data files and logs are excluded from the package; PDFs in the package are
  attached as pages; a copy of the paper inside the package is dropped.
- The red-team finders, math chain and code audit run in parallel, as do
  independent checks later on: a run takes roughly half as long.
- The combined page limit rises from 500 to 1,000 pages.
- 03c is renamed `03c_list_v3`; it now checks the proportioned list.
- Prompts updated throughout to their final hosted versions.

### Removed
- The `03c_list_v2` and `01fd_math_sober` prompts.
- The `tiktoken` dependency.

## 1.0.2 — 2026-04-21

- Mathpix requests now send `metadata.improve_mathpix: false`, opting out of
  Mathpix's data retention for QA/model-improvement. Source PDFs and
  intermediate outputs are still deleted explicitly via `DELETE /v3/pdf/{id}`
  after each extraction, as before.

## 1.0.1

- Earlier patch release.

## 1.0.0

- Initial public release.
