# Changelog

## 1.0.2 — 2026-04-21

- Mathpix requests now send `metadata.improve_mathpix: false`, opting out of
  Mathpix's data retention for QA/model-improvement. Source PDFs and
  intermediate outputs are still deleted explicitly via `DELETE /v3/pdf/{id}`
  after each extraction, as before.

## 1.0.1

- Earlier patch release.

## 1.0.0

- Initial public release.
