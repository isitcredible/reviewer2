# reviewer2

Adversarial peer review for academic PDFs, powered by Google Gemini.

`reviewer2` runs a multi-stage "Red Team" pipeline over a PDF manuscript and
produces a plain-text critical review that surfaces methodological errors,
data inconsistencies, and logical flaws. It is the open-source pipeline behind
the paid service at [isitcredible.com](https://isitcredible.com/).

**Status:** pre-release (0.1.0). APIs will change.

## Requirements

- Python 3.10+
- A Google Gemini API key ([aistudio.google.com](https://aistudio.google.com/))
- (Optional) A Mathpix account, for the math-audit add-on
- `qpdf` on `PATH`, for PDF preprocessing

## Install

```bash
pip install git+https://github.com/isitcredible/reviewer2
```

Or clone for development:

```bash
git clone https://github.com/isitcredible/reviewer2
cd reviewer2
pip install -e .
```

## Quickstart

```bash
export GEMINI_API_KEY=your_key_here
reviewer2 paper.pdf -o report.txt
```

By default this runs the core review plus the code-audit, copyedit, and
editor's note stages. The math-audit add-on is off by default because it
requires a separate [Mathpix](https://mathpix.com/) account; enable it with
`--math`:

```bash
export MATHPIX_APP_ID=...
export MATHPIX_APP_KEY=...
reviewer2 paper.pdf --math -o report.txt
```

For a minimal run that skips every add-on:

```bash
reviewer2 paper.pdf --base -o report.txt
```

## Add-ons

All add-ons except `--math` are on by default.

| Flag                | Default | What it does                                         | Extra cost |
|---------------------|---------|------------------------------------------------------|------------|
| `--math`            | off     | Math-audit stages. Requires Mathpix OCR credentials. | Mathpix API + extra Gemini calls |
| `--no-code`         | —       | Skip code / replication-audit stages.                | — |
| `--no-copyedit`     | —       | Skip copyedit / polish stages.                       | — |
| `--no-editor-note`  | —       | Skip the editor's note stage.                        | — |
| `--base`            | off     | Core review only. Disables every add-on.             | Cheapest run |

## Configuration

Environment variables:

| Variable                | Required        | Purpose                                           |
|-------------------------|-----------------|---------------------------------------------------|
| `GEMINI_API_KEY`        | yes             | Google Gemini API key.                            |
| `GEMINI_MODEL_OVERRIDE` | no              | Force all stages to a specific model (e.g. `gemini-3.1-flash-lite-preview` for cheap smoke runs). |
| `MATHPIX_APP_ID`        | only with `--math` | Mathpix app ID.                                |
| `MATHPIX_APP_KEY`       | only with `--math` | Mathpix app key.                               |

## Cost

Each full review costs a few dollars of Gemini API usage (more with `--math`
and `--code`). Use `GEMINI_MODEL_OVERRIDE` with a cheaper model for
experimentation; note that smaller models significantly degrade review quality
on the harder stages.

## How it works

The pipeline runs 30+ LLM stages organized as a directed graph: metadata
extraction, adversarial critique generation, filtering and evaluation,
fact-checking, review drafting, and final assembly. Each stage is a prompt in
the `prompts/` directory. Stage outputs are written as `.txt` files into a
working directory and chained together.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the stage graph and dependencies.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The pipeline and prompts in this repository are the result of many months of
empirical tuning by The Catalogue of Errors Ltd. Contributions that improve
review quality on benchmark papers are welcome.

The names "Reviewer 2", "isitcredible.com", and "The Catalogue of Errors" are
trademarks and are not licensed under Apache 2.0. Forks must use a different
name.
