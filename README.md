# reviewer2

Adversarial review of academic PDFs, powered by Google Gemini.

`reviewer2` is the pipeline that ran the Reviewer 2 service at
[isitcredible.com](https://isitcredible.com/), in the final form it took
there. The hosted service has been retired; this repository is now the way to
run it. It produces a plain-text critical review of a PDF manuscript through a
chain of more than 30 LLM calls. The chain is built around one idea:
aggressive prompting is what gets a language model to read a paper carefully,
and a verification cascade is what removes the hallucinations aggression
produces.

It is best used by authors on their own work, before submission: a demanding
first reader that finds what a hostile referee would. Its findings are leads
to check, not verdicts.

The accompanying paper, *Yell at It: Prompt Engineering for Automated Peer
Review*, explains the design of version 1 and reports benchmark comparisons
against five alternatives (15 wins, 4 ties, 1 loss across 20 pairings). It is
in [`paper/yellatit.pdf`](paper/yellatit.pdf). Version 2 is described below
and in the [changelog](CHANGELOG.md).

**Version:** 2.0.0.

## Requirements

- Python 3.10+
- A Google Gemini API key ([aistudio.google.com](https://aistudio.google.com/))
- (Optional) A Mathpix account, for the math-audit add-on

## Install

```bash
pip install git+https://github.com/isitcredible/reviewer2
```

Two optional extras:

```bash
pip install "reviewer2[pdf] @ git+https://github.com/isitcredible/reviewer2"
pip install "reviewer2[deepseek] @ git+https://github.com/isitcredible/reviewer2"
```

- `pdf` installs PyMuPDF, which shrinks PDFs larger than Gemini's 50 MB upload
  limit (by dropping photographs, never text or tables) and gives a slightly
  better page-number scan. PyMuPDF is licensed under the AGPL-3.0; without it
  the pipeline uses pypdf and uploads large PDFs unchanged.
- `deepseek` installs the client for the optional DeepSeek routing (see
  Configuration).

Or clone for development:

```bash
git clone https://github.com/isitcredible/reviewer2
cd reviewer2
pip install -e ".[pdf,dev]"
pytest
```

The tests run the whole pipeline offline, with canned model responses.

## Quickstart

```bash
export GEMINI_API_KEY=your_key_here
reviewer2 paper.pdf -o report.txt
```

A default run takes 15 to 45 minutes and costs a few dollars in Gemini
usage. It produces the core review, the reference check, and the
author-facing editor's note and copyedit. The math and code audits are
opt-in (see Add-ons).

## Input files

- **The main PDF.** The positional argument; always required.

  ```bash
  reviewer2 paper.pdf
  ```

- **Supplementary PDFs.** Pass `--supp PATH` for each supplementary file;
  the flag is repeatable. Supplements are merged after the main paper and
  visible to every stage except the math Proofreader, which reads the
  main-only PDF so that equation OCR is not confused by material from the
  supplements.

  ```bash
  reviewer2 paper.pdf --supp appendix.pdf --supp online_appendix.pdf
  ```

- **A replication package.** Pass `--code-dir PATH` to enable the
  code-audit add-on. Code and text files (R, Python, Stata, MATLAB and so
  on) and Word documents are compiled into one PDF with a map of the
  package; PDFs in the package are attached as pages; data files and logs
  are skipped; a copy of the paper itself inside the package is dropped.

  ```bash
  reviewer2 paper.pdf --code-dir ./replication/
  ```

- **A note to the reviewer.** `--note TEXT` or `--note-file PATH` adds an
  instruction page in front of the paper: what to focus on, what has
  changed since a previous draft, what the reader should know.

The run stops if the paper, supplements and compiled package together
exceed 1,000 pages. Override with `--skip-size-check`.

## What the report contains

- **Is It Credible?** A short essay (up to 700 words) answering the
  titular question.
- **The Bottom Line.** Three to five sentences on whether the paper's
  contributions hold.
- **Potential Issues.** A verified list of the problems found.
- **Future Research.** Constructive proposals that follow from the
  critique.
- **Reference Check.** Every entry in the paper's bibliography looked up in
  OpenAlex, Crossref and OpenLibrary, with a grounded web search for what
  those cannot confirm. Entries that could not be confirmed are listed for
  checking by hand; the section never asserts that a reference is
  fabricated.
- **Data Editor** (with `--code-dir`). A report on the replication package,
  with each finding framed as something for the author to check.
- **Editor's Note, Copyediting, Proofreading** (on by default). Advice to
  the author on how to revise, specific revision suggestions, and language
  errors.

## How it works

Each stage is a prompt in `src/reviewer2/prompts/`. Stage outputs are
written as `.txt` files into a working directory, and independent stages
run in parallel.

### 0. Ingestion

Metadata and the paper's claimed contributions are extracted. A page map
records which printed page number each PDF page carries, built from the
PDF's page labels and a text scan, corrected by a model that looks at the
pages, and validated deterministically. Stages then cite printed page
numbers, and a later guard rejects any citation correction that points
outside the document. The reference check starts in the background.

### 1. Red Team: adversarial agents

Each agent attacks the paper from a different angle, prompted to find
problems rather than verify the author's claims.

- **The Breaker** targets intellectual foundations: whether the theoretical
  framework predetermines the findings, whether disputed premises are
  treated as obvious, whether causal design labels are earned, and whether
  the design recovers the parameter the conclusions invoke.
- **The Butcher** dissects the empirical machinery: whether the method can
  answer the question, whether measures capture the constructs, whether
  robustness checks threaten anything, whether effect sizes matter.
- **The Shredder** audits procedural claims against documentation: whether
  sample sizes cohere across methods, results and tables; whether
  randomisation and pre-registration claims match what is reported.
- **The Collector** returns to every location the Butcher and Shredder
  flagged and picks up footnotes, table notes and cross-references they
  missed. It cannot raise new issues.
- **The Void** catalogues what the paper does not say: unmeasured
  confounds, reverse causation, selection, untested alternatives.

For non-empirical papers, the Butcher and Shredder are replaced by two
further Breaker rounds that probe theoretical and mathematical arguments.

The **math audit** (`--math`) runs alongside: a Re-Deriver, a Proofreader
and an Auditor attack the mathematics; their findings are consolidated
without judgment; a defence stage argues the paper's side and can run
Python to check claims numerically; an assessment stage rules on each
finding with the whole paper in view; and only findings it confirms are
written up. Discarded means false, not minor: a real but small error stays.

The **code audit** (`--code-dir`) also runs alongside: three agents hunt for
divergences between paper and code, bugs, and data-handling problems; a
grounding stage checks each finding against the package; then a
deterministic check confirms that every quoted line of code exists in the
file cited, moves it to the right file if it is elsewhere, and drops it if
it is nowhere.

### 2. Filtering

The Red Team produces both real findings and the hallucinations that
aggressive prompting introduces. The rest of the pipeline tells them apart.

- **Numbers** checks the Red Team's mathematical claims against the paper.
- **The Blue Team** argues the paper's defence against every finding.
- **The Assessment** rules on each: Red Team right, Blue Team right, or
  debatable.
- **List v1** checks quotations in context and removes false positives.
- **Proportionality** demotes cheap shots an author could dismiss by
  pointing at their own text, and deepens the issues the argument rests on.
  **List v2** finalises the proportioned list.

### 3. Checks and writing

- **The Fact Checker** and **External Check** confirm page numbers,
  quotations and equations against the PDF, and audit claims about outside
  sources.
- **The Reviewer** writes the assessment from the verified list.
- **Review Checker**, **Citation Verifier** and **Reviser** check the
  assembled review against the PDF and correct it.
- **The Legal pass** removes imputed intent, fraud accusations, competence
  attacks and unsubstantiated absolutes. **The Formatter** cleans the
  structure.
- **Data Editor** writes the code-audit section.
- *(On by default.)* **The Alchemist** advises the author how to revise
  (with a **Math Advisor** routing each confirmed math error to its least
  costly remedy); **The Polisher** rewrites the advice in an editorial
  voice; **The Proofreader** and **Copy-Editor** produce language
  corrections and revision suggestions.

## Options

| Flag                  | Default | What it does |
|-----------------------|---------|--------------|
| `--math`              | off     | Math audit (see above). Requires Mathpix credentials. |
| `--code-dir PATH`     | off     | Replication-package audit (see above). |
| `--no-references`     | (on)    | Skip the reference check. |
| `--no-copyedit`       | (on)    | Skip the author-facing stages. |
| `--no-editor-note`    | (on)    | Skip the author-facing stages. Copyedit and editor's note share inputs, so disabling either skips both. |
| `--base`              | off     | Core review only: no math, code, copyedit or editor's note. |
| `--note`, `--note-file` | none  | Instructions to the reviewer, added as a first page. |
| `--work-dir PATH`     | temp    | Keep every stage output in `PATH`. |
| `--start-stage N`     | 0       | Resume a run in `--work-dir` from stage `N` (e.g. `2.0`, `4.0`). |
| `--stop-stage N`      | end     | Stop after the last stage numbered `N` or below (e.g. `2.9` stops after filtering). |
| `--skip-size-check`   | off     | Allow more than 1,000 combined pages. |

To resume an interrupted run, rerun the same command with the same
`--work-dir` and a `--start-stage` at or before the first stage that did not
finish; earlier outputs are read back from disk. Stage numbers: 0
ingestion, 1 Red Team and audits, 2 filtering, 3 checks, 4 reviewer, 5
revision, 6 legal, 7 format, 8–9 author-facing stages.

## Configuration

| Variable                   | Required           | Purpose |
|----------------------------|--------------------|---------|
| `GEMINI_API_KEY`           | yes                | Google Gemini API key. |
| `MATHPIX_APP_ID`, `MATHPIX_APP_KEY` | only with `--math` | Mathpix credentials. |
| `REVIEWER2_CONTACT_EMAIL`  | no                 | Your email, sent with OpenAlex and Crossref lookups so the reference check uses their faster "polite pool". |
| `SEMANTIC_SCHOLAR_API_KEY` | no                 | Adds Semantic Scholar as a lookup source for the reference check (its keyless pool is too rate-limited to use). |
| `DEEPSEEK_API_KEY`         | no                 | Route six clerical, text-only stages to DeepSeek V4 Flash, which is cheaper. Falls back to Gemini on any error. Needs the `deepseek` extra. |
| `GEMINI_MODEL_OVERRIDE`    | no                 | Force every stage to one model (e.g. `flash_lite_3`) for cheap test runs. Review quality drops sharply. |
| `REVIEWER2_PROMPTS_DIR`    | no                 | Load prompts from another directory, for experimenting. |

## Cost

A default run costs a few dollars in Gemini API usage; a per-stage cost
breakdown is printed at the end. `--math` and `--code-dir` each add a few
more. Model prices are in `src/reviewer2/data/pricing.csv`; update them if
Google's change.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The pipeline and prompts in this repository are the result of many months
of empirical tuning by The Catalogue of Errors Ltd.

The names "Reviewer 2", "isitcredible.com", and "The Catalogue of Errors"
are trademarks and are not licensed under Apache 2.0. Forks must use a
different name.
