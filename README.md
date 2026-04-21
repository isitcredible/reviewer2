# reviewer2

Adversarial peer review for academic PDFs, powered by Google Gemini.

`reviewer2` is the open-source pipeline behind the paid service at
[isitcredible.com](https://isitcredible.com/). It produces a plain-text
critical review of a PDF manuscript through a 30+ stage chain of LLM calls.
The chain is built around one idea: aggressive prompting is what gets a
language model to read a paper carefully, and a verification cascade is what
removes the hallucinations aggression produces.

The accompanying paper, *Yell at It: Prompt Engineering for Automated Peer
Review*, explains the design and reports benchmark comparisons against five
alternatives (15 wins, 4 ties, 1 loss across 20 pairings). It is in
[`paper/yellatit.pdf`](paper/yellatit.pdf).

**Version:** 1.0.2. The hosted service at isitcredible.com will continue to
evolve past this snapshot; breaking changes here will bump the major version
per semver.

## Requirements

- Python 3.10+
- A Google Gemini API key ([aistudio.google.com](https://aistudio.google.com/))
- `qpdf` on `PATH`, for PDF preprocessing
- (Optional) A Mathpix account, for the math-audit add-on

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

A default run takes 15 to 45 minutes and costs a few dollars in Gemini
usage. It produces the core review plus the author-facing copyedit and
editor's-note stages. The math and code audits are opt-in (see Add-ons).

## Input files

Three kinds of input can be passed in.

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

- **A replication-code directory.** Pass `--code-dir PATH` to enable the
  code-audit add-on. Any text source files in the directory (R, Python,
  Stata and so on) are compiled into a single PDF and attached behind the
  paper for the code-audit stages.

  ```bash
  reviewer2 paper.pdf --code-dir ./replication/
  ```

A volume circuit breaker halts the run if the combined page count of
paper, supplements and compiled code exceeds 500 pages. Override with
`--skip-size-check` when a legitimately large package needs to get
through.

## What the report contains

The default report has four sections:

- **Is It Credible?** A short essay (up to 700 words) answering the
  titular question.
- **The Bottom Line.** Three to five sentences on whether the paper's
  contributions hold.
- **Potential Issues.** A verified, categorised list of the problems found
  during review.
- **Future Research.** Constructive proposals that follow from the
  critique.

With the copyedit and editor's-note stages enabled (the default), the
report also carries an author-facing editor's note with revision advice.

## How it works

The pipeline has three phases. Each stage is a prompt in `prompts/`; stage
outputs are written as `.txt` files into a working directory and chained
together, so a run can be resumed by pointing `--work-dir` at an existing
folder.

### 1. Red Team: five adversarial agents

Each agent attacks the paper from a different angle, prompted to find
problems rather than verify the author's claims.

- **The Breaker** targets intellectual foundations: whether the theoretical
  framework predetermines the findings, whether disputed premises are
  treated as obvious, whether causal design labels (difference-in-differences,
  instrumental variables) are earned rather than merely claimed.
- **The Butcher** dissects the empirical machinery: whether the method can
  answer the question, whether measures capture the theoretical constructs,
  whether robustness checks threaten anything, whether reported effect
  sizes are large enough to matter.
- **The Shredder** audits procedural claims against documentation: whether
  sample sizes cohere across methods, results and tables; whether blinding
  and randomisation are described rather than asserted; whether
  pre-registration claims match reported outcomes.
- **The Collector** returns to every location the Butcher and Shredder
  flagged, reads within two pages for footnotes, table notes and
  cross-references, and picks up what the attackers missed. It cannot raise
  new issues.
- **The Void** catalogues what the paper does not say: unmeasured
  confounds, reverse causation, selection bias, alternative explanations
  the authors never tested.

For non-empirical papers, the Butcher and Shredder are replaced by a second
Breaker pass that probes theoretical and mathematical arguments more
aggressively.

### 2. Filtering: Blue Team and verification

The Red Team produces both real findings and the sycophancy-driven
hallucinations that aggressive prompting introduces. The rest of the
pipeline exists to tell them apart.

- **The Blue Team** argues the paper's defence, presenting the strongest
  possible counterargument for each Red Team finding.
- **The Assessment** stage rules on each finding: Red Team correct, Blue
  Team correct, or debatable. Mathematical claims are independently
  re-derived before they pass.
- **The Fact Checker** and **External Check** confirm page numbers,
  quotations and equations against the PDF, and audit citations of outside
  sources. Every external citation is assumed wrong until confirmed.
- **Review Checker**, **Citation Verifier** and **Reviser** apply the same
  discipline to the assembled review before it is finalised.

### 3. Writing: review and author-facing advice

- **The Reviewer** drafts the credibility assessment from the verified
  issues list.
- **The Legal pass** removes defamatory or legally risky phrasing: imputed
  intent, fraud accusations, competence attacks, unsubstantiated
  absolutes.
- **The Formatter** cleans up structure and markdown.
- *(Optional, on by default.)* **The Alchemist** advises the author how
  to revise, with a defence plan and suggested caveats; **The Polisher**
  rewrites it in an editorial voice; **The Proofreader** and
  **Copy-Editor** produce specific revision suggestions and catch language
  errors.

The appendix of the accompanying paper lists every stage in execution order
with the Gemini model assigned to each.

## Add-ons

| Flag                 | Default | What it does                                                                                       | Extra cost |
|----------------------|---------|----------------------------------------------------------------------------------------------------|------------|
| `--math`             | off     | Math audit. Four stages re-derive key results, check text-to-equation consistency, audit the mathematical framework, and consolidate through an independent sober re-check on MathPix OCR text. Requires Mathpix OCR credentials. | Mathpix API + extra Gemini calls |
| `--code-dir PATH`    | off     | Replication-code audit. Three agents examine the code (Divergence Hunter for claims-vs-code, Bug Hunter, Data Archaeologist), followed by a verification pass that tests each finding against the actual code. Pass a directory of source files to enable. | Several extra Gemini calls |
| `--no-copyedit`      | (on)    | Skip the copyedit stages (proofreading and specific revision suggestions).                         | — |
| `--no-editor-note`   | (on)    | Skip the editor's-note stage (the Alchemist's revision advice).                                    | — |
| `--base`             | off     | Disable every add-on. Produces the core review only.                                               | Cheapest run |

Copyedit and editor's note share inputs and are coupled in this release:
disabling either skips both.

## Configuration

Environment variables:

| Variable                | Required           | Purpose                                                                 |
|-------------------------|--------------------|-------------------------------------------------------------------------|
| `GEMINI_API_KEY`        | yes                | Google Gemini API key.                                                  |
| `GEMINI_MODEL_OVERRIDE` | no                 | Force all stages to a specific model (e.g. `gemini-3.1-flash-lite-preview` for cheap smoke runs). |
| `MATHPIX_APP_ID`        | only with `--math` | Mathpix app ID.                                                         |
| `MATHPIX_APP_KEY`       | only with `--math` | Mathpix app key.                                                        |

## Cost

A default run costs a few dollars in Gemini API usage. `--math` and
`--code-dir` each add a few more. `GEMINI_MODEL_OVERRIDE` lets you run the
whole pipeline for well under a dollar as a smoke test, but smaller models
significantly degrade review quality on the harder stages.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The pipeline and prompts in this repository are the result of many months
of empirical tuning by The Catalogue of Errors Ltd. Contributions that
improve review quality on benchmark papers are welcome.

The names "Reviewer 2", "isitcredible.com", and "The Catalogue of Errors"
are trademarks and are not licensed under Apache 2.0. Forks must use a
different name.
