"""Reference checker.

Extracts the text's OWN bibliography and looks each entry up in scholarly
records, then writes a standalone "Reference Check" section for the report.
It is a TRIAGE FLAGGER, not an adjudicator: it never asserts a citation is
fabricated; an unlocatable entry is "could not be confirmed", a human makes
the call.

Runs on a background thread (pipeline.py) so the slow throttled API lookups
overlap the main Gemini pipeline. Feeds nothing downstream.

Set REVIEWER2_CONTACT_EMAIL to your address to join the OpenAlex and Crossref
"polite pools" (faster, more reliable lookups); without it requests are sent
anonymously.

Pipeline of work:
  1. EXTRACT the bibliography from the PDF with Gemini Flash 2.5 reading the
     document directly. A multimodal pass sees the page layout, so it knows
     where the reference list ends and STOPS there — it does not run on into
     appendices, footnotes, or proofs (the old PyMuPDF "heading-to-end-of-file"
     slice swept those in and the parser fabricated entries from prose).
  2. VERIFY the extraction with a second Flash 2.5 pass against the same PDF:
     it returns any bibliography entries the first pass MISSED and flags any
     SPURIOUS items that are not genuine references. Merge the corrections.
  3. LOOK each entry up: OpenAlex + Crossref (free, fast) first. These article
     indexes are near-complete for journals and PATCHY for monographs, so for a
     non-article they cannot confirm we fall to OpenLibrary — the independent
     Internet-Archive book catalogue that actually holds scholarly books and the
     edited volumes chapters sit in. OpenLibrary is a deterministic, echo-proof
     source (a catalogue record, not a model's say-so), and because it runs on
     separate infrastructure it also stops a single index's bad day from dumping a
     shelf of real books into the flag list. Only what NONE of these confirm goes
     to a real web search (Gemini + Google Search grounding), which is gated to
     non-article types and an OpenLibrary aboutness check. Semantic Scholar is
     wired in as an extra index but only fires when SEMANTIC_SCHOLAR_API_KEY is set
     (its keyless pool 429s on the first request). A printed DOI is the strong
     case; otherwise a match must be corroborated by author or year before it
     counts — a similar title alone is never treated as a match, and a fuzzy
     near-title is never called a miscite.
  4. WRITE the section in the report voice (Gemini Pro, prompts/reference_check.txt).
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher

import requests

from reviewer2.core import call_gemini, load_prompt, save_output
from reviewer2.helpers import get_citation_block

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    raw: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    type: str | None = None


@dataclass
class Candidate:
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    venue: str | None
    source: str  # 'crossref' | 'openalex' | 'semanticscholar'


@dataclass
class LookupResult:
    reference: Reference
    verdict: str  # 'CONFIRMED' | 'FLAG'
    reason: str
    title_sim: float = 0.0
    best: Candidate | None = None


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9 ]+", " ", _fold(s))
    return re.sub(r"\s+", " ", s).strip()


def _title_sim(a: str | None, b: str | None) -> float:
    """Symmetric title similarity in [0, 1]. Uses the max of a sequence ratio
    and a token-set F1. F1 penalises BOTH missing and EXTRA words, so a short
    title that is merely a substring of a longer, different one ("A Theory of
    Trade" vs "A Theory of *International* Trade") no longer scores a spurious 1.0 — that asymmetric containment
    bug is what produced false "miscite" flags."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    f1 = (2 * len(ta & tb) / (len(ta) + len(tb))) if (ta and tb) else 0.0
    return max(ratio, f1)


def _book_title_from_venue(venue: str | None) -> str:
    """For a chapter, the extractor packs the containing book as
    "Book Title, Place: Publisher" into ``venue``. Return just the book title by
    stripping a trailing ", <Place>: <Publisher>" clause. A plain venue — a
    journal name, or a book title with no publisher tail — is returned unchanged.
    Conservative: it strips ONLY when a "Place: Publisher" tail is clearly present
    (a capitalised place, a colon, then the publisher), so a journal such as
    "Journal of Law, Economics & Policy" is left alone, and a subtitle colon
    ("…Trade: New Evidence…") is not mistaken for a publisher because it is
    not preceded by a comma + capitalised place."""
    v = (venue or "").strip()
    if not v:
        return v
    m = re.search(r",\s*[A-Z][\w.&'-]*(?:[ ,]+[A-Z][\w.&'-]*)*:\s+\S.*$", v)
    return v[:m.start()].strip() if m else v


def _lead_title(title: str | None) -> str:
    """The work's lead title with any subtitle (everything after the first colon)
    dropped — the form a library catalogue indexes a book under. OpenLibrary's
    title index returns NOTHING for a full "Main Title: Long Subtitle" string but
    finds the book on the lead alone. Also
    the fair string to score a catalogue hit against, since the record frequently
    stores only the short title."""
    return (title or "").split(":", 1)[0].strip()


def _surname(name: str) -> str:
    folded = _fold(name)
    if "," in folded:
        folded = folded.split(",", 1)[0]
    toks = re.findall(r"[a-z]{2,}", folded)
    return toks[-1] if toks else ""


def _author_overlap(ref_authors: list[str], cand_authors: list[str]) -> bool:
    ref = {_surname(a) for a in ref_authors if _surname(a)}
    cand = {_surname(a) for a in cand_authors if _surname(a)}
    return bool(ref & cand)


# --------------------------------------------------------------------------- #
# API clients (free public REST — OpenAlex + Crossref + Semantic Scholar)
# --------------------------------------------------------------------------- #
_MAILTO = os.environ.get("REVIEWER2_CONTACT_EMAIL", "").strip()
_UA = (f"reviewer2-refcheck (mailto:{_MAILTO})" if _MAILTO
       else "reviewer2-refcheck (+https://github.com/isitcredible/reviewer2)")
_TIMEOUT = 20
_CROSSREF = "https://api.crossref.org/works"
_OPENALEX = "https://api.openalex.org/works"
_S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
_TITLE_OK = 0.82
_TITLE_MAYBE = 0.60

# Each host gets its own courtesy throttle (one request per interval). They are
# independent services, so a serial background thread can interleave them.
_THROTTLES: dict[str, list] = {}

# Per-run tally of lookups that returned NO data — an HTTP error or exhausted
# retries, NOT a 200-with-empty-results. A degraded index fails silently here
# (OpenAlex rate-limiting under load is the classic), and a silent failure looks
# exactly like "this work does not exist", quietly dumping real books into the
# flag list. The orchestrator logs this at the end so a bad-index run is
# diagnosable in seconds instead of by autopsy. Reset per run.
_INDEX_FAILURES: dict[str, int] = {}


def _throttle(host: str, interval: float) -> None:
    last = _THROTTLES.setdefault(host, [0.0])
    wait = interval - (time.monotonic() - last[0])
    if wait > 0:
        time.sleep(wait)
    last[0] = time.monotonic()


def _safe_get(url: str, params: dict, host: str, interval: float,
              headers: dict | None = None) -> dict | None:
    hdrs = {"User-Agent": _UA}
    if headers:
        hdrs.update(headers)
    params = {k: v for k, v in params.items() if v != ""}  # no empty mailto
    for attempt in range(4):
        _throttle(host, interval)
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=_TIMEOUT)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                _INDEX_FAILURES[host] = _INDEX_FAILURES.get(host, 0) + 1
                return None
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
    _INDEX_FAILURES[host] = _INDEX_FAILURES.get(host, 0) + 1
    return None


def _crossref_item_to_candidate(it: dict) -> Candidate:
    title = (it.get("title") or [""])[0]
    authors = [a.get("family", "") for a in it.get("author", []) if a.get("family")]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (it.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    venue = (it.get("container-title") or [""])[0] or None
    return Candidate(title, authors, year, it.get("DOI"), venue, "crossref")


def crossref_lookup(ref: Reference, rows: int = 5) -> list[Candidate]:
    bib = ref.raw if (ref.raw and len(ref.raw.split()) > 6) else (ref.title or "")
    params = {"rows": rows, "mailto": _MAILTO, "query.bibliographic": bib}
    if ref.authors:
        params["query.author"] = " ".join(_surname(a) for a in ref.authors)
    data = _safe_get(_CROSSREF, params, "crossref", 1.0)
    if not data:
        return []
    items = data.get("message", {}).get("items", [])
    return [_crossref_item_to_candidate(it) for it in items]


def crossref_by_doi(doi: str) -> Candidate | None:
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    _throttle("crossref", 1.0)
    try:
        r = requests.get(f"{_CROSSREF}/{doi}", headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        return _crossref_item_to_candidate(r.json().get("message", {}))
    except ValueError:
        return None


def _openalex_item_to_candidate(it: dict) -> Candidate:
    authors = [(a.get("author") or {}).get("display_name", "") for a in it.get("authorships", [])]
    authors = [a for a in authors if a]
    doi = it.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/")
    venue = ((it.get("primary_location") or {}).get("source") or {}).get("display_name")
    return Candidate(it.get("display_name") or "", authors, it.get("publication_year"), doi, venue, "openalex")


def openalex_lookup(ref: Reference, rows: int = 5) -> list[Candidate]:
    query = re.sub(r"[^\w\s'-]", " ", ref.title or ref.raw)
    query = " ".join(re.sub(r"\s+", " ", query).strip().split()[:12])
    params = {"search": query, "per-page": rows, "mailto": _MAILTO}
    data = _safe_get(_OPENALEX, params, "openalex", 1.0)
    if not data:
        return []
    return [_openalex_item_to_candidate(it) for it in data.get("results", [])]


def _s2_item_to_candidate(it: dict) -> Candidate:
    authors = [a.get("name", "") for a in (it.get("authors") or []) if a.get("name")]
    doi = (it.get("externalIds") or {}).get("DOI")
    return Candidate(it.get("title") or "", authors, it.get("year"),
                     doi, it.get("venue") or None, "semanticscholar")


def semanticscholar_lookup(ref: Reference, rows: int = 5) -> list[Candidate]:
    """Semantic Scholar relevance search. Covers book chapters and obscure
    journals that OpenAlex and Crossref under-index. No key required; an
    optional SEMANTIC_SCHOLAR_API_KEY lifts the rate limit."""
    # The keyless shared pool is saturated and 429s on the first request, while
    # its retry backoff burns ~15s per reference for nothing. So SS only runs
    # when a (free) API key is configured; without one the grounded web pass is
    # the third source. Set SEMANTIC_SCHOLAR_API_KEY to activate.
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not key:
        return []
    query = re.sub(r"[^\w\s'-]", " ", ref.title or ref.raw)
    query = " ".join(re.sub(r"\s+", " ", query).strip().split()[:20])
    if not query:
        return []
    params = {"query": query, "fields": "title,year,authors,venue,externalIds", "limit": rows}
    data = _safe_get(_S2, params, "semanticscholar", 1.2, headers={"x-api-key": key})
    if not data:
        return []
    return [_s2_item_to_candidate(it) for it in data.get("data", [])]


_BOOK_TYPES_CROSSREF = {"book", "monograph", "edited-book", "reference-book",
                        "book-set", "book-series"}


def _volume_in_record(volume_title: str, ref_year: int | None) -> bool:
    """True when a chapter's containing VOLUME is catalogued as a book in the
    scholarly indexes, even if the chapter's own entry is not. Very recent edited
    volumes — and the chapters inside them — routinely lag the article indexes and
    the library catalogues by a year or two: the chapter is unlocatable on its own
    title, yet the book it sits in is already a record. Corroborating the volume
    against an INDEPENDENT index (never a model's say-so) confirms the chapter
    without the echo risk that grounding carries. A fabricated chapter cites a
    volume that is simply not there.

    Match bar: a book-type record whose title strongly matches the cited volume
    (>= _TITLE_OK), or moderately matches it (>= _TITLE_MAYBE) with the year
    agreeing within one. The year-backed moderate case covers the common mismatch
    where the index stores the main title and the citation prints title + subtitle
    (or vice versa)."""
    vt = (volume_title or "").strip()
    if len(_norm(vt).split()) < 3:
        return False

    def _ok(cand_title: str | None, cand_year) -> bool:
        sim = _title_sim(vt, cand_title)
        if sim >= _TITLE_OK:
            return True
        return bool(sim >= _TITLE_MAYBE and ref_year and cand_year
                    and abs(ref_year - int(cand_year)) <= 1)

    # OpenAlex, restricted to books.
    data = _safe_get(_OPENALEX, {"search": " ".join(vt.split()[:12]), "per-page": 5,
                                 "mailto": _MAILTO, "filter": "type:book"}, "openalex", 1.0)
    for it in (data or {}).get("results", []):
        if _ok(it.get("display_name"), it.get("publication_year")):
            return True

    # Crossref, restricted to book-like types.
    data = _safe_get(_CROSSREF, {"query.bibliographic": vt, "rows": 5, "mailto": _MAILTO},
                     "crossref", 1.0)
    for it in (data or {}).get("message", {}).get("items", []):
        if it.get("type") not in _BOOK_TYPES_CROSSREF:
            continue
        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            parts = (it.get(key) or {}).get("date-parts") or []
            if parts and parts[0] and parts[0][0]:
                year = int(parts[0][0])
                break
        if _ok((it.get("title") or [""])[0], year):
            return True

    # OpenLibrary — the independent book catalogue. Edited volumes are routinely held here
    # when the article indexes carry neither the chapter nor the volume. Query the
    # lead title: the catalogue does not match a full "Title: Subtitle" string.
    for cand_title, _, cand_year in _openlibrary_search("title", _lead_title(vt)):
        if _ok(cand_title, cand_year):
            return True
    return False


# --------------------------------------------------------------------------- #
# grounded web search (Gemini + Google Search) — the residual source
# --------------------------------------------------------------------------- #
# Book chapters and obscure journals are routinely
# absent from OpenAlex/Crossref. A real web search finds them. This runs ONLY on
# references the bibliographic indexes could not confirm, so the cost is bounded
# to the stragglers. It is a confirmation tool, not a generator: a result counts
# only if it corroborates the citation's author AND year/title — never on the
# model's say-so alone (grounding can over-read a passing mention).
_GROUNDED_SYSTEM = ("You are a careful bibliographic verifier with live web access. "
                    "You confirm whether a cited work genuinely exists by searching. "
                    "You never fabricate a work, an author, or a URL.")

_GROUNDED_PROMPT = """You are part of an automated reviewer of academic texts. Your \
role is to check whether this cited work actually exists as a real publication:

CITATION:
{CITATION}

We have already searched for this work in the major scholarly databases (OpenAlex \
and Crossref) and could not find it. You must determine whether it is a genuine \
publication or not. For example, try searching for an exact match of the title. If \
you have any doubt about whether the cited work exists, output "exists": false.

Return ONE JSON object and nothing else:
  {{
    "exists": true | false,
    "title": "<the work's actual title as published, or null>",
    "authors": ["<surnames>"],
    "year": <integer or null>,
    "venue": "<journal / book / publisher, or null>",
    "url": "<a stable page for the work — publisher, journal, catalogue, or the \
book it is a chapter in — or null>"
  }}

RULES:
- Set "exists": true ONLY if you actually locate the work (a publisher or journal \
page, a library/catalogue record, or the edited volume it is a chapter in). A book \
chapter is confirmed if you find the book and it contains that chapter.
- A mere passing mention or citation-by-others is NOT confirmation. If you cannot \
locate the work itself, set "exists": false.
- Do NOT invent a URL, a title, or authors. Leave a field null if unknown.
- Output ONLY the JSON object. No prose, no markdown fences."""


_OPENLIBRARY = "https://openlibrary.org/search.json"


def _openlibrary_search(field: str, value: str,
                        author: str | None = None) -> list[tuple[str, list[str], int | None]]:
    """OpenLibrary search → [(title, [author names], first_publish_year)]. An
    independent library catalogue (Internet Archive), free and key-less, with deep
    coverage of books and edited volumes — the works the scholarly article indexes
    under-cover. An optional ``author`` narrows the search — needed when a generic
    one-word lead title ("Harvests") returns a page of unrelated books."""
    toks = _norm(value).split()
    # Reject only genuinely sparse queries. A single DISTINCTIVE word is a valid
    # book title ("Harvests", "1848", "Bimetallism") and the lead-title query
    # routinely reduces to one — so allow a lone token of >= 4 chars, block stop
    # words ("a", "of") and noise.
    if not toks or (len(toks) == 1 and len(toks[0]) < 4):
        return []
    params = {field: value, "fields": "title,author_name,first_publish_year", "limit": 5}
    if author:
        params["author"] = author
    data = _safe_get(_OPENLIBRARY, params, "openlibrary", 1.0)
    if not data:
        return []
    out: list[tuple[str, list[str], int | None]] = []
    for d in data.get("docs", []):
        y = d.get("first_publish_year")
        out.append((d.get("title") or "", d.get("author_name") or [],
                    int(y) if isinstance(y, int) else None))
    return out


def _catalogue_corroborates(ref: Reference, claim: dict) -> bool:
    """Independent existence check for a grounded confirmation. The model saying
    "yes" is not evidence: on its own it confirms fabricated works. Instead we
    verify the work against OpenLibrary, keyed on its identity:

      • BOOK  — the work's own title must be catalogued WITH an overlapping author
        surname (title alone can collide; a fabricated title is simply absent).
      • CHAPTER — chapter authorship is not catalogued, so we corroborate that the
        containing volume (the venue) is a real book; a wholly fabricated chapter
        cites a venue that does not exist.

    A fabricated work clears neither bar; a real book / edited volume clears one."""
    authors = [a for a in (claim.get("authors") or []) if a] or ref.authors
    work_title = claim.get("title") or ref.title
    if work_title:
        for cand_title, cand_auth, _ in _openlibrary_search("title", work_title):
            if _title_sim(work_title, cand_title) >= _TITLE_MAYBE and (
                    not authors or _author_overlap(authors, cand_auth)):
                return True
    if (ref.type or "").strip().lower() == "chapter" and ref.venue:
        for cand_title, _, _ in _openlibrary_search("title", ref.venue):
            if _title_sim(ref.venue, cand_title) >= _TITLE_OK:
                return True
    return False


def openlibrary_lookup(ref: Reference) -> list[Candidate]:
    """OpenLibrary as a PRIMARY lookup for the works the article indexes drop:
    scholarly books and the edited volumes that hold chapters. OpenAlex and
    Crossref are near-complete for journal articles and patchy for monographs, so
    a real book routinely has NO usable record there and the only candidate they return is an
    unrelated best-of-a-bad-lot collision. OpenLibrary is the independent book
    catalogue that holds it. Deterministic and echo-proof (a catalogue record, not
    a model's say-so), so unlike the grounding pass it cannot be talked into a
    confirmation. Scoped to NON-article types: journals are fully covered by the
    article indexes, and letting a book catalogue confirm an article would open a
    false-confirm path. Returns candidates for the shared corroboration gate — a
    surname or year must still agree before any of these counts as a match."""
    if (ref.type or "").strip().lower() == "article":
        return []
    title = ref.title or ref.raw
    if not title:
        return []
    lead = _lead_title(title)
    author = _surname(ref.authors[0]) if ref.authors else None
    # Lead title first (the catalogue does not match a full "Title: Subtitle"
    # string); narrow by author to disambiguate a generic one-word lead, then a
    # title-only retry so a catalogue author-name variant ("Macdonald" vs the
    # cited "MacDonald") does not lose the book; then the full title as a last resort.
    attempts = [(lead, author), (lead, None)]
    if lead != title:
        attempts.append((title, None))
    for q, au in attempts:
        rows = _openlibrary_search("title", q, author=au)
        if rows:
            return [Candidate(t, a, y, None, None, "openlibrary") for (t, a, y) in rows]
    return []


# A grounding call is an optional, per-reference web lookup that should take a
# few seconds; failing it just leaves the reference flagged (a safe "worth
# checking by hand"). So bound it on every axis: a SHORT finite read timeout, a
# low retry cap, and NO infinite rate-limit retry — instead of core.py's 1800s
# PDF-stage default x 10 retries x retry-forever-on-429. Without this a single
# dead socket (or a sustained 429) stalls the whole sequential ref-check
# (observed: a run frozen ~17min on one hung grounding call). ~_GROUNDED_TIMEOUT_S
# x 2 attempts is the ceiling.
_GROUNDED_TIMEOUT_S = 90


def grounded_lookup(ref: Reference) -> Candidate | None:
    """Real web search for one reference the indexes could not confirm."""
    citation = ref.raw or " ".join(filter(None, [
        _fmt_authors(ref.authors), f"({ref.year})" if ref.year else "", ref.title, ref.venue]))
    try:
        raw = call_gemini(
            prompt=_GROUNDED_PROMPT.format(CITATION=citation), pdf_file_path=None,
            model_type="flash_2_5", temperature=0.0, thinking_budget=0,
            step="reference_grounded", use_search=True, max_output_tokens=2000,
            timeout=_GROUNDED_TIMEOUT_S, max_retries=2, retry_forever_on_rate_limit=False)
        data = _loads_tolerant(raw)
    except Exception as e:
        print(f"  ⚠  ref-check: grounded lookup failed ({e})")
        return None
    if not isinstance(data, dict) or not data.get("exists"):
        return None
    # The model's "yes" is not evidence: on its own it has false-confirmed
    # fabricated citations. Corroborate the
    # claim against an INDEPENDENT library catalogue (OpenLibrary) keyed on the
    # work's identity. A fabricated title is absent there; a real book or edited
    # volume is present. (This replaced a fetch-the-model's-URL check that broke
    # on JS publisher pages and on chapters, whose own title is not on the
    # containing-book page.) Fail closed: no catalogue corroboration → not found.
    if not _catalogue_corroborates(ref, data):
        return None
    authors = [a for a in (data.get("authors") or []) if a]
    return Candidate(
        title=data.get("title") or "",
        authors=authors,
        year=data.get("year") if isinstance(data.get("year"), int) else None,
        doi=None, venue=data.get("venue") or (data.get("url") or ""), source="web")


# --------------------------------------------------------------------------- #
# scoring / triage
# --------------------------------------------------------------------------- #
def _best_candidate(ref: Reference, cands: list[Candidate]) -> tuple[Candidate | None, float]:
    """Best title match, but a CORROBORATED candidate (author or year agrees) wins
    over a higher-similarity decoy that corroborates nothing. Without this, an
    uncorroborated near-title — a book review carrying the work's exact title, a
    reprint catalogued under a different year — shadows the real record and the
    work flags. The final confirm gate still
    applies; this only decides WHICH candidate it is applied to, and which one a
    flag reports as the near-miss."""
    target = ref.title or ref.raw
    scored = [(c, _title_sim(target, c.title)) for c in cands]
    if not scored:
        return None, 0.0
    corrob = [(c, s) for (c, s) in scored if _corroborated(ref, c)]
    best, best_sim = max(corrob or scored, key=lambda cs: cs[1])
    return best, best_sim


def _corroborated(ref: Reference, cand: Candidate) -> bool:
    """A title match alone is not enough — a different work can share a title.
    Require an author surname in common OR a year within one (online-ahead-of
    -print tolerance). Only when the citation prints neither author nor year do
    we fall back to title-only."""
    if ref.authors and cand.authors and _author_overlap(ref.authors, cand.authors):
        return True
    if ref.year and cand.year and abs(ref.year - cand.year) <= 1:
        return True
    return not (ref.year or ref.authors)


def _confirm_match(ref: Reference, best: Candidate | None, sim: float) -> LookupResult | None:
    """The two deterministic confirm rules applied to the best candidate: a strong
    title match (>= _TITLE_OK) with any corroboration, or a moderate match
    (>= _TITLE_MAYBE) with BOTH author and year agreeing — the book/chapter case
    where the indexed title carries a subtitle the citation omits. Returns a
    CONFIRMED result or None."""
    if best and sim >= _TITLE_OK and _corroborated(ref, best):
        return LookupResult(ref, "CONFIRMED", f"located ({best.source})", sim, best)
    if (best and sim >= _TITLE_MAYBE and ref.authors and best.authors
            and _author_overlap(ref.authors, best.authors)
            and ref.year and best.year and abs(ref.year - best.year) <= 1):
        return LookupResult(ref, "CONFIRMED", f"located ({best.source})", sim, best)
    return None


def check_reference(ref: Reference, use_grounding: bool = True) -> LookupResult:
    # --- A printed DOI is a POSITIVE shortcut, never a veto. If it resolves and
    # the record corroborates (title OR author OR year), confirm fast. But a DOI
    # that fails to resolve, or resolves to a record with a truncated/different
    # title, does NOT prove the work is absent — the work is often perfectly
    # findable by title. So we remember the DOI problem and FALL THROUGH to the
    # title/author/year search and grounding, flagging only if those also fail. ---
    doi_note = None
    if ref.doi:
        cand = crossref_by_doi(ref.doi)
        if cand is not None:
            sim = _title_sim(ref.title, cand.title) if ref.title else 1.0
            if (not ref.title) or sim >= _TITLE_MAYBE or _corroborated(ref, cand):
                return LookupResult(ref, "CONFIRMED", "the printed DOI resolves", sim, cand)
            doi_note = "the printed DOI resolves to a record with a different title"
        else:
            doi_note = "the printed DOI does not resolve"

    # --- Fuzzy case: gather candidates from all three indexes and take the best
    # corroborated match. No corroboration anywhere → "could not be confirmed".
    # We NEVER infer a miscite from a fuzzy near-title; a different work with a
    # similar title is simply not a confirmation. ---
    cands = openalex_lookup(ref) + crossref_lookup(ref) + semanticscholar_lookup(ref)
    best, sim = _best_candidate(ref, cands)
    hit = _confirm_match(ref, best, sim)
    if hit:
        return hit

    # --- Book-catalogue fallback. OpenAlex + Crossref are near-complete for
    # journal articles and patchy for monographs: a real scholarly book often has
    # NO usable record there, and the candidate they DO return is an unrelated
    # best-of-a-bad-lot collision (an unrelated book sharing only the year, or
    # the same author's earlier book).
    # OpenLibrary is the independent book catalogue that actually holds them.
    # Deterministic and echo-proof — it confirms books here, before any grounding
    # call. Scoped to non-article types inside openlibrary_lookup, and still
    # subject to the same author/year corroboration gate. ---
    ol = openlibrary_lookup(ref)
    if ol:
        best, sim = _best_candidate(ref, cands + ol)
        hit = _confirm_match(ref, best, sim)
        if hit:
            return hit
        # A catalogue commonly stores a book under its SHORT title, which sinks
        # the full-title similarity even for the right record. For a book, a
        # lead-title match (subtitle stripped both sides) is a confirmation when
        # the author corroborates — and we REQUIRE an author match whenever the
        # citation prints authors, so a same-title/same-year book by a DIFFERENT
        # author is never confirmed; year-only corroboration is allowed only for a
        # corporate/author-less cite. Report the full-title similarity so a reader
        # sees the honest score.
        lead = _lead_title(ref.title or ref.raw)
        for c in ol:
            if _title_sim(lead, _lead_title(c.title)) < _TITLE_OK:
                continue
            author_ok = bool(ref.authors and c.authors and _author_overlap(ref.authors, c.authors))
            year_ok = bool(ref.year and c.year and abs(ref.year - c.year) <= 1)
            if author_ok or (not ref.authors and year_ok):
                return LookupResult(ref, "CONFIRMED", "located (openlibrary)",
                                    _title_sim(ref.title or ref.raw, c.title), c)

    # --- A very recent edited-volume CHAPTER is routinely absent from the indexes
    # on its own title while the BOOK it sits in is already catalogued. Confirm
    # such a chapter from the independent existence of its containing volume — a
    # deterministic, echo-proof check (it corroborates against an index record,
    # not a model's say-so) that also spares a grounding call. A fabricated chapter
    # cites a volume that is not in the record either, so this does not open a new
    # false-confirm path. ---
    if (ref.type or "").strip().lower() == "chapter" and ref.venue:
        if _volume_in_record(_book_title_from_venue(ref.venue), ref.year):
            return LookupResult(ref, "CONFIRMED",
                                "located (containing volume in record)", sim, best)

    # --- Residual: a real web search for what the indexes missed. Confirms only
    # on author corroboration plus a year or title match — never on "exists"
    # alone — AND only for reference TYPES the indexes genuinely under-cover.
    # A journal ARTICLE is excluded: OpenAlex + Crossref index those venues fully,
    # so an article that BOTH miss is overwhelmingly fabricated. Letting the web
    # step "confirm" it on the model's say-so is exactly the echo that lets
    # fabricated citations through. An unconfirmable article FLAGS
    # (a safe "worth checking by hand"); it never CONFIRMS on grounding. Books,
    # chapters, reports, working papers, datasets, archival material still go
    # through grounding — backstopped by the aboutness gate in grounded_lookup. ---
    if use_grounding and (ref.type or "").strip().lower() != "article":
        wc = grounded_lookup(ref)
        if wc and ref.authors and wc.authors and _author_overlap(ref.authors, wc.authors):
            wsim = _title_sim(ref.title or ref.raw, wc.title)
            year_ok = bool(ref.year and wc.year and abs(ref.year - wc.year) <= 1)
            if wsim >= _TITLE_MAYBE or year_ok:
                return LookupResult(ref, "CONFIRMED", "located (web)", max(sim, wsim), wc)

    return LookupResult(ref, "FLAG", doi_note or "could not be confirmed", sim, best)


# --------------------------------------------------------------------------- #
# bibliography extraction + verification (Gemini Flash 2.5, reads the PDF)
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = ("You are a precise bibliography extractor. You are NOT a reviewer or "
                   "fact-checker — you only transcribe and structure what is printed. You "
                   "never judge whether a reference is real and you never invent fields.")

_EXTRACT_PROMPT = """The attached PDF is an academic paper or book. Find its \
bibliography — the section headed “References”, “Bibliography”, “Works Cited”, \
“Literature Cited”, or the like — and transcribe EVERY entry in it.

Read the document’s layout. The reference list has a clear beginning and END: \
appendices, proofs, data, footnotes, or back matter often follow it. STOP at the \
end of the reference list. Do NOT transcribe anything after it, and do NOT pull \
in-text citations or footnotes from the body as if they were entries.

Return ONE JSON object: {"references": [ ... ]}, one array element per distinct \
bibliographic entry, IN THE ORDER PRINTED. Each element:

  {
    "authors": [SURNAMES only, in citation order; [] for a corporate/government \
author — put that body in "venue"]. An author printed “Surname, Firstname” (e.g. \
“Smith, Adam” or “Keynes, John M.”) is ONE author — the surname is the part \
BEFORE the comma (“Smith”, “Keynes”); the given name is NOT a second author. Only \
add another surname when a genuinely different person is listed,
    "year": integer publication year, or null if none is printed,
    "title": the work’s title (for a chapter, the CHAPTER title); repair \
line-break hyphenation ("Indus- trial" -> "Industrial"),
    "venue": journal / book / publisher / series / issuing body, or null \
(for a chapter, the BOOK or volume it appears in),
    "doi": the DOI string ONLY if one is explicitly printed, else null,
    "type": one of "article","book","chapter","dataset","report","working_paper",\
"archival","other",
    "raw": the FULL entry exactly as printed, on one line (hyphenation repaired, \
internal line-breaks collapsed to single spaces)
  }

HARD RULES — you will be checked against the PDF:
- Do NOT invent, complete, or correct any field. If the printed entry lacks a \
year / DOI / venue, the value is null. Transcribe what is there.
- RESOLVE author dashes: many lists replace a repeated author with “—”, “———”, \
or a long rule for successive works by the same author(s). When you see this, \
fill "authors" and the leading author text in "raw" from the entry IMMEDIATELY \
above. A "raw" must never begin with a dash, comma, or “and”.
- Do NOT drop entries and do NOT merge two entries into one. The count matters.
- Do NOT treat a running head, a page number, or a body-text sentence as an entry.
- Output ONLY the JSON object. No prose, no markdown fences.

If the document has NO bibliography at all, output exactly: {"references": []}"""

_VERIFY_SYSTEM = ("You are a meticulous proofreader checking a transcription against "
                  "its source. You only report discrepancies; you never invent references.")

_VERIFY_PROMPT = """The attached PDF is an academic paper or book. Below is a JSON \
list of references that a first pass transcribed from its bibliography. Check the \
list AGAINST the actual reference list in the PDF and report only what is wrong.

Return ONE JSON object:
  {
    "missing":  [ <full records, same schema as below, for any entry that IS in \
the PDF’s reference list but is ABSENT from the list below> ],
    "spurious": [ <the exact "raw" string of any listed item below that is NOT a \
genuine bibliography entry — e.g. body text, a footnote, an in-text citation \
sentence, a running head, or appendix prose> ]
  }

Record schema for "missing": {"authors":[surnames],"year":int|null,"title":...,\
"venue":...|null,"doi":...|null,"type":...,"raw":...}.

RULES:
- Judge ONLY against the printed reference list. If an item below is a real \
bibliography entry, it is neither missing nor spurious — leave it alone.
- An item is "spurious" if it is not part of the reference list at all (the most \
common cause is text from an appendix or footnote that follows the bibliography).
- Do NOT invent references. If nothing is missing and nothing is spurious, return \
{"missing": [], "spurious": []}.
- Output ONLY the JSON object. No prose, no markdown fences.

THE TRANSCRIBED LIST:
---
{LIST}
---"""


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _loads_tolerant(s: str) -> dict:
    """Parse the model's JSON, salvaging a truncated references array (a very
    long bibliography can hit the output-token ceiling mid-array)."""
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    cut = s.rfind("}")
    if cut == -1:
        return {}
    salvaged = s[:cut + 1].rstrip().rstrip(",")
    for suffix in ("]}", "}"):
        try:
            return json.loads(salvaged + suffix)
        except json.JSONDecodeError:
            continue
    return {}


def _record_to_reference(r: dict) -> Reference:
    return Reference(
        raw=(r.get("raw") or "").strip(),
        title=(r.get("title") or None),
        authors=[a for a in (r.get("authors") or []) if a],
        year=r.get("year") if isinstance(r.get("year"), int) else None,
        venue=(r.get("venue") or None),
        doi=(r.get("doi") or None),
        type=(r.get("type") or None),
    )


_EXTRACT_ATTEMPTS = 3      # extraction retries on an empty result
_EXTRACT_RETRY_PAUSE = 2.0  # seconds between retries


def _extract_once(pdf_path: str) -> list[Reference]:
    raw = call_gemini(
        prompt=_EXTRACT_PROMPT, pdf_file_path=pdf_path,
        model_type="flash_2_5", temperature=0.0, thinking_budget=0,
        step="reference_extract", max_output_tokens=60000,
        system_instruction=_EXTRACT_SYSTEM)
    data = _loads_tolerant(raw)
    return [_record_to_reference(r) for r in data.get("references", []) if r.get("raw") or r.get("title")]


def extract_references_gemini(pdf_path: str, attempts: int = _EXTRACT_ATTEMPTS) -> list[Reference]:
    """Flash 2.5 reads the PDF and returns structured bibliography records.

    Retry-on-empty guard: the extract call occasionally returns 0 references on
    a paper that HAS a bibliography (a transient empty/garbled Flash response —
    stochastic even at temperature 0), which would otherwise be silently dropped
    as =NULL=. An immediate re-probe reliably recovers it, so retry while the
    result is empty. A genuinely bibliography-less paper just returns [] on every
    attempt (a few cheap extra calls — acceptable). If EVERY attempt raises (API
    hung / blocked / rate-limited rather than a clean empty answer), re-raise the
    last error so the caller degrades to "no Reference Check section" instead of
    falsely concluding the paper has no bibliography."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            refs = _extract_once(pdf_path)
            last_exc = None
        except Exception as e:
            last_exc = e
            refs = []
            print(f"  ⚠  ref-check: extraction call failed (attempt {attempt}/{attempts}): {e}")
        if refs:
            if attempt > 1:
                print(f"  → ref-check: extraction recovered {len(refs)} references on attempt {attempt}")
            return refs
        if attempt < attempts:
            print(f"  ⚠  ref-check: extraction returned 0 references "
                  f"(attempt {attempt}/{attempts}); retrying...")
            time.sleep(_EXTRACT_RETRY_PAUSE)
    # Exhausted. If every attempt errored, surface it; otherwise it's a genuine
    # (or persistently unrecoverable) empty bibliography.
    if last_exc is not None:
        raise last_exc
    return []


def verify_and_merge(pdf_path: str, refs: list[Reference]) -> list[Reference]:
    """Second Flash 2.5 pass: drop spurious entries, add missed ones."""
    listing = json.dumps(
        [{"authors": r.authors, "year": r.year, "title": r.title,
          "venue": r.venue, "doi": r.doi, "type": r.type, "raw": r.raw} for r in refs],
        ensure_ascii=False)
    try:
        raw = call_gemini(
            prompt=_VERIFY_PROMPT.replace("{LIST}", listing), pdf_file_path=pdf_path,
            model_type="flash_2_5", temperature=0.0, thinking_budget=512,
            step="reference_verify", max_output_tokens=30000,
            system_instruction=_VERIFY_SYSTEM)
        data = _loads_tolerant(raw)
    except Exception as e:
        print(f"  ⚠  ref-check: verify pass failed ({e}); using first-pass extraction")
        return refs

    spurious = {_norm(s) for s in (data.get("spurious") or []) if s}
    kept = [r for r in refs if _norm(r.raw) not in spurious]
    added = [_record_to_reference(r) for r in (data.get("missing") or [])
             if r.get("raw") or r.get("title")]
    # Don't re-add something already present (match on normalised raw).
    have = {_norm(r.raw) for r in kept}
    added = [r for r in added if _norm(r.raw) not in have]
    dropped, gained = len(refs) - len(kept), len(added)
    if dropped or gained:
        print(f"  → ref-check: verify pass dropped {dropped} spurious, added {gained} missed")
    return kept + added


# --------------------------------------------------------------------------- #
# triage + section writer
# --------------------------------------------------------------------------- #
def _fmt_authors(authors: list[str]) -> str:
    if len(authors) <= 1:
        return authors[0] if authors else ""
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return ", ".join(authors[:-1]) + f", and {authors[-1]}"


def _display_citation(ref: Reference) -> str:
    """Show the reference as printed. Extraction has already resolved author
    dashes, so the raw should be clean and complete; fall back to assembling
    from fields only if it is somehow empty or dash-led."""
    raw = (ref.raw or "").strip()
    if raw and raw[:1] not in "—–-," and not raw.lower().startswith("and "):
        return raw
    if ref.authors:
        head = _fmt_authors(ref.authors)
        bits = [head]
        if ref.year:
            bits.append(f"({ref.year})")
        if ref.title:
            bits.append(ref.title)
        if ref.venue:
            bits.append(ref.venue)
        rebuilt = " ".join(bits).strip()
        # If raw was dash-led, splice the recovered author head onto it.
        if raw[:1] in "—–-," or raw.lower().startswith("and "):
            return re.sub(r"^\s*(and\s+|[—–-]+\s*|,\s*)", head + " ", raw)
        return rebuilt
    return raw


# --------------------------------------------------------------------------- #
# classify the unconfirmed set (genre + whether the absence is expected)
# --------------------------------------------------------------------------- #
# Flash 2.5 (with web search for the genuinely ambiguous ones) reads each
# unconfirmed reference and judges WHAT it is and whether its absence from the
# records is EXPECTED for that kind of source. This is the judgement the writer
# used to improvise from a stock example; doing it here, per reference, lets the
# overview name the kinds actually present and keeps M + E + S = N exact.
_CLASSIFY_SYSTEM = ("You are a careful bibliographic classifier with web access. You "
                    "identify what kind of source a citation is. You never invent a work.")

_CLASSIFY_PROMPT = """These references could NOT be matched to scholarly records.
For EACH, decide two things from the citation itself — use your own knowledge;
search the web ONLY when the string is too sparse to tell what the source is:

  1. genre  — what kind of source this is.
  2. expected — whether its absence from scholarly records is EXPECTED for that genre.

EXPECTED (expected=true) — sources that are routinely not indexed:
  government / parliamentary / official institutional publications; official
  statistical series, bulletins, datasets; primary or archival documents;
  newspaper, magazine, or web items; unpublished manuscripts.
NOT EXPECTED (expected=false) — a published journal article, scholarly book, or
  book chapter. These SHOULD be locatable, so failure to find one is worth a
  human check, even if old.

Judgement, not mechanics: a 1949 institutional research report is expectedly
absent; a 2015 article in a major journal is not. Decide each on what it is.

CITATIONS:
{LIST}

Return ONE JSON object:
  {"items":[{"n":<index>,"genre":"<journal_article|scholarly_book|book_chapter|working_paper|government_or_official|statistics_or_dataset|primary_or_archival|news_or_web|unpublished_manuscript|other>","expected":true|false,"note":"<=12 words: what it is / why expected"}]}
One element per citation, same indices. Output ONLY the JSON object, no fences."""

_GENRE_KIND = {
    "government_or_official": "government and official publications",
    "statistics_or_dataset": "official statistics and datasets",
    "primary_or_archival": "primary and archival sources",
    "news_or_web": "newspaper and web sources",
    "unpublished_manuscript": "unpublished manuscripts",
    "working_paper": "working papers",
    "other": "other non-indexed sources",
}


def classify_flagged(flagged: list[dict]) -> dict[int, dict]:
    """Return {index: {"genre", "expected", "note"}} for the unconfirmed rows.
    Never raises: on failure returns {} and the caller treats every unconfirmed
    reference as worth checking by hand — the safe direction."""
    if not flagged:
        return {}
    listing = "\n".join(f'{i}. {r.get("citation", "")}' for i, r in enumerate(flagged))
    try:
        raw = call_gemini(
            prompt=_CLASSIFY_PROMPT.replace("{LIST}", listing), pdf_file_path=None,
            model_type="flash_2_5", temperature=0.0, thinking_budget=0,
            use_search=True, step="reference_classify", max_output_tokens=8000,
            system_instruction=_CLASSIFY_SYSTEM)
        data = _loads_tolerant(raw)
    except Exception as e:
        print(f"  ⚠  ref-check: classify pass failed ({e}); all unconfirmed → worth checking")
        return {}
    out: dict[int, dict] = {}
    for it in (data.get("items") or []):
        if isinstance(it, dict) and isinstance(it.get("n"), int):
            out[it["n"]] = it
    return out


# --------------------------------------------------------------------------- #
# writer payload — the split is settled upstream; the writer only renders it
# --------------------------------------------------------------------------- #
def _category_of(row: dict) -> str:
    reason = (row.get("reason") or "").lower()
    if "does not resolve" in reason:
        return "doi_no_resolve"
    if "differ" in reason:
        return "record_differs"
    return "could_not_locate"


def _writer_payload(triage: dict) -> str:
    """Three settled buckets for the writer. M + E + S == checked by construction;
    'kinds' names the expected-absence genres actually present (no stock list)."""
    flagged = [r for r in triage["references"] if r["verdict"] != "CONFIRMED"]
    expected = [r for r in flagged if r.get("expected_absence")]
    worth = [r for r in flagged if not r.get("expected_absence")]
    kinds: list[str] = []
    for r in expected:
        k = _GENRE_KIND.get(r.get("genre"), "other non-indexed sources")
        if k not in kinds:
            kinds.append(k)
    refs = []
    for r in worth:
        cat = _category_of(r)
        item = {"citation": r["citation"], "category": cat}
        if cat == "record_differs":
            item["differs_note"] = r.get("reason")
        refs.append(item)
    return json.dumps({
        "checked": triage["total"],
        "located_and_matched": triage["confirmed"],
        "expected_absences": {"count": len(expected), "kinds": kinds},
        "worth_checking": {"count": len(worth), "references": refs},
    }, indent=2, ensure_ascii=False)


def _build_triage(refs: list[Reference]) -> dict:
    rows = []
    for i, ref in enumerate(refs, 1):
        if i % 25 == 0:
            print(f"  → ref-check: looked up {i}/{len(refs)} references")
        res = check_reference(ref)
        rows.append({
            "citation": _display_citation(ref),
            "authors": ref.authors, "year": ref.year, "title": ref.title,
            "venue": ref.venue, "doi": ref.doi, "type": ref.type,
            "verdict": res.verdict, "reason": res.reason,
            "title_sim": round(res.title_sim, 2),
            "match": asdict(res.best) if res.best else None,
        })
    confirmed = sum(r["verdict"] == "CONFIRMED" for r in rows)
    return {"total": len(rows), "confirmed": confirmed,
            "flagged": len(rows) - confirmed, "references": rows}


def _italicize_publications(section: str, rows: list[dict]) -> str:
    """Put each listed work's publication name in markdown italics, the way a
    reference list prints it: the journal (article/chapter) or the book's own
    title (book). Done as a deterministic post-pass on the composed section, NOT
    in the writer prompt, so it never depends on the model preserving emphasis
    markers (the writer is told never to emit '*'). Only entries actually shown —
    the worth-checking ones — are touched; confirmed and expected-absence entries
    are not in the list, so their strings simply don't occur."""
    pubs = []
    for r in rows:
        if r.get("verdict") == "CONFIRMED" or r.get("expected_absence"):
            continue
        rtype = (r.get("type") or "").strip().lower()
        if rtype == "book":
            pub = r.get("title")
        elif rtype == "chapter":
            # For a chapter the publication shown in italics is the CONTAINING
            # BOOK's title only — not the "Place: Publisher" tail the extractor
            # packs into ``venue`` (italicising that would print the place and
            # publisher as if they were the book title).
            pub = _book_title_from_venue(r.get("venue"))
        else:
            pub = r.get("venue")
        if pub and len(pub.strip()) >= 3:
            pubs.append(pub.strip())
    # Longest first: a venue that contains a shorter string is wrapped whole, and
    # the lookarounds then stop the shorter one from re-wrapping inside it.
    for pub in sorted(set(pubs), key=len, reverse=True):
        pat = re.compile(r"(?<!\*)" + re.escape(pub) + r"(?!\*)")
        section = pat.sub(lambda m, p=pub: f"*{p}*", section)
    return section


def write_section(triage: dict, metadata: dict, output_dir: str) -> str:
    """Gemini-Pro section writer (prompts/reference_check.txt, report voice)."""
    prompt = load_prompt("prompts/reference_check.txt")
    prompt = prompt.replace("{{CITATION}}", get_citation_block(metadata))
    prompt = prompt.replace("{{REFERENCE_TRIAGE}}", _writer_payload(triage))
    return call_gemini(prompt, None, model_type="pro_3_1", temperature=0.3,
                       thinking_level="low", step="reference_check").strip()


# --------------------------------------------------------------------------- #
# orchestrator (called on a background thread by pipeline.py)
# --------------------------------------------------------------------------- #
def run_reference_check(pdf_path: str, output_dir: str, metadata: dict) -> str | None:
    """Full reference check → writes reference_check.txt + reference_triage.json.
    Returns the section text, or None when there is no bibliography to check.
    Never raises — a failure degrades to no Reference Check section."""
    try:
        t0 = time.monotonic()
        _INDEX_FAILURES.clear()
        refs = extract_references_gemini(pdf_path)
        if not refs:
            print("  → ref-check: no bibliography extracted; emitting =NULL= (no References section)")
            save_output("=NULL=", "reference_check.txt", output_dir)
            return "=NULL="
        print(f"  → ref-check: extracted {len(refs)} references; verifying against the PDF...")
        refs = verify_and_merge(pdf_path, refs)
        print(f"  → ref-check: {len(refs)} references to look up...")
        triage = _build_triage(refs)
        # Make a degraded index loud. A run where OpenAlex (or Crossref) returned
        # no data on a large share of lookups produces an inflated flag list that
        # is NOT a property of the bibliography — surface it so it is not mistaken
        # for fabricated references.
        deg = {h: n for h, n in _INDEX_FAILURES.items() if n}
        if deg:
            worst = max(deg.values())
            level = "⚠ " if worst >= max(5, len(refs) // 4) else "→ "
            summary = ", ".join(f"{h}={n}" for h, n in sorted(deg.items()))
            print(f"  {level} ref-check: lookups returned no data this run ({summary}); "
                  f"a degraded index inflates the unconfirmed count")
        # Classify the unconfirmed set: genre + whether the absence is expected.
        # The annotations drive the writer's three-bucket overview.
        flagged = [r for r in triage["references"] if r["verdict"] != "CONFIRMED"]
        if flagged:
            print(f"  → ref-check: classifying {len(flagged)} unconfirmed references...")
            cls = classify_flagged(flagged)
            for i, r in enumerate(flagged):
                c = cls.get(i) or {}
                r["genre"] = c.get("genre")
                r["expected_absence"] = bool(c.get("expected"))
        with open(os.path.join(output_dir, "reference_triage.json"), "w", encoding="utf-8") as f:
            json.dump(triage, f, indent=2, ensure_ascii=False)
        section = write_section(triage, metadata, output_dir)
        section = _italicize_publications(section, triage["references"])
        save_output(section, "reference_check.txt", output_dir)
        print(f"  ✓ ref-check: {triage['confirmed']}/{triage['total']} matched, "
              f"{triage['flagged']} flagged ({time.monotonic() - t0:.0f}s)")
        return section
    except Exception as e:
        print(f"  ⚠  ref-check: unexpected failure ({e}); no Reference Check section")
        return None
